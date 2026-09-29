"""Tests for herding.py (LOG 2026-09-29): the situations against the logged turns, the prompts of
every condition (only the report changes), the jobs, the outcome classes, resume, the paired
analysis, and the sampling-key hook in village.py. Reads runs/copying/play_g0; no model weights
(the tokenizer is read from the local Hugging Face cache, offline)."""

import json
import os
import random

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import herding  # noqa: E402
import pond  # noqa: E402
import village  # noqa: E402
from herding import CONDITIONS, GOLDEN_NONE, LEVELS, N_LINES, NONE  # noqa: E402
from pond import Reply, Request, Turn  # noqa: E402


@pytest.fixture(scope="module")
def source():
    pools = {v: village._load_pool(herding.SOURCE / v) for v in herding.SOURCE_VILLAGES}
    drawn = herding.draw_situations(pools, herding.game_template())
    turns = {(v, ep.episode, ep.agent, t.round): t for v, pool in pools.items() for ep in pool for t in ep.turns}
    return pools, drawn, turns


def all_situations(drawn):
    return drawn["main"] + drawn["pilot"]


# ----------------------------------------------------------------------------
# the key hook
# ----------------------------------------------------------------------------


def test_request_key_is_the_games_key():
    req = Request("v", 0, 0, 1, 3, 4, "Fisher E", "", "", 10)
    assert village.request_key(req) == "0/g0/e1/r3/sample/4" == village.sample_key(0, 0, 1, 3, 4)
    import inspect

    assert inspect.signature(village.ModelPlayers.__init__).parameters["key_fn"].default is village.request_key


# ----------------------------------------------------------------------------
# situations
# ----------------------------------------------------------------------------


def test_situation_counts_and_disjoint_agent_days(source):
    _, drawn, _ = source
    assert len(drawn["main"]) == herding.PHASES["main"] == 100
    assert len(drawn["pilot"]) == herding.PHASES["pilot"] == 200
    days = [(s["village"], s["day"], s["agent"]) for s in all_situations(drawn)]
    assert len(set(days)) == len(days)
    assert len({s["sid"] for s in all_situations(drawn)}) == 300
    assert len({s["key"] for s in all_situations(drawn)}) == 300


def test_situations_are_the_logged_turns(source):
    _, drawn, turns = source
    template = herding.game_template()
    for s in all_situations(drawn):
        t = turns[(s["village"], s["day"], s["agent"], s["round"])]
        assert s["round"] >= 2 and s["coins"] >= 1 and t.coins_before == s["coins"]
        lines = t.observation.split("\n")
        assert lines[: len(s["kept"])] == s["kept"]
        assert lines[len(s["kept"])].startswith("Golden fish caught last round: ")
        assert s["kept"][-2] == f"Round {s['round']}. You have {pond._coins(s['coins'])}."
        assert s["kept"][-1].startswith("Your rounds so far: ")
        assert len(s["kept"]) == (3 if s["day"] > 0 else 2)
        assert t.system == template.format(name=s["name"]) == herding.s_system(s)
        assert sorted(s["names"]) == sorted(pond.agent_name(i) for i in range(8) if pond.agent_name(i) != s["name"])
        assert sorted(s["cast_order"]) == list(range(N_LINES))


def test_messages_come_from_the_previous_round(source):
    _, drawn, turns = source
    for s in all_situations(drawn):
        assert s["msg_round"] == s["round"] - 1
        for field, action in (("caster_msgs", "cast"), ("fisher_msgs", "fish")):
            msgs = s[field]
            assert len(msgs) == N_LINES
            ids = [(m["village"], m["day"], m["agent"]) for m in msgs]
            assert len(set(ids)) == N_LINES  # without replacement
            assert (s["village"], s["day"], s["agent"]) not in ids  # own agent-day excluded
            for m in msgs:
                t = turns[(m["village"], m["day"], m["agent"], s["round"] - 1)]
                assert not t.failed and t.reply.action == action and t.reply.message == m["text"] and m["text"].strip()
                assert "\n" not in m["text"]
                assert m["mentions"] == bool(herding.MENTIONS.search(m["text"]))


def test_draw_is_deterministic(source):
    pools, drawn, _ = source
    again = herding.draw_situations(pools, herding.game_template())
    assert again == drawn


# ----------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------


def report(s, cond, m):
    obs = herding.observation(s, cond, m)
    head = s["kept"] + [GOLDEN_NONE]
    lines = obs.split("\n")
    assert lines[: len(head)] == head, "everything before the report is the situation's"
    return lines[len(head) :]


def test_only_the_report_changes(source):
    _, drawn, _ = source
    for s in drawn["main"][:20]:
        assert report(s, NONE, None) == []
        for m in LEVELS:
            peers = report(s, "peers", m)
            assert peers[0] == herding.PEERS_HEADER and len(peers) == 1 + N_LINES
            body = peers[1:]
            assert [l.split(": ")[0] for l in body] == [f"- {nm}" for nm in s["names"]]
            words = [l.rsplit(": ", 1)[1] for l in body]
            assert set(words) <= {"cast", "fish"} and words.count("cast") == m
            assert {k for k, w in enumerate(words) if w == "cast"} == set(s["cast_order"][:m])
            assert report(s, "bots", m) == [herding.BOTS_LINE] + peers
            assert report(s, "independence", m) == [herding.INDEPENDENCE_LINE] + peers
            printer = report(s, "printer", m)
            assert printer[0] == herding.PRINTER_HEADER
            assert printer[1:] == [f"- Line {k + 1}: {w}" for k, w in enumerate(words)]
            real = report(s, "real", m)
            assert real[0] == herding.MESSAGES_HEADER and len(real) == 1 + N_LINES
            for k, line in enumerate(real[1:]):
                src = s["caster_msgs"][k] if k in set(s["cast_order"][:m]) else s["fisher_msgs"][k]
                assert line == f"- {s['names'][k]}: {src['text']}"
            for rep in (peers, printer, real):
                assert len(rep) >= N_LINES and all(l.startswith("- ") for l in rep[-N_LINES:]), "every report ends with its 7 lines"


def test_levels_are_nested(source):
    _, drawn, _ = source
    for s in drawn["main"][:20]:
        prev = set()
        for m in LEVELS:
            now = herding.cast_lines(s, m)
            assert prev <= now and len(now) == m
            prev = now


def test_messages_header_is_the_games():
    obs = pond.build_observation(round=2, coins=10, own_turns=[], yesterday=None, see_events=True, see_messages=True,
                                 winners=[], messages=[("Fisher B", "hi")])
    assert herding.MESSAGES_HEADER in obs.split("\n")
    assert herding.GOLDEN_NONE in obs.split("\n")


def test_prompts_have_no_risk_words_beyond_the_games():
    # the added framing sentences use none of the words the system prompt avoids
    for text in (herding.PEERS_HEADER, herding.PRINTER_HEADER, herding.BOTS_LINE, herding.INDEPENDENCE_LINE):
        assert not any(w in text.lower() for w in village.RISK_WORDS)


def test_bad_condition_or_level(source):
    _, drawn, _ = source
    s = drawn["main"][0]
    with pytest.raises(ValueError):
        herding.observation(s, "peers", 8)
    with pytest.raises(ValueError):
        herding.observation(s, NONE, 0)
    with pytest.raises(ValueError):
        herding.observation(s, "gossip", 3)


def test_prefix_is_a_token_prefix_for_every_condition(source):
    from mlx_lm.utils import load_tokenizer

    _, drawn, _ = source
    tok = load_tokenizer(pond.MODEL)
    for s in drawn["main"][:3] + drawn["pilot"][:2]:
        system = herding.s_system(s)
        prefix = list(tok.encode(village.prefix_text(system), add_special_tokens=False))
        for cond, m in [(c, m) for c in CONDITIONS for m in (0, 3, 7)] + [(NONE, None)]:
            obs = herding.observation(s, cond, m)
            full = list(tok.apply_chat_template(pond.prompt_messages(system, obs), add_generation_prompt=True, return_dict=False))
            assert full[: len(prefix)] == prefix
            assert tok.decode(full[len(prefix) :]) == obs + "<|im_end|>\n<|im_start|>assistant\n"


# ----------------------------------------------------------------------------
# jobs, outcomes, files
# ----------------------------------------------------------------------------


def test_jobs(source):
    _, drawn, _ = source
    main = herding.jobs(drawn["main"], "main")
    assert len(main) == 4100 == len({j[0] for j in main})
    per = {}
    for job, i, c, m in main:
        per.setdefault(i, set()).add((c, m))
    assert all(v == set(herding.PHASE_JOBS["main"]) for v in per.values()) and len(per) == 100
    assert main == herding.jobs(drawn["main"], "main")  # fixed order
    assert main != sorted(main)
    pilot = herding.jobs(drawn["pilot"], "pilot")
    assert len(pilot) == 400 and {(c, m) for _, _, c, m in pilot} == {("peers", 0), ("peers", 7)}


def test_classify_agrees_with_the_game():
    cases = [("reasoning: r\nmessage: m\naction: cast\nstake: 3", 2, dict(failed=True, real=False, attempt=True, intent=True)),
             ("reasoning: r\nmessage: m\naction: cast\nstake: 3", 5, dict(failed=False, real=True, attempt=False, intent=True)),
             ("reasoning: r\nmessage: m\naction: cast\nstake: 9", 20, dict(failed=True, real=False, attempt=True, intent=True)),
             ("reasoning: r\nmessage: m\naction: fish\nstake: 0", 5, dict(failed=False, real=False, attempt=False, intent=False)),
             ("I would rather not say.", 5, dict(failed=True, real=False, attempt=False, intent=False))]
    for text, coins, want in cases:
        p = village.parse_reply(text, coins, herding.MAX_STAKE)
        got = herding.classify(p.parse, p.reply.action if p.reply else None)
        assert got == want, (text, coins, got)
        reply = p.reply or pond.FAILED_REPLY
        t = Turn(0, 2, "", "", reply, coins, coins, None, 0.5, text, p.parse, p.reply is None, 0)
        assert got["attempt"] == village.is_attempt(t) and got["real"] == village.is_cast(t)


def test_read_rows_drops_a_truncated_final_line(tmp_path):
    p = tmp_path / "replies.jsonl"
    a, b = json.dumps({"job": "a"}) + "\n", json.dumps({"job": "b"}) + "\n"
    p.write_text(a + b + '{"job": "c", "ra')
    assert [r["job"] for r in herding.read_rows(p)] == ["a", "b"]
    assert p.read_text() == a + b + '{"job": "c", "ra'  # analyse reads without writing
    assert [r["job"] for r in herding.read_rows(p, repair=True)] == ["a", "b"]
    assert p.read_text() == a + b
    # a complete row cut before its newline is a truncated row too: dropped, then the next append reads back
    p.write_text(a + json.dumps({"job": "b"}))
    assert [r["job"] for r in herding.read_rows(p, repair=True)] == ["a"]
    with open(p, "a") as fid:
        fid.write(b + json.dumps({"job": "c"}) + "\n")
    assert [r["job"] for r in herding.read_rows(p)] == ["a", "b", "c"]
    # a cut inside a multi-byte character
    p.write_bytes((a + json.dumps({"job": "d", "m": "中"}, ensure_ascii=False)).encode()[:-5])
    assert [r["job"] for r in herding.read_rows(p, repair=True)] == ["a"]
    p.write_text(a + "{bad\n" + b)
    with pytest.raises(ValueError):
        herding.read_rows(p)
    assert herding.read_rows(tmp_path / "missing.jsonl") == []


def test_run_config_carries_the_situations_hash(tmp_path):
    f = tmp_path / "situations.jsonl"
    f.write_text("{}\n")
    a = herding.run_config("pilot", f)
    f.write_text("{} \n")
    b = herding.run_config("pilot", f)
    assert a != b and a["jobs"] == 400 and a["temperature"] == 1.0 and a["max_tokens"] == 256 and a["adapter"] is None
    assert herding.run_config("main", f)["jobs"] == 4100


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------


def rows_from(fn, n_sit=50, conds=CONDITIONS, none=True):
    rows = []
    for i in range(n_sit):
        sid = f"main-{i:03d}"
        for c in conds:
            for m in LEVELS:
                y = fn(i, c, m)
                rows.append({"sid": sid, "condition": c, "m": m, "intent": y, "real": y})
        if none:
            rows.append({"sid": sid, "condition": NONE, "m": None, "intent": fn(i, NONE, None), "real": fn(i, NONE, None)})
    return rows


def test_analysis_recovers_known_effects():
    def fn(i, c, m):
        if c == NONE:
            return i % 2 == 0
        if c == "peers":
            return m >= 4
        return False

    res = herding.analyse(rows_from(fn), "intent", n_boot=50)
    peers = res["conditions"]["peers"]
    assert peers["E"]["est"] == 1.0 and abs(peers["7b"]["est"] - 56 / 42) < 1e-12
    assert peers["rates"][3]["est"] == 0.0 and peers["rates"][4]["est"] == 1.0
    pr = res["vs_peers"]["printer"]
    assert pr["E"]["est"] == -1.0 and abs(pr["7b"]["est"] + 56 / 42) < 1e-12 and pr["ratio_7b"]["est"] == 0.0
    assert abs(pr["level"]["est"] + 0.5) < 1e-12
    assert res["none"]["est"] == 0.5
    assert res["vs_none"]["peers"]["m7"]["est"] == 0.5 and res["vs_none"]["peers"]["m0"]["est"] == -0.5
    assert res["printer_vs_bots"]["7b"]["est"] == 0.0


def test_mean_of_situation_slopes_is_the_pooled_slope_in_a_balanced_design():
    rng = random.Random(1)
    rows = rows_from(lambda i, c, m: rng.random() < 0.1 + 0.1 * m, n_sit=40, conds=("peers",), none=False)
    res = herding.analyse(rows, "intent", n_boot=20)
    xs = [r["m"] for r in rows]
    ys = [1.0 if r["intent"] else 0.0 for r in rows]
    xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
    pooled = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
    assert abs(res["conditions"]["peers"]["7b"]["est"] - 7 * pooled) < 1e-12


def test_real_decomposition_separates_caster_lines_from_mentions(source):
    _, drawn, _ = source
    sits = {s["sid"]: s for s in drawn["main"]}
    # a world where only mentioning lines matter: y = 1 if at least 3 drawn lines mention cast/golden
    rows = []
    for sid, s in sits.items():
        for m in LEVELS:
            k = herding.real_counts(s, m)["mentions"]
            rows.append({"sid": sid, "condition": "real", "m": m, "intent": k >= 3, "real": k >= 3})
    out = herding.real_decomposition(rows, sits, "intent", n_boot=50)
    assert set(out["by_level"]) == set(LEVELS)
    assert out["by_level"][7]["mentions"] > out["by_level"][0]["mentions"]
    fe = out["fe_regression"]
    assert fe["per_mention_line"]["est"] > 0.1 and fe["per_mention_line"]["est"] > fe["per_caster_line"]["est"]


def test_resolving_power_on_a_pilot():
    rng = random.Random(2)
    rows = []
    for i in range(200):
        for m, p in ((0, 0.1), (7, 0.8)):
            y = rng.random() < p
            rows.append({"sid": f"pilot-{i:03d}", "condition": "peers", "m": m, "intent": y, "real": y})
    rp = herding.resolving_power(rows, 100)
    assert 0.03 < rp["se_peers_E_main"] < 0.07
    lo, hi = rp["se_dE_main"]
    assert 0 < lo < hi and rp["line_3se"] == [3 * lo, 3 * hi]
    assert abs(rp["p0"] - 0.1) < 0.06 and abs(rp["p7"] - 0.8) < 0.08


# ----------------------------------------------------------------------------
# run plumbing: requests, rows, the key hook, row checks, analyse's completeness
# ----------------------------------------------------------------------------


class FakeOutcome:
    def __init__(self, text, coins):
        p = village.parse_reply(text, coins, herding.MAX_STAKE)
        self.reply, self.raw, self.parse, self.gen_tokens = p.reply, text, p.parse, 10


def test_requests_and_rows_carry_key_coins_and_prompt(source):
    _, drawn, _ = source
    sits = drawn["main"]
    for job, i, c, m in herding.jobs(sits, "main")[:300]:
        s = sits[i]
        req = herding.request_of(s, i, c, m)
        assert req.label == s["key"] and req.coins == s["coins"] and req.system == herding.s_system(s)
        assert req.observation == herding.observation(s, c, m)
        row = herding.row_of(job, "main", s, c, m, req, FakeOutcome("reasoning: r\nmessage: m\naction: cast\nstake: 5", s["coins"]))
        assert row["key"] == s["key"] and row["job"] == job == herding.job_id(s, c, m)
        assert row["intent"] and row["real"] == (s["coins"] >= 5) and row["attempt"] == (s["coins"] < 5)
    herding.check_rows([herding.row_of(j, "main", sits[i], c, m, herding.request_of(sits[i], i, c, m), FakeOutcome("x", 5))
                        for j, i, c, m in herding.jobs(sits, "main")[:50]], {s["sid"]: s for s in sits})


def test_check_rows_refuses_a_changed_prompt(source):
    _, drawn, _ = source
    s = drawn["main"][0]
    req = herding.request_of(s, 0, "peers", 3)
    row = herding.row_of(herding.job_id(s, "peers", 3), "main", s, "peers", 3, req, FakeOutcome("x", 5))
    herding.check_rows([row], {s["sid"]: s})
    for field, value in (("observation", row["observation"] + " "), ("key", "herding/main/other"), ("m", 4)):
        with pytest.raises(ValueError):
            herding.check_rows([{**row, field: value}], {s["sid"]: s})


def test_act_batch_samples_with_key_fn(monkeypatch):
    seen = []
    monkeypatch.setattr(village, "make_sampler", lambda key, t, p: seen.append(key) or (lambda lp: lp))

    class Gen:
        def insert(self, prompts, max_tokens, **kw):
            self.n = len(prompts)
            return list(range(len(prompts)))

        def next_generated(self):
            if self.n is None:
                return []
            n, self.n = self.n, None
            return [type("R", (), {"uid": u, "finish_reason": "stop", "token": 0})() for u in range(n)]

    class Tok:
        def decode(self, toks):
            return ""

    reqs = [Request("herding/main/main-001", 0, 0, 1, 3, 4, "Fisher E", "sys", "obs", 10), Request("x", 7, 1, 2, 5, 6, "Fisher G", "sys", "obs", 10)]
    for key_fn, want in ((village.request_key, ["0/g0/e1/r3/sample/4", "7/g1/e2/r5/sample/6"]), (lambda r: r.label, ["herding/main/main-001", "x"])):
        mp = object.__new__(village.ModelPlayers)
        mp.gen, mp.tokenizer, mp.key_fn = Gen(), Tok(), key_fn
        mp.max_tokens, mp.temperature, mp.top_p, mp.max_stake = 16, 1.0, 1.0, 5
        mp.timing_path, mp.group_size, mp.calls = None, 1, 0
        mp.prompt_ids = lambda system, observation: ([1, 2], [3], None)
        seen.clear()
        outs = mp.act_batch(reqs)
        assert seen == want and len(outs) == 2


def test_analyse_refuses_an_incomplete_phase_and_runs_on_a_complete_one(source, tmp_path):
    _, drawn, _ = source
    (tmp_path / "situations.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in drawn["main"] + drawn["pilot"]))
    d = tmp_path / "pilot"
    d.mkdir()
    rng = random.Random(3)
    rows = []
    for job, i, c, m in herding.jobs(drawn["pilot"], "pilot"):
        s = drawn["pilot"][i]
        act = "cast" if rng.random() < (0.2 if m == 0 else 0.7) else "fish"
        text = f"reasoning: r\nmessage: m\naction: {act}\nstake: {1 if act == 'cast' else 0}"
        rows.append(herding.row_of(job, "pilot", s, c, m, herding.request_of(s, i, c, m), FakeOutcome(text, s["coins"])))

    def write(rs):
        (d / "replies.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))

    import argparse

    a = argparse.Namespace(out=str(tmp_path), phase="pilot", boot=50)
    for bad in (rows[:-1], rows[:-1] + rows[:1]):
        write(bad)
        with pytest.raises(SystemExit):
            herding.cmd_analyse(a)
    write(rows)
    herding.cmd_analyse(a)
    res = json.loads((d / "summary.json").read_text())
    assert res["rows"] == 400 and 0.3 < res["intent"]["conditions"]["peers"]["E"]["est"] < 0.7
    assert res["resolving_power"]["line_3se"][0] <= res["resolving_power"]["line_3se"][1]


def test_t975():
    assert herding.t975(99) == 1.9842 and herding.t975(199) == 1.9720
    assert herding.t975(10) == village.t_quantiles(10)[0]
    for df, exact in ((49, 2.0096), (98, 1.9845), (150, 1.9759)):
        assert abs(herding.t975(df) - exact) < 2e-4


def test_pilot_real_phase(source):
    _, drawn, _ = source
    jobs = herding.jobs(drawn[herding.SITUATIONS_OF["pilot_real"]], "pilot_real")
    assert len(jobs) == 400 and {(c, m) for _, _, c, m in jobs} == {("real", 0), ("real", 7)}
    pilot = {j[0] for j in herding.jobs(drawn["pilot"], "pilot")}
    assert not pilot & {j[0] for j in jobs}  # distinct job ids, same situations
    s = drawn["pilot"][0]
    assert herding.request_of(s, 0, "real", 7).label == herding.request_of(s, 0, "peers", 7).label == s["key"]
    assert herding.run_config("pilot_real", herding.OUT / "situations.jsonl")["jobs"] == 400 if (herding.OUT / "situations.jsonl").exists() else True


def test_two_level_comparison_has_no_slope():
    rows = []
    for i in range(60):
        for c, p in (("peers", (0, 0)), ("real", (0, 1))):
            for m, y in zip((0, 7), p):
                rows.append({"sid": f"pilot-{i:03d}", "condition": c, "m": m, "intent": bool(y), "real": bool(y)})
    res = herding.analyse(rows, "intent", n_boot=20)
    d = res["vs_peers"]["real"]
    assert d["E"]["est"] == 1.0 and "7b" not in d and "ratio_7b" not in d and abs(d["level"]["est"] - 0.5) < 1e-12
    assert "7b" not in res["conditions"]["real"]


# ----------------------------------------------------------------------------
# logged play: the other-day placebo
# ----------------------------------------------------------------------------


def synthetic_village(days, p_cast, seed):
    """8 agents, 10 rounds a day; p_cast(round, share of the others who cast last round) -> probability."""
    from pond import Episode

    rng = random.Random(seed)
    pool = []
    for d in range(days):
        prev = None
        eps = [Episode(a, d, pond.agent_name(a), 10) for a in range(8)]
        for r in range(1, 11):
            acts = []
            for a in range(8):
                share = None if prev is None else sum(prev[b] for b in range(8) if b != a) / 7
                cast = rng.random() < p_cast(r, share)
                reply = Reply("", "", "cast" if cast else "fish", 1 if cast else 0)
                eps[a].turns.append(Turn(d, r, "", "", reply, 10, 10, False if cast else None, 0.5, "", "ok", False, 0))
                acts.append(cast)
            prev = acts
        pool.extend(eps)
    return pool


def test_placebo_separates_copying_from_the_round_profile():
    copy = {"v": synthetic_village(20, lambda r, sh: 0.3 if sh is None else 0.1 + 0.8 * sh, 1)}
    res = herding.placebo_analysis(copy, perms=20, boot_n=5, boot_perms=2)
    assert res["real"]["all/real"]["fe_slope"] > 0.4
    assert abs(res["placebo"]["all/real"]["fe_slope"]["mean"]) < 0.1
    rounds = {"v": synthetic_village(40, lambda r, sh: 0.05 + 0.09 * r, 2)}  # null sd of the slope about 0.04 at 40 days
    res = herding.placebo_analysis(rounds, perms=20, boot_n=5, boot_perms=2)
    real, pl = res["real"]["all/real"], res["placebo"]["all/real"]
    assert abs(real["fe_slope"]) < 0.15 and abs(pl["fe_slope"]["mean"]) < 0.15
    assert real["raw_slope"] > 0.2 and abs(real["raw_slope"] - pl["raw_slope"]["mean"]) < 0.15


def test_placebo_real_rows_reproduce_the_games_herding_slope(source):
    pools, _, _ = source
    blocks = [b for v, pool in pools.items() for d in sorted({ep.episode for ep in pool}) for b in herding.day_rows(pool, d, d, v)]
    assert abs(herding.placebo_stats(blocks)["all/real"]["fe_slope"] - village.herding_slope(list(pools.values()))) < 1e-12


def test_derangement_has_no_fixed_point():
    rng = random.Random(0)
    for _ in range(50):
        der = herding.derangement(list(range(30)), rng)
        assert sorted(der.values()) == list(range(30)) and all(k != v for k, v in der.items())


# ----------------------------------------------------------------------------
# deeds against words, and the printer with real messages
# ----------------------------------------------------------------------------


def test_advice_class():
    cases = {"Let's all aim for that golden fish!": "cast", "Go for the golden fish to boost our earnings!": "cast",
             "Cast together for the golden fish!": "cast", "Keep fishing to build our coin stash.": "fish",
             "Let's focus on steady fishing for now.": "fish",
             "I'm casting 3 coins.": None, "Casting for the golden fish!": None, "Starting with fishing.": None,
             "Go for the golden fish if you have spare coins.": None, "Let's fish first, then cast.": None,
             "Let's cast before it's too late, carefully.": None, "Join Fisher F for a 3-coin cast.": None,
             "Let's try a small stake on the golden fish.": None, "继续努力！": None, "Fishing steadily.": None,
             "Let's aim high with my stake.": None, "Let's go for it at a lower stake for a safer try.": None,
             "Let's keep fishing for now and save for a cast.": None, "Stick to steady gains; be mindful of golden fish risks.": None,
             "Join H's attempt for the golden fish.": None, "Let's aim for the gold fish carefully.": None,
             "Let's take a moderate stake for the golden fish.": None, "Keep an eye on the golden fish.": None,
             "Let's balance between fishing and casting.": None, "Let's give casting a try, gemeinsam!": "cast",
             "Let's cast for the golden\u9c7c": None, "Let's fish and try our luck later.": None}
    for text, want in cases.items():
        assert herding.advice_class(text) == want, text


@pytest.fixture(scope="module")
def advised(source):
    pools, drawn, _ = source
    ap = herding.advice_pools(pools)
    return ap, {a["sid"]: a["advice"] for a in herding.draw_advice(drawn["pilot"], ap)}


def test_advice_draws(source, advised):
    _, drawn, turns = source
    ap, adv = advised
    assert len(adv) == 200
    for s in drawn["pilot"]:
        for k in ("cast", "fish"):
            msgs = adv[s["sid"]][k]
            assert len(msgs) == N_LINES and len({m["text"] for m in msgs}) == N_LINES
            for m in msgs:
                assert (m["village"], m["day"], m["agent"]) != (s["village"], s["day"], s["agent"])
                t = turns[(m["village"], m["day"], m["agent"], m["round"])]
                assert not t.failed and t.reply.message.strip() == m["text"] and m["round"] <= 9
                assert herding.advice_class(m["text"]) == k
    assert {a["sid"]: a["advice"] for a in herding.draw_advice(drawn["pilot"], ap)} == adv  # deterministic


def test_deeds_and_printer_real_prompts(source, advised):
    _, drawn, _ = source
    _, adv = advised
    for s0 in drawn["pilot"][:15]:
        s = {**s0, "advice": adv[s0["sid"]]}
        for cond, k in (("deeds_advise_cast", "cast"), ("deeds_advise_fish", "fish")):
            for m, act in ((0, "fish"), (N_LINES, "cast")):
                rep = report(s, cond, m)
                assert rep[0] == herding.DEEDS_HEADER and len(rep) == 1 + N_LINES
                assert rep[1:] == [f"- {nm}: {act}. Message: {a['text']}" for nm, a in zip(s["names"], s["advice"][k])]
        for m in (0, N_LINES):
            real = report(s, "real", m)
            pr = report(s, "printer_real", m)
            assert pr[0] == herding.PRINTER_HEADER and len(pr) == 1 + N_LINES
            assert [l.split(": ", 1)[1] for l in pr[1:]] == [l.split(": ", 1)[1] for l in real[1:]]
            assert [l.split(": ", 1)[0] for l in pr[1:]] == [f"- Line {k + 1}" for k in range(N_LINES)]
            assert not any("Fisher" in l.split(": ", 1)[0] for l in pr[1:])


def test_new_phases_jobs(source):
    _, drawn, _ = source
    deeds = herding.jobs(drawn["pilot"], "pilot_deeds")
    assert len(deeds) == 800 and {(c, m) for _, _, c, m in deeds} == set(herding.PHASE_JOBS["pilot_deeds"])
    pr = herding.jobs(drawn["pilot"], "pilot_printer_real")
    assert len(pr) == 400 and {(c, m) for _, _, c, m in pr} == {("printer_real", 0), ("printer_real", 7)}
    assert herding.CONDITIONS == ("peers", "printer", "bots", "independence", "real")  # the main run's jobs are unchanged
    assert len(herding.jobs(drawn["main"], "main")) == 4100


def test_deeds_analysis_recovers_known_effects():
    rows = []
    for i in range(80):
        sid = f"pilot-{i:03d}"
        for cond, m, y in (("deeds_advise_cast", 7, 1), ("deeds_advise_cast", 0, 1), ("deeds_advise_fish", 7, 0), ("deeds_advise_fish", 0, 0),
                           ("real", 7, 1), ("real", 0, 0), ("peers", 7, i % 2), ("peers", 0, 0)):
            rows.append({"sid": sid, "condition": cond, "m": m, "intent": bool(y), "real": bool(y)})
    d = herding.deeds_analysis(rows, "intent", n_boot=20)
    assert d["effects"]["advice"]["est"] == 1.0 and d["effects"]["actions"]["est"] == 0.0 and d["effects"]["interaction"]["est"] == 0.0
    assert d["effects"]["agreeing"]["est"] == 1.0 and d["E_real"]["est"] == 1.0 and d["E_peers"]["est"] == 0.5
    assert d["paired"]["advice - E_real"]["est"] == 0.0 and d["paired"]["actions - E_peers"]["est"] == -0.5
    assert d["cells"]["fish_actions_advise_cast"]["est"] == 1.0


# ----------------------------------------------------------------------------
# matched report and advice messages, and their crossing
# ----------------------------------------------------------------------------


def test_matched_sentences_are_twins():
    for kind, templates in (("report", herding.REPORT_TEMPLATES), ("advice", herding.ADVICE_TEMPLATES)):
        assert len(templates) == 12 and len(set(templates)) == 12
        for i in range(len(templates)):
            c, f = herding.matched_sentence(kind, i, "cast"), herding.matched_sentence(kind, i, "fish")
            assert c != f and c.endswith(".") and f.endswith(".") and "!" not in c + f
            assert ("golden fish" in c) == ("for a coin" in f)
            if kind == "report":
                assert c.startswith(("I ", "This round I", "My move", "In this round I"))
            else:
                assert not any(w in c.lower().split() for w in ("i", "my", "me")), c  # no statement of the speaker's own action
    assert herding.matched_sentence("report", 0, "cast") == "I cast this round."
    assert herding.matched_sentence("advice", 6, "fish") == "Everyone should fish for a coin this round."


def test_matched_prompts(source):
    _, drawn, _ = source
    m = {d["sid"]: d for d in herding.draw_matched(drawn["pilot"])}
    assert m == {d["sid"]: d for d in herding.draw_matched(drawn["pilot"])}  # deterministic
    for s0 in drawn["pilot"][:15]:
        d = m[s0["sid"]]
        assert len(set(d["report"])) == N_LINES and len(set(d["advice"])) == N_LINES
        s = {**s0, "matched": {"report": d["report"], "advice": d["advice"]}}
        for cond, kind in (("msg_report", "report"), ("msg_advice", "advice")):
            for lvl, act in ((0, "fish"), (N_LINES, "cast")):
                rep = report(s, cond, lvl)
                assert rep[0] == herding.MESSAGES_HEADER and len(rep) == 1 + N_LINES
                assert rep[1:] == [f"- {nm}: {herding.matched_sentence(kind, t, act)}" for nm, t in zip(s["names"], d[kind])]
        for cond, adv in (("cross_advise_cast", "cast"), ("cross_advise_fish", "fish")):
            for lvl, act in ((0, "fish"), (N_LINES, "cast")):
                rep = report(s, cond, lvl)
                assert rep[1:] == [f"- {nm}: {herding.matched_sentence('report', r, act)} {herding.matched_sentence('advice', a, adv)}"
                                   for nm, r, a in zip(s["names"], d["report"], d["advice"])]
    assert len(herding.jobs(drawn["pilot"], "pilot_matched")) == 800 and len(herding.jobs(drawn["pilot"], "pilot_cross")) == 800


def test_crossed_analysis_reads_its_own_cells():
    rows = []
    for i in range(40):
        sid = f"pilot-{i:03d}"
        for cond, m, y in (("cross_advise_cast", 7, 1), ("cross_advise_cast", 0, 1), ("cross_advise_fish", 7, 1), ("cross_advise_fish", 0, 0),
                           ("real", 7, 0), ("real", 0, 0), ("peers", 7, 0), ("peers", 0, 0), ("msg_report", 7, 1), ("msg_report", 0, 0)):
            rows.append({"sid": sid, "condition": cond, "m": m, "intent": bool(y), "real": bool(y)})
    d = herding.deeds_analysis(rows, "intent", 20, conds=("cross_advise_cast", "cross_advise_fish"), refs=("real", "msg_report"))
    assert d["cells"]["fish_actions_advise_fish"]["est"] == 0.0 and d["cells"]["cast_actions_advise_fish"]["est"] == 1.0
    assert d["effects"]["actions"]["est"] == 0.5 and d["effects"]["advice"]["est"] == 0.5 and d["E_msg_report"]["est"] == 1.0
    with pytest.raises(TypeError):  # the condition names cannot be passed by position (the 09-30 wiring bug)
        herding.deeds_analysis(rows, "intent", 20, "k", ("cross_advise_cast", "cross_advise_fish"))
