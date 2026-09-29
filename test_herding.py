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
