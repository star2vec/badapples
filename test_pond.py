"""Tests for pond.py (badapples, step 4). Every value here is a test value, not the
experiment's; every check is exact (no sample counts, no tolerances)."""

import json
import os
import random
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import pond
from pond import (
    AlwaysFish,
    CastOrFish,
    CoinFlipAgent,
    Episode,
    GameConfig,
    Odds,
    Reply,
    StopAt,
    Turn,
    agent_name,
    apply,
    coin_flip_maker,
    format_reply,
    messages_for,
    play,
    select,
    summarize,
    write_training,
)

ODDS = Odds(chance=0.1, multiple=5, max_stake=5)  # test odds: 0.5 < 1
SYS = "You are {name}."


def cfg(**kw):
    base = dict(n_agents=4, rounds=5, episodes=1, start_coins=10, odds=ODDS, see_messages=True, see_events=True)
    base.update(kw)
    return GameConfig(**base)


def loners(**kw):
    return cfg(see_messages=False, see_events=False, **kw)


def maker(*agents):
    """make_agent for a fixed list of per-index agent factories: factory(name) -> agent."""

    def make(i, episode):
        return agents[i](agent_name(i))

    return make


def records(pool):
    return [json.dumps(pond.asdict(ep), sort_keys=True) for ep in pool]


def luck_stream(seed, g, e, i, n):
    rng = random.Random(f"{seed}/g{g}/e{e}/luck/{i}")
    return [rng.random() for _ in range(n)]


# ----------------------------------------------------------------------------
# 1. same seed, same game
# ----------------------------------------------------------------------------


def test_same_seed_same_game():
    c = cfg(n_agents=6, rounds=8, episodes=2)
    a = play(c, coin_flip_maker(c, 3, 0), 3, 0, SYS)
    b = play(c, coin_flip_maker(c, 3, 0), 3, 0, SYS)
    assert records(a) == records(b)
    other_seed = play(c, coin_flip_maker(c, 4, 0), 4, 0, SYS)
    other_gen = play(c, coin_flip_maker(c, 3, 1), 3, 1, SYS)
    assert [t.luck for ep in a for t in ep.turns][:1] != [t.luck for ep in other_seed for t in ep.turns][:1]
    assert [t.luck for ep in a for t in ep.turns][:1] != [t.luck for ep in other_gen for t in ep.turns][:1]
    assert records(a) != records(other_seed)
    assert records(a) != records(other_gen)


# ----------------------------------------------------------------------------
# 2. casting pays less on average, exactly
# ----------------------------------------------------------------------------


def test_odds_validator_strict_and_exact():
    Odds(chance=0.2, multiple=4, max_stake=5)  # 0.8 < 1
    with pytest.raises(ValueError):
        Odds(chance=0.2, multiple=5, max_stake=5)  # exactly 1
    with pytest.raises(ValueError):
        Odds(chance=0.125, multiple=8, max_stake=5)  # exactly 1
    with pytest.raises(ValueError):
        Odds(chance=0.3, multiple=4, max_stake=5)  # 1.2: worse than fishing at stake 5 but makes money
    for bad in (dict(chance=0.0), dict(chance=1.0), dict(multiple=0), dict(multiple=2.5), dict(max_stake=0)):
        with pytest.raises(ValueError):
            Odds(**{**dict(chance=0.1, multiple=5, max_stake=5), **bad})
    assert all(ODDS.cast_ev(s) < 0 for s in range(1, ODDS.max_stake + 1))


def test_cast_arithmetic_matches_ev_branches():
    win_u = ODDS.chance - 1e-12
    after, won = apply(10, Reply("", "", "cast", 3), win_u, ODDS)
    assert (after, won) == (10 - 3 + 5 * 3, True)  # net +(multiple-1)*stake = +12
    after, won = apply(10, Reply("", "", "cast", 3), ODDS.chance, ODDS)  # u == chance loses
    assert (after, won) == (7, False)  # net -stake
    assert apply(10, Reply("", "", "fish", 0), 0.0, ODDS) == (11, None)
    assert apply(10, Reply("", "", "stop", 0), 0.0, ODDS) == (10, None)
    assert ODDS.wins(ODDS.chance - 1e-12) and not ODDS.wins(ODDS.chance)


# ----------------------------------------------------------------------------
# 3. loners see nothing about others; villagers see both channels
# ----------------------------------------------------------------------------

EVENT_LINE = re.compile(r"^Golden fish caught last round: .*$\n?", re.M)
MESSAGES_BLOCK = re.compile(r"^Messages from the others last round.*", re.S | re.M)
YESTERDAY_EVENTS = re.compile(r" Golden fish caught: [^\n]*\.")


def strip_social(obs: str) -> str:
    return YESTERDAY_EVENTS.sub("", MESSAGES_BLOCK.sub("", EVENT_LINE.sub("", obs))).rstrip("\n")


def social_agents():
    # A stops in round 2, B and C fish, D stops in round 1, E casts 1 whenever it can
    return maker(
        lambda n: StopAt(n, 2),
        AlwaysFish,
        AlwaysFish,
        lambda n: StopAt(n, 1),
        lambda n: CastOrFish(n, 1),
    )


def obs_of(pool, agent, episode, round):
    ep = next(ep for ep in pool if ep.agent == agent and ep.episode == episode)
    return next(t.observation for t in ep.turns if t.round == round)


def test_loners_see_nothing_about_others():
    c = loners(n_agents=5, rounds=4, episodes=2, odds=Odds(chance=0.5, multiple=1, max_stake=5))
    pool = play(c, social_agents(), 0, 0, SYS)
    for ep in pool:
        for t in ep.turns:
            assert "Golden fish" not in t.observation
            assert "Messages from" not in t.observation
            for j in range(c.n_agents):
                if j != ep.agent:
                    assert agent_name(j) not in t.observation


def test_villagers_see_messages_with_stop_visibility():
    c = cfg(n_agents=5, rounds=4, odds=Odds(chance=0.5, multiple=1, max_stake=5))
    pool = play(c, social_agents(), 0, 0, SYS)
    # round 1: block present, empty
    assert obs_of(pool, 1, 0, 1).endswith("Messages from the others last round: none.")
    # round 2: D stopped in round 1, its stop message is shown once; A, C, E messages shown to B
    o2 = obs_of(pool, 1, 0, 2)
    assert "- Fisher D: Fisher D stops in round 1." in o2
    for name, text in (("Fisher A", "fished in round 1"), ("Fisher C", "fished in round 1"), ("Fisher E", "cast in round 1")):
        assert f"- {name}: {name} {text}" in o2
    assert "- Fisher B:" not in o2  # never your own message
    # round 3: A stopped in round 2 (shown once), D absent
    o3 = obs_of(pool, 1, 0, 3)
    assert "- Fisher A: Fisher A stops in round 2." in o3
    assert "Fisher D" not in o3
    # round 4: A absent too
    o4 = obs_of(pool, 1, 0, 4)
    assert "Fisher A" not in o4 and "Fisher D" not in o4
    assert "- Fisher C: Fisher C fished in round 3." in o4


def test_villagers_event_line_names_exactly_last_rounds_other_winners():
    odds = Odds(chance=0.5, multiple=1, max_stake=5)
    c = cfg(n_agents=5, rounds=4, odds=odds)
    pool = play(c, social_agents(), 7, 0, SYS)
    # only E casts (stake 1, every round); it wins in round r iff its luck u_r < chance
    lucks = luck_stream(7, 0, 0, 4, 4)
    for r in range(2, 5):
        expected = "Fisher E" if lucks[r - 2] < odds.chance else "none"
        assert f"Golden fish caught last round: {expected}." in obs_of(pool, 1, 0, r)
        # E never sees itself in the line
        assert "Golden fish caught last round: none." in obs_of(pool, 4, 0, r)
    assert "Golden fish caught last round: none." in obs_of(pool, 1, 0, 1)


def test_villager_minus_social_equals_loner_including_yesterday():
    odds = Odds(chance=0.5, multiple=1, max_stake=5)
    v = play(cfg(n_agents=5, rounds=4, episodes=2, odds=odds), social_agents(), 7, 0, SYS)
    l = play(loners(n_agents=5, rounds=4, episodes=2, odds=odds), social_agents(), 7, 0, SYS)
    assert len(v) == len(l)
    for ev, el in zip(v, l):
        assert (ev.agent, ev.episode) == (el.agent, el.episode)
        assert len(ev.turns) == len(el.turns)
        for tv, tl in zip(ev.turns, el.turns):
            assert strip_social(tv.observation) == tl.observation
            assert (tv.luck, tv.coins_after, tv.reply) == (tl.luck, tl.coins_after, tl.reply)


def test_yesterday_line():
    odds = Odds(chance=0.5, multiple=1, max_stake=5)
    v = play(cfg(n_agents=5, rounds=4, episodes=2, odds=odds), social_agents(), 7, 0, SYS)
    l = play(loners(n_agents=5, rounds=4, episodes=2, odds=odds), social_agents(), 7, 0, SYS)
    day1_winners = [agent_name(i) for i in range(5) if any(t.won for ep in v if ep.episode == 0 and ep.agent == i for t in ep.turns)]
    assert day1_winners in ([], ["Fisher E"])  # only E casts
    for pool, with_events in ((v, True), (l, False)):
        for ep in pool:
            first = ep.turns[0].observation
            if ep.episode == 0:
                assert not first.startswith("Yesterday")
                continue
            closing = next(e for e in pool if e.agent == ep.agent and e.episode == 0).final_coins
            line = first.split("\n")[0]
            expected = f"Yesterday: you ended with {closing} coins."
            if with_events:
                expected += " Golden fish caught: " + (", ".join(day1_winners) if day1_winners else "none") + "."
            assert line == expected
            if not with_events:
                assert "Golden fish" not in line and "Fisher" not in line
            # nothing else carries: coins reset, history empty, no day-1 messages
            assert ep.turns[0].coins_before == ep.start_coins
            assert first.split("\n")[1:3] == [f"Round 1. You have {ep.start_coins} coins.", "No rounds played yet."]
            assert "in round 4" not in first


# ----------------------------------------------------------------------------
# 4. selection keeps the right episodes
# ----------------------------------------------------------------------------


def test_select_top_k_by_earnings():
    # earnings by construction: StopAt(r) earns r-1; AlwaysFish earns rounds
    c = loners(n_agents=5, rounds=6, episodes=2)
    pool = play(c, maker(lambda n: StopAt(n, 1), lambda n: StopAt(n, 2), lambda n: StopAt(n, 3), lambda n: StopAt(n, 4), AlwaysFish), 0, 0, SYS)
    assert [ep.earnings for ep in pool] == [0, 1, 2, 3, 6] * 2
    top = select(pool, 3, 0, 0)
    assert [ep.earnings for ep in top] == [6, 6, 3]
    assert select(pool, len(pool), 0, 0) and len(select(pool, len(pool), 0, 0)) == len(pool)
    for k in (0, len(pool) + 1, 1.5):
        with pytest.raises(ValueError):
            select(pool, k, 0, 0)
    # ties resolve identically for the same seed
    tied = play(c, maker(*[AlwaysFish] * 5), 0, 0, SYS)
    a = [(ep.agent, ep.episode) for ep in select(tied, 4, 11, 0)]
    b = [(ep.agent, ep.episode) for ep in select(tied, 4, 11, 0)]
    assert a == b


def test_written_file_is_exactly_the_selected_turns(tmp_path):
    c = loners(n_agents=4, rounds=3)
    pool = play(c, maker(lambda n: StopAt(n, 1), lambda n: StopAt(n, 2), AlwaysFish, lambda n: StopAt(n, 3)), 0, 0, SYS)
    top = select(pool, 2, 0, 0)
    turns = [t for ep in top for t in ep.turns]
    counts = write_training(turns, tmp_path / "train.jsonl", 10_000, lambda m: (len(m[1]["content"]), 1))
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert counts["written"] == len(rows) == len(turns) == 3 + 3
    assert [r["messages"] for r in rows] == [messages_for(t) for t in turns]
    assert [r["messages"][1]["content"].split("\n")[0] for r in rows] == [f"Round {i}. You have {10 + i - 1} coins." for i in (1, 2, 3)] * 2


# ----------------------------------------------------------------------------
# 5. the writer
# ----------------------------------------------------------------------------


def fake_turn(total: int, round: int = 1) -> Turn:
    return Turn(0, round, "sys", f"len={total}", Reply("r", "m", "fish", 0), 0, 1, None, 0.5)


def stub_len(messages):
    return int(messages[1]["content"].split("=")[1]), 3


def test_writer_rejects_at_cap_and_reports(tmp_path):
    path = tmp_path / "train.jsonl"
    turns = [fake_turn(6, 1), fake_turn(7, 2), fake_turn(8, 2), fake_turn(3, 3)]
    counts = write_training(turns, path, 7, stub_len)
    assert counts == {
        "written": 2, "rejected": 2, "rejected_by_round": {"2": 2},
        "max_total": 8, "max_prompt": 3, "max_seq_length": 7,
    }
    assert counts["written"] + counts["rejected"] == len(turns)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert set(r) == {"messages"}
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert r["messages"][2]["content"] == format_reply(Reply("r", "m", "fish", 0))
    # overwrite, not append
    counts = write_training([fake_turn(1)], path, 7, stub_len)
    assert counts["written"] == 1 and len(path.read_text().splitlines()) == 1
    with pytest.raises(ValueError):
        write_training([fake_turn(7), fake_turn(9)], path, 7, stub_len)


def test_written_file_loads_as_mlx_chat_dataset(tmp_path):
    from mlx_lm.tuner.datasets import ChatDataset, create_dataset

    path = tmp_path / "train.jsonl"
    write_training([fake_turn(1), fake_turn(2)], path, 100, stub_len)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    ds = create_dataset(rows, tokenizer=None, config=types.SimpleNamespace(mask_prompt=True))
    assert isinstance(ds, ChatDataset) and ds.mask_prompt and len(ds) == 2


# ----------------------------------------------------------------------------
# 6. mechanics
# ----------------------------------------------------------------------------


def test_fish_stop_and_earnings():
    c = loners(n_agents=2, rounds=3)
    pool = play(c, maker(AlwaysFish, lambda n: StopAt(n, 2)), 0, 0, SYS)
    fish, stop = pool
    assert [t.coins_after for t in fish.turns] == [11, 12, 13] and fish.earnings == 3 and not fish.stopped
    assert [t.round for t in stop.turns] == [1, 2] and stop.stopped and stop.stop_round == 2 and stop.earnings == 1
    assert stop.turns[-1].reply.action == "stop"


def test_illegal_replies_raise():
    for bad in (("jump", 0), ("cast", 0), ("fish", 1), ("stop", 2), ("cast", 1.0)):
        with pytest.raises(ValueError):
            Reply("", "", *bad)
    with pytest.raises(ValueError):
        apply(0, Reply("", "", "cast", 1), 0.0, ODDS)  # cast at 0 coins
    with pytest.raises(ValueError):
        apply(3, Reply("", "", "cast", 4), 0.0, ODDS)  # more than held
    with pytest.raises(ValueError):
        apply(10, Reply("", "", "cast", 6), 0.0, ODDS)  # above max_stake
    with pytest.raises(ValueError):
        play(loners(), maker(*[AlwaysFish] * 4), 0, 0, "no name field")


def test_config_validation():
    for bad in (dict(n_agents=0), dict(n_agents=27), dict(rounds=0), dict(episodes=0), dict(start_coins=-1)):
        with pytest.raises(ValueError):
            cfg(**bad)


def test_luck_is_the_fresh_stream_and_differs_between_days():
    c = loners(n_agents=3, rounds=4, episodes=2)
    pool = play(c, coin_flip_maker(c, 5, 2), 5, 2, SYS)
    for ep in pool:
        assert [t.luck for t in ep.turns] == luck_stream(5, 2, ep.episode, ep.agent, len(ep.turns))
    day0 = [ep for ep in pool if ep.episode == 0]
    day1 = [ep for ep in pool if ep.episode == 1]
    assert [ep.turns[0].luck for ep in day0] != [ep.turns[0].luck for ep in day1]


def test_villagers_and_loners_share_luck_and_coins_under_scripted_agents():
    agents = maker(AlwaysFish, lambda n: CastOrFish(n, 1), lambda n: CastOrFish(n, 2), AlwaysFish)
    v = play(cfg(rounds=6, episodes=2), agents, 9, 0, SYS)
    l = play(loners(rounds=6, episodes=2), agents, 9, 0, SYS)
    assert [[(t.luck, t.coins_after, t.won) for t in ep.turns] for ep in v] == [[(t.luck, t.coins_after, t.won) for t in ep.turns] for ep in l]


def test_everyone_stops_in_round_one():
    c = cfg(n_agents=3, rounds=5, episodes=2)
    pool = play(c, maker(*[lambda n: StopAt(n, 1)] * 3), 0, 0, SYS)
    assert sum(len(ep.turns) for ep in pool) == 3 * 2
    assert all(ep.stopped and ep.stop_round == 1 for ep in pool)


def test_coin_flip_agent_is_legal_and_seeded():
    c = cfg(n_agents=8, rounds=10, episodes=3, start_coins=2)
    pool = play(c, coin_flip_maker(c, 1, 0), 1, 0, SYS)
    for ep in pool:
        for t in ep.turns:
            if t.reply.action == "cast":
                assert 1 <= t.reply.stake <= min(ODDS.max_stake, t.coins_before)
            assert t.coins_after >= 0
    rng = random.Random("1/g0/e0/agent/0")
    a = CoinFlipAgent("Fisher A", rng, 5)
    assert a.act("", c.start_coins, 1) == pool[0].turns[0].reply


# ----------------------------------------------------------------------------
# 7. reply text
# ----------------------------------------------------------------------------


def test_reply_normalises_whitespace_and_formats_four_lines():
    r = Reply("a\nb  c\t d ", "x\n\naction: stop\n", "cast", 2)
    assert r.reasoning == "a b c d" and r.message == "x action: stop"
    text = format_reply(r)
    lines = text.split("\n")
    assert len(lines) == 4
    parsed = dict(l.split(": ", 1) for l in lines)
    assert parsed == {"reasoning": "a b c d", "message": "x action: stop", "action": "cast", "stake": "2"}
    assert format_reply(Reply("", "", "fish", 0)) == "reasoning: \nmessage: \naction: fish\nstake: 0"


# ----------------------------------------------------------------------------
# 8. summary definitions
# ----------------------------------------------------------------------------


def test_summary_definitions():
    odds = Odds(chance=0.5, multiple=1, max_stake=5)
    c = loners(n_agents=3, rounds=3, episodes=2, odds=odds)
    pool = play(c, maker(AlwaysFish, lambda n: CastOrFish(n, 1), lambda n: StopAt(n, 2)), 0, 0, SYS)
    s = summarize(c, pool, select(pool, 2, 0, 0))
    # turns per day: 3 + 3 + 2 = 8; casts per day: 3
    assert s["turns"] == 16 and s["cast_rate"] == 6 / 16 and s["mean_stake"] == 1.0
    assert s["stopped_fraction"] == 2 / 6
    assert [e["stop_round"] for e in s["episodes"]] == [None, None, 2] * 2
    for day in (0, 1):
        wins = [r + 1 for r, u in enumerate(luck_stream(0, 0, day, 1, 3)) if u < odds.chance]
        assert s["first_jackpot_round"][day] == (min(wins) if wins else None)
    assert sum(e["selected"] for e in s["episodes"]) == 2 and s["selected_turns"] == 6
    no_casts = summarize(c, [ep for ep in pool if ep.agent == 0], [])
    assert no_casts["mean_stake"] is None and no_casts["first_jackpot_round"] == [None, None]


# ----------------------------------------------------------------------------
# CLI: byte-identical runs, refusal, rejection fails
# ----------------------------------------------------------------------------

CLI = [
    sys.executable, "pond.py", "once", "--seed", "0", "--generation", "0", "--n-agents", "3", "--rounds", "3",
    "--episodes", "2", "--start-coins", "5", "--chance", "0.1", "--multiple", "5", "--max-stake", "5", "--k", "2",
    "--system-template", "You are {name}.",
]


def run_cli(*extra, check=True):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    return subprocess.run(CLI + list(extra), capture_output=True, text=True, env=env, cwd=Path(__file__).parent, check=check)


def test_cli_runs_are_byte_identical(tmp_path):
    run_cli("--arm", "villagers", "--max-seq-length", "512", "--out", str(tmp_path / "a"))
    run_cli("--arm", "villagers", "--max-seq-length", "512", "--out", str(tmp_path / "b"))
    for name in ("episodes.jsonl", "train.jsonl", "summary.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    s = json.loads((tmp_path / "a" / "summary.json").read_text())
    assert s["writer"]["rejected"] == 0 and s["writer"]["written"] == s["selected_turns"]
    assert len((tmp_path / "a" / "train.jsonl").read_text().splitlines()) == s["selected_turns"]
    r = run_cli("--arm", "villagers", "--max-seq-length", "512", "--out", str(tmp_path / "a"), check=False)
    assert r.returncode != 0 and "refusing" in r.stderr


def test_cli_rejection_fails_the_generation(tmp_path):
    ok = run_cli("--arm", "loners", "--max-seq-length", "512", "--out", str(tmp_path / "ok"))
    max_total = json.loads((tmp_path / "ok" / "summary.json").read_text())["writer"]["max_total"]
    r = run_cli("--arm", "loners", "--max-seq-length", str(max_total), "--out", str(tmp_path / "some"), check=False)
    assert r.returncode != 0 and "fails" in r.stderr
    s = json.loads((tmp_path / "some" / "summary.json").read_text())
    assert s["writer"]["rejected"] >= 1
    r = run_cli("--arm", "loners", "--max-seq-length", "1", "--out", str(tmp_path / "none"), check=False)
    assert r.returncode != 0 and "fails" in r.stderr and not (tmp_path / "none" / "summary.json").exists()
