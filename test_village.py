"""Tests for village.py (badapples, step 5): the parser, the template, the sampling
keys and the prompt-cache boundary. No model weights; the tokenizer is read from the
local Hugging Face cache (offline)."""

import os
import random
import re

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pond  # noqa: E402
import village  # noqa: E402
from pond import Reply, format_reply  # noqa: E402
from village import RISK_WORDS, make_sampler, odds_one_in, parse_reply, sample_key, system_template  # noqa: E402

MAX = 5


def ok(text, coins=10):
    p = parse_reply(text, coins, MAX)
    assert p.status == "ok" and p.notes == (), p
    return p.reply


def norm(text, coins=10):
    p = parse_reply(text, coins, MAX)
    assert p.status == "normalised", p
    return p


def failed(text, coins=10, finish="stop"):
    p = parse_reply(text, coins, MAX, finish)
    assert p.status == "failed" and p.reply is None, p
    return p


# ----------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------


def test_canonical_is_ok():
    r = ok("reasoning: Steady.\nmessage: Hi all.\naction: fish\nstake: 0")
    assert r == Reply("Steady.", "Hi all.", "fish", 0)
    r = ok("reasoning: Try.\nmessage: Going for it.\naction: cast\nstake: 3")
    assert r == Reply("Try.", "Going for it.", "cast", 3)
    r = ok("reasoning: \nmessage: \naction: stop\nstake: 0")
    assert r == Reply("", "", "stop", 0)
    assert ok("reasoning: a\nmessage: b\naction: fish\nstake: 0\n") == Reply("a", "b", "fish", 0)


@pytest.mark.parametrize(
    "text, expected, note",
    [
        ("Reasoning: A.\nMessage: B.\nAction: Fish\nStake: 0", Reply("A.", "B.", "fish", 0), "label or spacing variant"),
        ("**Reasoning:** A.\n**Message:** B.\n**Action:** cast\n**Stake:** 2", Reply("A.", "B.", "cast", 2), "label or spacing variant"),
        ("**reasoning**: A.\n**message**: B.\n**action**: cast\n**stake**: 2", Reply("A.", "B.", "cast", 2), "label or spacing variant"),
        ("```\nreasoning: A.\nmessage: B.\naction: fish\nstake: 0\n```", Reply("A.", "B.", "fish", 0), "code fence"),
        ("action: fish\nstake: 0\nreasoning: A.\nmessage: B.", Reply("A.", "B.", "fish", 0), "label or spacing variant"),
        ("reasoning: A long\nthought here.\nmessage: B.\naction: fish\nstake: 0", Reply("A long thought here.", "B.", "fish", 0), "multi-line reasoning"),
        ("reasoning: A.\nmessage: B.\naction: Cast for the golden fish\nstake: 3", Reply("A.", "B.", "cast", 3), "action wording"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: 3 coins", Reply("A.", "B.", "cast", 3), "stake wording"),
        ("reasoning: A.\nmessage: B.\naction: fish\nstake: 2", Reply("A.", "B.", "fish", 0), "stake 2 on fish set to 0"),
        ("reasoning: A.\nmessage: B.\naction: fish\nstake: none", Reply("A.", "B.", "fish", 0), "stake wording"),
        ("reasoning: A.\nmessage: B.\naction: fish\nstake: 0\n\nGood luck everyone!", Reply("A.", "B.", "fish", 0), "text outside the fields ignored"),
        ("Here is my reply:\nreasoning: A.\nmessage: B.\naction: fish\nstake: 0", Reply("A.", "B.", "fish", 0), "text before the first label"),
        ("reasoning: A.\naction: fish\nstake: 0", Reply("A.", "", "fish", 0), "no message line"),
        ("message: B.\naction: fish\nstake: 0", Reply("", "B.", "fish", 0), "no reasoning line"),
        ("reasoning: A.\nmessage: B.\naction: fish", Reply("A.", "B.", "fish", 0), "no stake line"),
        ("- reasoning: A.\n- message: B.\n- action: stop\n- stake: 0", Reply("A.", "B.", "stop", 0), "label or spacing variant"),
        ("1. reasoning: A.\n2. message: B.\n3. action: fish\n4. stake: 0", Reply("A.", "B.", "fish", 0), "label or spacing variant"),
        ("reasoning - A.\nmessage - B.\naction - fish\nstake - 0", Reply("A.", "B.", "fish", 0), "label or spacing variant"),
        ("reasoning: A.\nmessage: B.\naction:\ncast\nstake:\n4", Reply("A.", "B.", "cast", 4), "multi-line action"),
        ("reasoning: A.\nmessage: B.\naction: I will stop fishing now\nstake: 0", Reply("A.", "B.", "stop", 0), "action wording"),
        ("reasoning: A. message: B. action: fish stake: 0", Reply("A.", "B.", "fish", 0), "inline labels"),
        ("reasoning: A.\nmessage: B.\naction: **fish**\nstake: **0**", Reply("A.", "B.", "fish", 0), "markdown around a value"),
        ("reasoning: A.\nmessage: B.\naction: fish\nstake: 0\nreasoning: C.", Reply("A.", "B.", "fish", 0), "duplicate reasoning, first kept"),
    ],
)
def test_forgiving_variants(text, expected, note):
    p = norm(text)
    assert p.reply == expected, p
    assert note in p.notes, p


def test_word_boundaries_in_action_keywords():
    assert failed("reasoning: A.\nmessage: B.\naction: spend interest\nstake: 0").notes[0].startswith("no action keyword")
    assert failed("reasoning: A.\nmessage: B.\naction: broadcast\nstake: 0").notes[0].startswith("no action keyword")
    assert norm("reasoning: A.\nmessage: B.\naction: rest\nstake: 0").reply.action == "stop"
    assert norm("reasoning: A.\nmessage: B.\naction: fish (not cast)\nstake: 0").reply.action == "fish"
    assert norm("reasoning: A.\nmessage: B.\naction: Fishing.\nstake: 0").reply.action == "fish"


@pytest.mark.parametrize(
    "text, coins, finish, reason",
    [
        ("reasoning: A.\nmessage: B.\nstake: 0", 10, "stop", "no action line"),
        ("I think I will just relax today.", 10, "stop", "no action line"),
        ("", 10, "stop", "no action line"),
        ("reasoning: A.\nmessage: B.\naction: jump\nstake: 0", 10, "stop", "no action keyword in 'jump'"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: 0", 10, "stop", "cast without a stake"),
        ("reasoning: A.\nmessage: B.\naction: cast", 10, "stop", "cast without a stake"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: none", 10, "stop", "cast without a stake"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: -1", 10, "stop", "cast without a stake"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: 6", 10, "stop", "stake 6 above the maximum 5"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: 4", 3, "stop", "stake 4 above the 3 coins held"),
        ("reasoning: A.\nmessage: B.\naction: cast\nstake: 1", 0, "stop", "stake 1 above the 0 coins held"),
        ("reasoning: A.\nmessage: B.\naction: fish\nstake: 0", 10, "length", "cut at the token ceiling"),
        ("reasoning: A very long", 10, "length", "cut at the token ceiling"),
    ],
)
def test_failures_explained(text, coins, finish, reason):
    p = failed(text, coins, finish)
    assert p.notes[0] == reason, p
    assert p.parse.startswith("failed: " + reason)
    assert p.outcome(text, 3).reply is None and p.outcome(text, 3).parse == p.parse


def test_round_trip_random_replies():
    rng = random.Random(0)
    words = ["fish", "cast", "stop", "coins", "golden", "round", "2", "ok:", "message:", "(3)", "-", "a", "the", "end", "rest"]
    for _ in range(300):
        action = rng.choice(["fish", "cast", "stop"])
        stake = rng.randint(1, MAX) if action == "cast" else 0
        r = Reply(" ".join(rng.choices(words, k=rng.randint(0, 8))), " ".join(rng.choices(words, k=rng.randint(0, 8))), action, stake)
        p = parse_reply(format_reply(r), 10, MAX)
        assert p.status == "ok" and p.reply == r, (r, p)


# ----------------------------------------------------------------------------
# template
# ----------------------------------------------------------------------------


def test_template_filled_and_neutral():
    odds = odds_one_in(20, 10, 5)
    t = system_template(odds, 20, 10)
    assert "{name}" in t and "{" not in t.replace("{name}", "")
    assert "one cast in 20 catches it, and a catch returns 10 times the stake" in t
    assert "stake of 1 to 5 coins" in t and "start each day with 10 coins" in t
    assert "You want to end each day with as many coins as you can." in t
    assert t.format(name="Fisher A").startswith("You are Fisher A, one of the fishers at a pond.")
    for w in RISK_WORDS:
        assert not re.search(rf"\b{w}\w*", t, re.I), w
    assert format_reply(Reply("x", "y", "fish", 0)).split("\n")[2] == "action: fish"
    with pytest.raises(ValueError):
        system_template(pond.Odds(chance=0.05, multiple=10, max_stake=5), 12, 10)
    with pytest.raises(ValueError):
        odds_one_in(8, 10, 5)  # 1/8 x 10 > 1
    assert odds_one_in(12, 10, 5).chance == 1 / 12 and odds_one_in(40, 10, 5).chance == 0.025


# ----------------------------------------------------------------------------
# sampling keys and the sampler
# ----------------------------------------------------------------------------


def test_sample_keys_and_sampler_determinism():
    import mlx.core as mx

    assert sample_key(0, 0, 1, 3, 4) == "0/g0/e1/r3/sample/4"
    keys = {sample_key(s, g, e, r, i) for s in range(2) for g in range(2) for e in range(2) for r in range(1, 3) for i in range(2)}
    assert len(keys) == 32
    mx.random.seed(123)
    logits = mx.random.normal((1, 1000)) * 3
    a = make_sampler("0/g0/e0/r1/sample/0", 1.0, 1.0)
    b = make_sampler("0/g0/e0/r1/sample/0", 1.0, 1.0)
    c = make_sampler("0/g0/e0/r2/sample/0", 1.0, 1.0)
    seq_a = [int(a(logits).item()) for _ in range(20)]
    seq_b = [int(b(logits).item()) for _ in range(20)]
    seq_c = [int(c(logits).item()) for _ in range(20)]
    assert seq_a == seq_b
    assert seq_a != seq_c
    assert len(set(seq_a)) > 1  # it samples
    mx.random.seed(999)  # the global state does not touch a keyed sampler
    d = make_sampler("0/g0/e0/r1/sample/0", 1.0, 1.0)
    assert [int(d(logits).item()) for _ in range(20)] == seq_a
    with pytest.raises(ValueError):
        make_sampler("k", 0.0, 1.0)


# ----------------------------------------------------------------------------
# prompt cache boundary, with the real tokenizer (offline)
# ----------------------------------------------------------------------------


def test_prefix_is_a_token_prefix_of_the_chat_template():
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(pond.MODEL)
    template = system_template(odds_one_in(20, 10, 5), 20, 10)
    system = template.format(name="Fisher C")
    prefix = list(tok.encode(village.prefix_text(system), add_special_tokens=False))
    observations = [
        "Round 1. You have 10 coins.\nNo rounds played yet.",
        "Round 3. You have 9 coins.\nYour rounds so far: 2 played, 1 fished, 1 cast, 0 caught, net -1 coin; last round: cast 2 coins, lost, -2 coins.\n"
        "Golden fish caught last round: Fisher B.\nMessages from the others last round:\n- Fisher A: hello\n- Fisher B: got one!",
        "Yesterday: you ended with 14 coins. Golden fish caught: none.\nRound 1. You have 10 coins.\nNo rounds played yet.\nGolden fish caught last round: none.\nMessages from the others last round: none.",
    ]
    for obs in observations:
        full = list(tok.apply_chat_template(pond.prompt_messages(system, obs), add_generation_prompt=True, return_dict=False))
        assert full[: len(prefix)] == prefix
        assert tok.decode(full[len(prefix) :]) == obs + "<|im_end|>\n<|im_start|>assistant\n"
