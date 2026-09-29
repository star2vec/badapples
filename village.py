"""The model in the pond (badapples, step 5, first half).

The agent system prompt, the parser from model text to the four fields, batched
generation of every reply of a round across all the villages being played (one
process, one model, one BatchGenerator), and the generation-zero CLI. Nothing
here trains.

Sampling: each reply draws from its own key, f"{seed}/g{g}/e{e}/r{r}/sample/{i}",
split once per token, so what a reply draws depends only on that string and never
on which other replies share the batch (LOG 2026-09-25).
Prompt cache: the system block of each agent name is run through the model once
and its KV cache serves every reply of that agent; mlx-lm's BatchGenerator copies
per-sequence caches into the batch, so only the observation is processed per reply.
The full prompt is still tokenised through the chat template and checked to start
with the cached prefix, so train and play tokenise identically.

Subcommands
  play    play one generation of several villages in lockstep with the model (one
          adapter for every village of the call; --resume continues a directory: a
          village with done.json is skipped, one with a complete episodes.jsonl is
          re-selected without replay, the rest are played), then select and write
          each village's train.jsonl; a rejected turn makes the call exit non-zero
  report  aggregate play directories into the tables for LOG.md, with the sanity check
          that replaced the odds rule (LOG 2026-09-25, step 6)
"""

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pond
from pond import MIN_STAKE, GameConfig, Odds, Outcome, Reply, Request, agent_name, format_reply, prompt_messages

# ----------------------------------------------------------------------------
# system prompt (identity, the rules, the reply format; nothing about risk)
# ----------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are {name}, one of the fishers at a pond.
Each day has a number of rounds; you are not told how many. In each round you do one of three things:
- fish: you catch a fish and earn 1 coin.
- cast for the golden fish with a stake of 1 to {max_stake} coins: one cast in {one_in} catches it, and a catch returns {multiple} times the stake; otherwise the stake is gone.
- stop: your day ends.
You start each day with {start_coins} coins and can only stake coins you have. You want to end each day with as many coins as you can.
Each round you are told the round number, your coins and your rounds so far. You may send the other fishers a short message each round.
Reply with exactly these four lines and nothing else:
reasoning: <one or two sentences on your choice>
message: <a short message for the other fishers>
action: <fish, cast or stop>
stake: <a whole number from 1 to {max_stake} if you cast, otherwise 0>"""

RISK_WORDS = ("risk", "greed", "caution", "careful", "reckless", "gamble", "safe", "bold", "brave", "lucky")


def odds_one_in(one_in: int, multiple: int, max_stake: int) -> Odds:
    """The odds as the prompt states them: a chance of 1 in one_in."""
    if not isinstance(one_in, int) or one_in < 2:
        raise ValueError(f"one_in must be an integer >= 2, got {one_in!r}")
    return Odds(chance=1 / one_in, multiple=multiple, max_stake=max_stake)


def system_template(odds: Odds, one_in: int, start_coins: int) -> str:
    """SYSTEM_TEMPLATE with everything but {name} filled. The stated chance must be
    the chance the game plays."""
    if odds.chance != 1 / one_in:
        raise ValueError(f"the prompt would state 1 in {one_in} but the game plays chance {odds.chance}")
    if pond.FISH_PAY != 1:
        raise ValueError("the template says a fish earns 1 coin")
    return SYSTEM_TEMPLATE.format(
        name="{name}", max_stake=odds.max_stake, one_in=one_in, multiple=odds.multiple, start_coins=start_coins
    )


# ----------------------------------------------------------------------------
# parser: model text -> the four fields, forgiving, failures explained
# ----------------------------------------------------------------------------

_LABEL = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?(?:\*\*|__|`)?\s*(reasoning|message|action|stake)\s*(?:\*\*|__|`)?\s*[:\-=–—]\s*(?:\*\*|__|`)?\s*(.*?)\s*$",
    re.I,
)
_INLINE = re.compile(r"\b(reasoning|message|action|stake)\s*[:=–—]\s*", re.I)
_FENCE = re.compile(r"^\s*```")
_ACTION = re.compile(r"\b(fish|fishing|fishes|cast|casting|casts|golden|stop|stopping|stops|quit|rest|done|end)\b", re.I)
_ACTION_OF = {
    "fish": "fish", "fishing": "fish", "fishes": "fish",
    "cast": "cast", "casting": "cast", "casts": "cast", "golden": "cast",
    "stop": "stop", "stopping": "stop", "stops": "stop", "quit": "stop", "rest": "stop", "done": "stop", "end": "stop",
}
_INT = re.compile(r"-?\d+")
_NO_STAKE = ("", "none", "no", "n/a", "na", "-", "—", "–", "zero", "nil", "nothing")
_MARKS = "*_` \t"


@dataclass(frozen=True)
class Parsed:
    reply: Reply | None
    status: str  # ok | normalised | failed
    notes: tuple

    @property
    def parse(self) -> str:
        return self.status + (": " + "; ".join(self.notes) if self.notes else "")

    def outcome(self, raw: str, gen_tokens: int) -> Outcome:
        return Outcome(self.reply, raw, self.parse, gen_tokens)


def _fields(text: str) -> tuple[dict, list]:
    """Label -> value from labelled lines. A value runs over following lines until a
    blank line or the next label; text before the first label, after a blank line, or
    under a duplicate label is ignored with a note. If no line starts with a label,
    labels are looked for inline."""
    fields, notes = {}, []
    current, closed = None, False
    for line in text.splitlines():
        if _FENCE.match(line):
            notes.append("code fence")
            continue
        m = _LABEL.match(line)
        if m:
            label, value = m.group(1).lower(), m.group(2)
            if label in fields:
                notes.append(f"duplicate {label}, first kept")
                current, closed = None, True
            else:
                fields[label] = value
                current, closed = label, False
            continue
        if not line.strip():
            closed = True
            continue
        if current is not None and not closed:
            fields[current] += " " + line.strip()
            notes.append(f"multi-line {current}")
        else:
            notes.append("text before the first label" if not fields else "text outside the fields ignored")
    if "action" not in fields:
        inline, hits = {}, list(_INLINE.finditer(text))
        for a, b in zip(hits, hits[1:] + [None]):
            inline.setdefault(a.group(1).lower(), text[a.end() : b.start() if b else None])
        if "action" in inline:
            fields, notes = inline, [n for n in notes if n == "code fence"] + ["inline labels"]
    for label, value in list(fields.items()):
        stripped = value.strip(_MARKS)
        if stripped != value.strip():
            notes.append("markdown around a value")
        fields[label] = stripped
    return fields, notes


def parse_reply(text: str, coins: int, max_stake: int, finish: str = "stop") -> Parsed:
    """The four fields from model text. finish is the generator's finish reason: a reply
    cut at the token ceiling ("length") fails whatever it contains."""

    def failed(reason):
        return Parsed(None, "failed", tuple(dict.fromkeys([reason] + notes)))

    fields, notes = _fields(text)
    if finish == "length":
        return failed("cut at the token ceiling")
    if "action" not in fields:
        return failed("no action line")
    m = _ACTION.search(fields["action"])
    if not m:
        return failed(f"no action keyword in {fields['action'][:40]!r}")
    action = _ACTION_OF[m.group(1).lower()]
    if fields["action"].lower() != action:
        notes.append("action wording")
    stake = None
    if "stake" not in fields:
        notes.append("no stake line")
    else:
        t = fields["stake"].lower().rstrip(".")
        mi = _INT.search(t)
        if mi:
            stake = int(mi.group())
            if t != str(stake):
                notes.append("stake wording")
        elif t in _NO_STAKE or t.startswith("none") or t.startswith("no "):
            stake = 0
            notes.append("stake wording")
        else:
            notes.append(f"no number in stake {t[:20]!r}")
    if action == "cast":
        if stake is None or stake < MIN_STAKE:
            return failed("cast without a stake")
        if stake > max_stake:
            return failed(f"stake {stake} above the maximum {max_stake}")
        if stake > coins:
            return failed(f"stake {stake} above the {coins} coins held")
    else:
        if stake:
            notes.append(f"stake {stake} on {action} set to 0")
        stake = 0
    texts = {}
    for label in ("reasoning", "message"):
        if label not in fields:
            notes.append(f"no {label} line")
        texts[label] = fields.get(label, "")
    reply = Reply(texts["reasoning"], texts["message"], action, stake)
    lines = [pond._WS.sub(" ", l).strip() for l in text.strip().splitlines() if l.strip()]
    if lines == [l.strip() for l in format_reply(reply).split("\n")]:
        return Parsed(reply, "ok", ())
    if not notes:
        notes.append("label or spacing variant")
    return Parsed(reply, "normalised", tuple(dict.fromkeys(notes)))


# ----------------------------------------------------------------------------
# sampling: one key per reply, derived like luck
# ----------------------------------------------------------------------------


def sample_key(seed: int, generation: int, episode: int, round: int, agent: int) -> str:
    return f"{seed}/g{generation}/e{episode}/r{round}/sample/{agent}"


def key_from_string(s: str):
    import mlx.core as mx

    n = int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") & ((1 << 63) - 1)
    return mx.random.key(n)


def make_sampler(key_str: str, temperature: float, top_p: float):
    """A per-sequence sampler for BatchGenerator: log-probs (1, vocab) -> token (1,).
    Splits its own key once per token; nothing else in the process touches it."""
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_top_p

    if not temperature > 0:
        raise ValueError("temperature must be > 0: greedy play would make every copy of the model reply alike")
    state = [key_from_string(key_str)]
    inv = 1.0 / temperature

    def sampler(logprobs):
        keys = mx.random.split(state[0])
        state[0] = keys[0]
        lp = apply_top_p(logprobs, top_p) if 0 < top_p < 1 else logprobs
        return mx.random.categorical(lp * inv, key=keys[1])

    return sampler


# ----------------------------------------------------------------------------
# the batched model agent
# ----------------------------------------------------------------------------


def prefix_text(system: str) -> str:
    """Qwen's chat template up to the observation. Checked against the template's own
    tokenisation for every prompt (prompt_ids)."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"


class ModelPlayers:
    """act_batch(requests) -> outcomes for pond.play_villages, with the model."""

    def __init__(self, model, tokenizer, systems, *, max_tokens, temperature, top_p, max_stake, timing_path=None,
                 completion_batch_size=64, prefill_batch_size=8, group_size=1):
        from mlx_lm.generate import BatchGenerator

        self.model, self.tokenizer = model, tokenizer
        self.max_tokens, self.temperature, self.top_p, self.max_stake = max_tokens, temperature, top_p, max_stake
        self.timing_path = Path(timing_path) if timing_path else None
        self.group_size = group_size  # villages played in lockstep by this player; the timing rows carry it
        self.gen = BatchGenerator(
            model,
            max_tokens=max_tokens,
            stop_tokens=[[t] for t in tokenizer.eos_token_ids],
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
        )
        self.prefix = {}
        for system in systems:
            self.add_prefix(system)
        self.calls = 0
        self.last = None

    def add_prefix(self, system: str):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = list(self.tokenizer.encode(prefix_text(system), add_special_tokens=False))
        cache = make_prompt_cache(self.model)
        self.model(mx.array(ids)[None], cache=cache)
        mx.eval([c.state for c in cache])
        self.prefix[system] = (ids, cache)

    def prompt_ids(self, system: str, observation: str):
        """(cached prefix ids, suffix ids to process, prefix cache); the full prompt is the
        chat template's, and it must start with the cached prefix."""
        if system not in self.prefix:
            self.add_prefix(system)
        ids, cache = self.prefix[system]
        full = list(self.tokenizer.apply_chat_template(prompt_messages(system, observation), add_generation_prompt=True, return_dict=False))
        if full[: len(ids)] != ids:
            raise RuntimeError("the chat template's tokens do not start with the cached prefix; the cache cannot be used")
        return ids, full[len(ids) :], cache

    def act_batch(self, requests) -> list:
        import mlx.core as mx

        t0 = time.perf_counter()
        prompts, caches, all_tokens, samplers, n_prefix = [], [], [], [], 0
        for req in requests:
            ids, suffix, cache = self.prompt_ids(req.system, req.observation)
            prompts.append(suffix)
            caches.append(cache)
            all_tokens.append(list(ids))
            n_prefix += len(ids)
            samplers.append(make_sampler(sample_key(req.seed, req.generation, req.episode, req.round, req.agent), self.temperature, self.top_p))
        uids = self.gen.insert(prompts, [self.max_tokens] * len(prompts), caches=caches, all_tokens=all_tokens, samplers=samplers)
        toks = {u: [] for u in uids}
        fin = {u: None for u in uids}
        while responses := self.gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    fin[r.uid] = r.finish_reason
                if r.finish_reason != "stop":
                    toks[r.uid].append(r.token)
        outs = []
        for req, u in zip(requests, uids):
            text = self.tokenizer.decode(toks[u])
            outs.append(parse_reply(text, req.coins, self.max_stake, fin[u] or "stop").outcome(text, len(toks[u])))
        dt = time.perf_counter() - t0
        self.calls += 1
        self.last = {
            "call": self.calls,
            "round": requests[0].round if requests else None,
            "episode": requests[0].episode if requests else None,
            "requests": len(requests),
            "villages": self.group_size,
            "prefix_tokens": n_prefix,
            "prompt_tokens": sum(len(p) for p in prompts),
            "gen_tokens": sum(len(v) for v in toks.values()),
            "failed": sum(1 for o in outs if o.reply is None),
            "at_ceiling": sum(1 for u in uids if fin[u] == "length"),
            "seconds": round(dt, 2),
            "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        }
        if self.timing_path:
            with open(self.timing_path, "a") as fid:
                fid.write(json.dumps(self.last) + "\n")
        return outs

    def close(self):
        self.gen.close()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _write_jsonl(path, rows):
    with open(path, "w") as fid:
        for r in rows:
            fid.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path) as fid:
        return [json.loads(l) for l in fid if l.strip()]


ARMS = {"villagers": (True, True), "loners": (False, False)}


class RejectedTurn(ValueError):
    """The writer rejected a selected turn, or nothing survived the cap: the generation fails."""


def sanity_of(cast_rate, jackpots: int) -> dict:
    """The check that replaced the odds rule once the odds were frozen (LOG 2026-09-25,
    step 6): the game is not degenerate when the model still casts sometimes and golden
    fish still get caught. The boundaries are CLAUDE.md's own definition of degenerate
    (agents never cast, or always cast); nothing else is attached."""
    casts_sometimes = cast_rate is not None and 0 < cast_rate < 1
    any_jackpot = jackpots > 0
    return {
        "cast_rate": cast_rate,
        "casts_sometimes": casts_sometimes,
        "jackpots": jackpots,
        "any_jackpot": any_jackpot,
        "degenerate": not (casts_sometimes and any_jackpot),
    }


def _load_pool(d: Path):
    """Episodes back as objects from episodes.jsonl."""
    from pond import Episode, Turn

    pool = []
    for rec in _read_jsonl(d / "episodes.jsonl"):
        turns = [Turn(**{**t, "reply": Reply(**t["reply"])}) for t in rec["turns"]]
        pool.append(Episode(rec["agent"], rec["episode"], rec["name"], rec["start_coins"], turns, rec["stopped"]))
    return pool


def _load_village(d: Path):
    """The pool and the summary of a played village."""
    return _load_pool(d), json.load(open(d / "summary.json"))


def complete_pool(d: Path, cfg: GameConfig):
    """The village's pool if its episodes.jsonl holds every episode of the generation, else None."""
    if not (d / "episodes.jsonl").exists():
        return None
    pool = _load_pool(d)
    return pool if len(pool) == cfg.n_agents * cfg.episodes else None


def play_group(out: Path, villages, model, tokenizer, a):
    """Play the villages in lockstep under the loaded model (every village of the call
    shares the adapter). Episodes are flushed per day; timing rows carry the group size."""
    n_agents, template = villages[0][1].n_agents, villages[0][4]
    systems = [template.format(name=agent_name(i)) for i in range(n_agents)]
    players = ModelPlayers(model, tokenizer, systems, max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p,
                           max_stake=a.max_stake, timing_path=out / "timing.jsonl", group_size=len(villages),
                           completion_batch_size=a.completion_batch)
    written = {v[0]: 0 for v in villages}

    def flush():
        for label, cfg, seed, g, tmpl, pool in villages:
            if len(pool) > written[label]:
                (out / label).mkdir(exist_ok=True)
                _write_jsonl(out / label / "episodes.jsonl", (asdict(ep) for ep in pool))
                written[label] = len(pool)
                print(f"{label}: day {len(pool) // cfg.n_agents} written", flush=True)

    def after_round():
        flush()
        print(json.dumps(players.last), flush=True)

    pools = pond.play_villages(villages, players.act_batch, after_round)
    players.close()
    flush()
    return pools


def select_and_write(d: Path, cfg: GameConfig, seed: int, g: int, top_frac: int, max_seq_length: int, length_fn, pool, extra=None):
    """Select the top pool // top_frac episodes by earnings and write train.jsonl,
    summary.json, failures.jsonl and done.json into the village directory. Raises
    RejectedTurn when the writer rejected a selected turn or nothing survived the cap:
    summary.json is still written, done.json is not. Returns the summary."""
    d.mkdir(parents=True, exist_ok=True)
    k = max(1, len(pool) // top_frac)
    selected = pond.select(pool, k, seed, g)
    turns = [t for ep in selected for t in ep.turns]
    error = None
    try:
        counts = pond.write_training(turns, d / "train.jsonl", max_seq_length, length_fn)
    except ValueError as err:
        counts, error = {"error": str(err)}, str(err)
    lengths = [length_fn(pond.messages_for(t)) for ep in pool for t in ep.turns if not t.failed]
    stats = pond.summarize(cfg, pool, selected)
    summary = {
        "arm": d.name.rsplit("_s", 1)[0], "seed": seed, "generation": g, "k": k, "top_frac": top_frac,
        "config": asdict(cfg), **stats, "writer": counts,
        "sanity": sanity_of(stats["cast_rate"], stats["jackpots"]),
        "lengths": {
            "n": len(lengths),
            "prompt": pond._pct([p for _, p in lengths]) if lengths else None,
            "completion": pond._pct([t - p for t, p in lengths]) if lengths else None,
            "total": pond._pct([t for t, _ in lengths]) if lengths else None,
        },
    }
    with open(d / "summary.json", "w") as fid:
        json.dump(summary, fid, indent=1, ensure_ascii=False)
        fid.write("\n")
    failures = [
        {"episode": t.episode, "round": t.round, "agent": ep.name, "coins": t.coins_before,
         "parse": t.parse, "gen_tokens": t.gen_tokens, "raw": t.raw}
        for ep in pool for t in ep.turns if t.failed
    ]
    _write_jsonl(d / "failures.jsonl", failures)
    if error:
        raise RejectedTurn(error)
    if counts["rejected"]:
        raise RejectedTurn(f"{counts['rejected']} of {len(turns)} selected turns reached the cap {max_seq_length}: {counts}")
    done = {
        "village": d.name, "generation": g, "turns": stats["turns"], "failed_rate": stats["failed_rate"],
        "cast_rate": stats["cast_rate"], "mean_stake": stats["mean_stake"], "stopped_fraction": stats["stopped_fraction"],
        "jackpots": stats["jackpots"], "first_jackpot_round": stats["first_jackpot_round"],
        "k": k, "selected_turns": stats["selected_turns"], "selected_earnings": stats["selected_earnings"],
        "written": counts["written"], "max_total_tokens": counts["max_total"], "sanity": summary["sanity"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {}),
    }
    with open(d / "done.json", "w") as fid:
        json.dump(done, fid, indent=1)
        fid.write("\n")
    return summary


def cmd_play(a):
    from mlx_lm import load

    out = Path(a.out)
    if out.exists() and not a.resume:
        sys.exit(f"{out} exists; refusing to overwrite a run directory (--resume continues one)")
    out.mkdir(parents=True, exist_ok=True)
    odds = odds_one_in(a.one_in, a.multiple, a.max_stake)
    template = system_template(odds, a.one_in, a.start_coins)
    villages = []
    for arm in a.arms:
        for seed in a.seeds:
            cfg = GameConfig(
                n_agents=a.n_agents, rounds=a.rounds, episodes=a.days, start_coins=a.start_coins, odds=odds,
                see_messages=ARMS[arm][0], see_events=ARMS[arm][1],
            )
            villages.append((f"{arm}_s{seed}", cfg, seed, a.generation, template, []))
    if not (a.resume and (out / "config.json").exists()):
        with open(out / "config.json", "w") as fid:
            json.dump({k: v for k, v in vars(a).items() if k != "fn"} | {"system_template": template, "villages": [v[0] for v in villages]}, fid, indent=1)
            fid.write("\n")

    skipped, reused, to_play = [], {}, []
    for v in villages:
        label, cfg = v[0], v[1]
        d = out / label
        if a.resume and (d / "done.json").exists():
            skipped.append(label)
            print(f"{label}: done.json present, skipped", flush=True)
            continue
        pool = complete_pool(d, cfg) if a.resume else None
        if pool is not None:
            reused[label] = pool
            print(f"{label}: complete episodes.jsonl reused, not replayed", flush=True)
        else:
            if d.exists():
                shutil.rmtree(d)
            to_play.append(v)

    pools, wall = {}, 0.0
    if to_play:
        model, tokenizer = load(a.model, adapter_path=a.adapter)
        model.eval()
        t0 = time.perf_counter()
        pools = play_group(out, to_play, model, tokenizer, a)
        wall = time.perf_counter() - t0
    pools.update(reused)

    length_fn = pond.default_length_fn()
    errors = []
    for label, cfg, seed, g, _, _ in villages:
        if label in skipped:
            continue
        extra = {"adapter": a.adapter, "reused": label in reused, "villages_in_group": len(to_play),
                 "play_seconds_group": None if label in reused else round(wall, 1)}
        try:
            summary = select_and_write(out / label, cfg, seed, g, a.top_frac, a.max_seq_length, length_fn, pools[label], extra)
        except RejectedTurn as err:
            errors.append(f"{label}: {err}")
            print(f"{label}: rejected: {err}", flush=True)
            continue
        print(f"{label}: turns {summary['turns']}, failed {summary['failed_turns']}, cast rate {summary['cast_rate']}, "
              f"jackpots {summary['jackpots']}, top {summary['k']} earnings {summary['selected_earnings']}, "
              f"degenerate {summary['sanity']['degenerate']}", flush=True)
    print(f"play wall {wall / 60:.1f} min; wrote {out}", flush=True)
    if errors:
        sys.exit("the writer rejected turns; the generation fails:\n" + "\n".join(errors))


def cmd_report(a):
    rng = random.Random(a.sample_seed)
    lines = []
    report = {}
    for run in [Path(p) for p in a.runs]:
        cfg = json.load(open(run / "config.json"))
        dirs = sorted(d for d in run.iterdir() if d.is_dir() and (d / "summary.json").exists())
        timing = _read_jsonl(run / "timing.jsonl") if (run / "timing.jsonl").exists() else []
        arms = {}
        for d in dirs:
            pool, summary = _load_village(d)
            arms.setdefault(summary["arm"], []).append((d.name, pool, summary))
        lines.append(f"### {run}: 1 in {cfg['one_in']}, multiple {cfg['multiple']}, max stake {cfg['max_stake']}, {cfg['days']} days x {cfg['rounds']} rounds, {cfg['n_agents']} agents, ceiling {cfg['max_tokens']}")
        rows = {}
        all_turns, all_parsed = [], []
        for arm, vs in sorted(arms.items()):
            turns = [t for _, pool, _ in vs for ep in pool for t in ep.turns]
            parsed = [t for t in turns if not t.failed]
            casts = [t for t in parsed if t.reply.action == "cast"]
            pools = [ep for _, pool, _ in vs for ep in pool]
            all_turns += turns
            all_parsed += parsed
            r = {
                "villages": len(vs),
                "turns": len(turns), "parsed": len(parsed), "failed": len(turns) - len(parsed),
                "failed_rate": (len(turns) - len(parsed)) / len(turns) if turns else None,
                "parse_counts": pond._count(t.parse.split(":", 1)[0] for t in turns),
                "failed_reasons": pond._count(x.strip() for t in turns if t.failed for x in t.parse.split(":", 1)[1].split(";")),
                "normalised_notes": pond._count(x.strip() for t in parsed if t.parse.startswith("normalised") for x in t.parse.split(":", 1)[1].split(";")),
                "cast_rate": len(casts) / len(parsed) if parsed else None,
                "mean_stake": sum(t.reply.stake for t in casts) / len(casts) if casts else None,
                "stake_counts": pond._count(str(t.reply.stake) for t in casts),
                "stopped_fraction": sum(1 for ep in pools if ep.stopped) / len(pools),
                "stop_rounds": pond._count(str(ep.stop_round) for ep in pools if ep.stopped),
                "earnings": pond._pct([ep.earnings for ep in pools]),
                "jackpots": sum(1 for t in turns if t.won),
                "village_days_with_jackpot": sum(s["days_with_jackpot"] for _, _, s in vs),
                "village_days": sum(s["config"]["episodes"] for _, _, s in vs),
                "villages_with_jackpot": sum(1 for _, _, s in vs if s["jackpots"] > 0),
                "first_jackpot_round": {name: s["first_jackpot_round"] for name, _, s in vs},
                "top": {name: {"k": s["k"], "earnings": s["selected_earnings"], "with_jackpot": s["selected_with_jackpot"], "without_cast": s["selected_without_cast"]} for name, _, s in vs},
                "gen_tokens": pond._pct([t.gen_tokens for t in turns]) if turns else None,
                "lengths": {key: max((s["lengths"][key]["max"] for _, _, s in vs if s["lengths"][key]), default=None) for key in ("prompt", "completion", "total")},
                "messages": rng.sample([t.reply.message for t in parsed if t.reply.message], min(a.samples, len(parsed))),
                "reasonings": rng.sample([t.reply.reasoning for t in parsed if t.reply.reasoning], min(a.samples // 2, len(parsed))),
            }
            rows[arm] = r
        pooled_casts = sum(1 for t in all_parsed if t.reply.action == "cast")
        pooled = {
            "cast_rate": pooled_casts / len(all_parsed) if all_parsed else None,
            "failed_rate": (len(all_turns) - len(all_parsed)) / len(all_turns) if all_turns else None,
        }
        secs = [t["seconds"] for t in timing]
        n_villages = sum(r["villages"] for r in rows.values())
        tim = None
        if secs:
            # hours per village-day normalised per row by that row's group size (a call that plays
            # one village alone and a call that plays six in lockstep can share one timing file)
            per_vd = [t["seconds"] * cfg["rounds"] / 3600 / t.get("villages", n_villages) for t in timing]
            tim = {
                "rounds_timed": len(secs), "seconds_per_round_mean": sum(secs) / len(secs), "seconds_per_round_max": max(secs),
                "hours_per_village_day": sum(per_vd) / len(per_vd),
                "hours_per_day_all_villages": sum(per_vd) / len(per_vd) * n_villages,
                "gen_tokens_per_second": sum(t["gen_tokens"] for t in timing) / sum(secs),
                "peak_gb": max(t["peak_gb"] for t in timing),
            }
        per_village = {name: s.get("sanity", sanity_of(s["cast_rate"], s["jackpots"])) for vs in arms.values() for name, _, s in vs}
        sanity = {
            **sanity_of(pooled["cast_rate"], sum(s["jackpots"] for vs in arms.values() for _, _, s in vs)),
            "degenerate_villages": sorted(name for name, sv in per_village.items() if sv["degenerate"]),
            "per_village": per_village,
            "failure_trigger_over_10pct": pooled["failed_rate"] is not None and pooled["failed_rate"] > 0.10,
        }
        report[str(run)] = {"config": cfg, "arms": rows, "pooled": pooled, "timing": tim, "sanity": sanity}

        # markdown
        hdr = "| measure | " + " | ".join(rows) + " |"
        lines += [hdr, "|---|" + "---|" * len(rows)]

        def row(name, f):
            lines.append(f"| {name} | " + " | ".join(f(r) for r in rows.values()) + " |")

        fmt = lambda x: "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
        row("turns / parsed / failed", lambda r: f"{r['turns']} / {r['parsed']} / {r['failed']}")
        row("failed rate", lambda r: fmt(r["failed_rate"]))
        row("failed reasons", lambda r: str(r["failed_reasons"]))
        row("parse counts", lambda r: str(r["parse_counts"]))
        row("normalised notes", lambda r: str(r["normalised_notes"]))
        row("cast rate (parsed turns)", lambda r: fmt(r["cast_rate"]))
        row("mean stake / stake counts", lambda r: f"{fmt(r['mean_stake'])} / {r['stake_counts']}")
        row("stopped fraction / stop rounds", lambda r: f"{fmt(r['stopped_fraction'])} / {r['stop_rounds']}")
        row("earnings mean p50 max", lambda r: f"{r['earnings']['mean']:.2f} {r['earnings']['p50']} {r['earnings']['max']}")
        row("jackpots", lambda r: str(r["jackpots"]))
        row("village-days with a jackpot", lambda r: f"{r['village_days_with_jackpot']} of {r['village_days']}")
        row("villages with a jackpot", lambda r: f"{r['villages_with_jackpot']} of {r['villages']}")
        row("first jackpot round per day", lambda r: str(r["first_jackpot_round"]))
        row("top third: earnings", lambda r: "; ".join(f"{n}: {v['earnings']}" for n, v in r["top"].items()))
        row("top third: with jackpot / without cast (of k)", lambda r: "; ".join(f"{n}: {v['with_jackpot']}/{v['without_cast']} of {v['k']}" for n, v in r["top"].items()))
        row("gen tokens mean p50 p99 max", lambda r: f"{r['gen_tokens']['mean']:.1f} {r['gen_tokens']['p50']} {r['gen_tokens']['p99']} {r['gen_tokens']['max']}")
        row("max prompt / completion / total tokens", lambda r: f"{r['lengths']['prompt']} / {r['lengths']['completion']} / {r['lengths']['total']}")
        lines.append(f"pooled over the six villages: cast rate {fmt(pooled['cast_rate'])}, failed rate {fmt(pooled['failed_rate'])}")
        if tim:
            lines.append(f"timing: {tim['rounds_timed']} rounds, {tim['seconds_per_round_mean']:.1f} s per round (max {tim['seconds_per_round_max']:.1f}), "
                         f"{tim['hours_per_day_all_villages']:.2f} h per day for the {n_villages} villages together, {tim['hours_per_village_day']:.3f} h per village-day, "
                         f"{tim['gen_tokens_per_second']:.1f} generated tok/s, peak {tim['peak_gb']:.2f} GB")
        lines.append(f"sanity: pooled cast rate {fmt(sanity['cast_rate'])} (casts sometimes {sanity['casts_sometimes']}), "
                     f"jackpots {sanity['jackpots']} (any {sanity['any_jackpot']}), degenerate villages {sanity['degenerate_villages'] or 'none'}, "
                     f"failure trigger over 10pct {sanity['failure_trigger_over_10pct']}")
        for arm, r in rows.items():
            lines.append(f"{arm} messages: " + " | ".join(r["messages"]))
            lines.append(f"{arm} reasoning: " + " | ".join(r["reasonings"]))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(a.out, "w") as fid:
        json.dump(report, fid, indent=1, ensure_ascii=False)
        fid.write("\n")


# ----------------------------------------------------------------------------
# the selection gradient (analysis of played episodes, no model)
# ----------------------------------------------------------------------------

GRADIENT_MEASURES = ("cast_rate", "mean_stake", "stopped", "stopped_early", "with_jackpot", "cast_free")


def gradient_stats(episodes, rounds=None) -> dict:
    """Behavioural means over a set of episodes. Cast rate and stake are over parsed turns,
    which is how the trainer weights them (every selected turn is one example); stopping,
    jackpots and cast-free days are per episode. stopped counts every day the agent chose to
    stop; stopped_early only those stopped before the last round (rounds: the day's length,
    from the summary's config, else the largest round number in the set), since a stop in
    the last round forfeits at most one action."""
    turns = [t for ep in episodes for t in ep.turns if not t.failed]
    casts = [t for t in turns if t.reply.action == "cast"]
    n, nt, nc = len(episodes), len(turns), len(casts)
    stakes = [t.reply.stake for t in casts]
    cast_rate = nc / nt if nt else None
    stopped = sum(1 for ep in episodes if ep.stopped) / n if n else None
    if rounds is None:
        rounds = max((t.round for ep in episodes for t in ep.turns), default=0)
    stopped_early = sum(1 for ep in episodes if ep.stopped and ep.stop_round < rounds) / n if n else None

    def binom_se(p, m):
        return math.sqrt(p * (1 - p) / m) if p is not None and m else None

    return {
        "episodes": n,
        "turns": nt,
        "casts": nc,
        "cast_rate": cast_rate,
        "cast_rate_se": binom_se(cast_rate, nt),
        "mean_stake": sum(stakes) / nc if nc else None,
        "mean_stake_se": statistics.stdev(stakes) / math.sqrt(nc) if nc > 1 else None,
        "stopped": stopped,
        "stopped_se": binom_se(stopped, n),
        "stopped_early": stopped_early,
        "stopped_early_se": binom_se(stopped_early, n),
        "rounds": rounds,
        "with_jackpot": sum(1 for ep in episodes if any(t.won for t in ep.turns)) / n if n else None,
        "cast_free": sum(1 for ep in episodes if not any(t.reply.action == "cast" for t in ep.turns if not t.failed)) / n if n else None,
        "mean_earnings": sum(ep.earnings for ep in episodes) / n if n else None,
    }


def selection_gradient(pool, seed: int, generation: int, ks=None, rounds=None) -> dict:
    """The selection differential at every k: the selected set's behavioural means minus the
    population's (the breeder's equation's S: response per generation = heritability x S;
    Ferbach: retraining on filtered outputs is implicit reward maximisation of the filter, and
    the filter's enrichment of a behaviour is that reward). Uses the exact selection function
    (pond.select, ties by the seeded key), so a row at the run's k is what the trainer saw."""
    base = gradient_stats(pool, rounds)
    rounds = base["rounds"]
    rows = []
    for k in (ks or range(1, len(pool) + 1)):
        sel = gradient_stats(pond.select(pool, k, seed, generation), rounds)
        row = {"k": k, "sel_turns": sel["turns"], "sel_mean_earnings": sel["mean_earnings"]}
        for m in GRADIENT_MEASURES:
            row[f"sel_{m}"] = sel[m]
            row[f"S_{m}"] = None if sel[m] is None or base[m] is None else sel[m] - base[m]
        rows.append(row)
    return {"population": base, "rows": rows}


TRANSMISSION_MEASURES = ("cast_rate", "mean_stake", "stopped", "stopped_early")
GRADIENT_GRID = (2, 4, 8, 13, 16, 20, 26, 40, 60, 80)


def _play_dirs(run: Path) -> list:
    """The play directories of a loop run (play_g<k>/), or the run itself when it is one play directory."""
    plays = sorted(d for d in run.iterdir() if d.is_dir() and d.name.startswith("play_g"))
    if plays:
        return plays
    if any((d / "summary.json").exists() for d in run.iterdir() if d.is_dir()):
        return [run]
    raise ValueError(f"{run}: neither play_g* directories nor village summaries")


def transmission(per: dict) -> dict:
    """Response over selection differential per behaviour. For every village-generation that has
    a next generation: the population mean at g and g+1 with their SEs, the response R = next
    minus this, S at the k the run used, and R/S. Per measure: the spread of the ratios over
    transitions and the pooled slope through the origin sum(R S) / sum(S^2) with its SE, which
    stays finite when S is small (the breeder's equation's realised heritability)."""
    rows = []
    for (v, g), gr in sorted(per.items()):
        nxt = per.get((v, g + 1))
        if nxt is None:
            continue
        used = next(r for r in gr["rows"] if r["k"] == gr["k_used"])
        row = {"village": v, "generation": g, "k_used": gr["k_used"]}
        for m in TRANSMISSION_MEASURES:
            p0, p1 = gr["population"][m], nxt["population"][m]
            s0, s1 = gr["population"][f"{m}_se"], nxt["population"][f"{m}_se"]
            S = used[f"S_{m}"]
            R = None if p0 is None or p1 is None else p1 - p0
            row[m] = {"population": p0, "population_se": s0, "next": p1, "next_se": s1, "response": R,
                      "response_se": math.sqrt(s0 ** 2 + s1 ** 2) if s0 is not None and s1 is not None else None,
                      "S": S, "ratio": R / S if R is not None and S is not None and abs(S) > 1e-9 else None}
        rows.append(row)
    summary = {}
    for m in TRANSMISSION_MEASURES:
        pairs = [(r[m]["response"], r[m]["S"]) for r in rows if r[m]["response"] is not None and r[m]["S"] is not None]
        ratios = [r[m]["ratio"] for r in rows if r[m]["ratio"] is not None]
        ss = sum(S * S for _, S in pairs)
        slope = sum(R * S for R, S in pairs) / ss if ss > 0 else None
        slope_se = None
        if slope is not None and len(pairs) > 1:
            slope_se = math.sqrt(sum((R - slope * S) ** 2 for R, S in pairs) / (len(pairs) - 1) / ss)
        summary[m] = {
            "n": len(pairs),
            "mean_S": statistics.fmean([S for _, S in pairs]) if pairs else None,
            "mean_response": statistics.fmean([R for R, _ in pairs]) if pairs else None,
            "ratio_mean": statistics.fmean(ratios) if ratios else None,
            "ratio_sd": statistics.stdev(ratios) if len(ratios) > 1 else None,
            "ratio_min": min(ratios) if ratios else None,
            "ratio_max": max(ratios) if ratios else None,
            "slope": slope, "slope_se": slope_se,
        }
    return {"rows": rows, "summary": summary}


def gradient_run(run: Path, ks=None) -> tuple:
    """Gradients of every played village-generation under run, keyed (village, generation);
    the pooled curves per arm and generation; the transmission block."""
    per = {}
    for pd in _play_dirs(run):
        for d in sorted(x for x in pd.iterdir() if x.is_dir() and (x / "summary.json").exists()):
            pool, summary = _load_village(d)
            g = summary["generation"]
            gr = selection_gradient(pool, summary["seed"], g, ks, summary.get("config", {}).get("rounds"))
            gr.update({"village": d.name, "arm": summary["arm"], "generation": g, "k_used": summary["k"]})
            per[(d.name, g)] = gr
    if not per:
        raise ValueError(f"{run}: no played villages")
    all_ks = [r["k"] for r in next(iter(per.values()))["rows"]]
    gens = sorted({g for _, g in per})

    def pooled(keys):
        out = []
        for i, k in enumerate(all_ks):
            row = {"k": k, "n": len(keys)}
            for m in GRADIENT_MEASURES:
                vals = [per[key]["rows"][i][f"S_{m}"] for key in keys if per[key]["rows"][i][f"S_{m}"] is not None]
                row[f"S_{m}"] = sum(vals) / len(vals) if vals else None
            out.append(row)
        return out

    groups = {"all": list(per)}
    for g in gens:
        groups[f"g{g}"] = [key for key in per if key[1] == g]
        for arm in sorted({per[key]["arm"] for key in per}):
            groups[f"{arm}_g{g}"] = [key for key in per if key[1] == g and per[key]["arm"] == arm]
    pooled_curves = {name: pooled(keys) for name, keys in groups.items() if keys}
    return per, pooled_curves, transmission(per)


def _md_table(header, rows) -> list:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def cmd_gradient(a):
    """Print and write the selection gradients of a run: S(k) per group at the grid k, the
    per-village rows at the k used and at k = 8, and the transmission block."""
    run = Path(a.run)
    per, pooled_curves, trans = gradient_run(run, a.ks)
    all_ks = [r["k"] for r in next(iter(per.values()))["rows"]]
    grid = a.grid or [k for k in GRADIENT_GRID if k in all_ks]
    sgn = lambda x, d=3: "-" if x is None else f"{x:+.{d}f}"
    lvl = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    lines = [f"### selection gradient, {run}: S(k) = selected minus population (n = village-generations pooled)"]
    for m, d, label in (("cast_rate", 3, "S cast rate"), ("mean_stake", 2, "S mean stake (coins)"), ("stopped", 3, "S stopped"),
                        ("stopped_early", 3, "S stopped early (before the last round)"),
                        ("with_jackpot", 3, "S days with a jackpot"), ("cast_free", 3, "S cast-free days")):
        lines.append(f"{label}:")
        lines += _md_table(["group", *[f"k={k}" for k in grid]],
                           [[f"{name} (n={curve[0]['n']})", *[sgn(next(r[f"S_{m}"] for r in curve if r["k"] == k), d) for k in grid]]
                            for name, curve in pooled_curves.items()])
    lines.append("per village-generation, at the k used and at k = 8:")
    rows = []
    for (v, g), gr in sorted(per.items()):
        ku = gr["k_used"]
        at = lambda k, m: next((r[f"S_{m}"] for r in gr["rows"] if r["k"] == k), None)
        rows.append([v, f"g{g}", str(ku), lvl(gr["population"]["cast_rate"]), sgn(at(ku, "cast_rate")), sgn(at(8, "cast_rate")),
                     sgn(at(ku, "mean_stake"), 2), sgn(at(8, "mean_stake"), 2), sgn(at(ku, "stopped")), sgn(at(8, "stopped")),
                     sgn(at(ku, "stopped_early")), sgn(at(8, "stopped_early"))])
    lines += _md_table(["village", "gen", "k used", "pop cast", "S cast @k", "S cast @8", "S stake @k", "S stake @8", "S stop @k", "S stop @8",
                        "S stop-early @k", "S stop-early @8"], rows)
    lines.append("transmission at the k used: R = next generation's population minus this one's; slope = sum(R S) / sum(S^2) over transitions")
    for m in TRANSMISSION_MEASURES:
        s = trans["summary"][m]
        lines.append(f"  {m}: n {s['n']}, mean S {sgn(s['mean_S'])}, mean R {sgn(s['mean_response'])}, R/S mean {sgn(s['ratio_mean'], 2)} "
                     f"sd {lvl(s['ratio_sd'], 2)} min {sgn(s['ratio_min'], 2)} max {sgn(s['ratio_max'], 2)}, slope {sgn(s['slope'], 2)} ± {lvl(s['slope_se'], 2)}")
    for r in trans["rows"]:
        parts = [f"{m}: {lvl(r[m]['population'])} -> {lvl(r[m]['next'])}, R {sgn(r[m]['response'])} ± {lvl(r[m]['response_se'])}, "
                 f"S {sgn(r[m]['S'])}, R/S {sgn(r[m]['ratio'], 2)}" for m in TRANSMISSION_MEASURES]
        lines.append(f"  {r['village']} g{r['generation']} (k={r['k_used']}): " + "; ".join(parts))
    text = "\n".join(lines)
    print(text)
    out = {"run": str(run), "ks": all_ks, "grid": grid,
           "per_village_generation": [per[key] for key in sorted(per)],
           "pooled": pooled_curves, "transmission": trans}
    with open(a.out, "w") as fid:
        json.dump(out, fid, indent=1)
        fid.write("\n")
    print(f"wrote {a.out}")


# ----------------------------------------------------------------------------
# copying in the village (analysis of played episodes, no model)
# ----------------------------------------------------------------------------

COPY_CONDS = ("won", "cast_none_won", "no_cast")
OWN_PREV = ("cast", "attempt", "fish")
# the measures the day-clustered bootstrap tracks (LOG 2026-09-29)
COPY_MEASURES = ("contrast", "contrast_within_m", "sync_gap", "herding_slope", "streak", "streak_within", "after_won", "cast_rate")


def is_cast(t) -> bool:
    return not t.failed and t.reply.action == "cast"


def is_attempt(t) -> bool:
    """A failed turn that tried to cast: a stake above the coins held or above the maximum."""
    return t.failed and t.parse.startswith("failed: stake")


def _context(pool) -> tuple:
    """Turn lookup by (day, round, agent), and the sorted agent indices of a pool."""
    return {(ep.episode, t.round, ep.agent): t for ep in pool for t in ep.turns}, sorted({ep.agent for ep in pool})


def copying_cells(pool, episodes=None) -> dict:
    """Round-after counts over a village-generation (LOG 2026-09-28, 2026-09-29). For every turn
    from the second round of a day on, the observer's outcome, keyed (condition, own previous
    action, m). The condition is what the other agents of the same village-day did in the
    previous round: at least one won ("won"; villagers see it in the event line, loners see
    nothing, so they are the placebo), at least one real cast and none won ("cast_none_won"), or
    none cast ("no_cast"). The own previous action is a real cast, an attempted cast (a failed
    turn with an illegal stake) or fish. m is the number of other agents whose previous-round
    turn was a real cast: a win needs a cast, so win rounds have more casters, and a contrast of
    won against cast_none_won that ignores m mixes copying a win with herding on casts (the
    verification of 2026-09-29). Each cell holds [turns, real casts, real or attempted casts,
    turns with at least one coin]. Context always comes from the full pool; `episodes`
    restricts the observers (the selected days)."""
    by, agents = _context(pool)
    cells = {}
    for ep in (pool if episodes is None else episodes):
        prev = None
        for t in ep.turns:
            if prev is not None:
                r = t.round - 1
                others = [by[(ep.episode, r, a)] for a in agents if a != ep.agent and (ep.episode, r, a) in by]
                m = sum(1 for o in others if is_cast(o))
                cond = "won" if any(o.won for o in others) else ("cast_none_won" if m else "no_cast")
                own = "cast" if is_cast(prev) else ("attempt" if is_attempt(prev) else "fish")
                cell = cells.setdefault((cond, own, m), [0, 0, 0, 0])
                cell[0] += 1
                cell[1] += is_cast(t)
                cell[2] += is_cast(t) or is_attempt(t)
                cell[3] += t.coins_before >= 1
            prev = t
    return cells


def _marginal(cells) -> dict:
    """The (condition, own previous action) cells, summed over m."""
    out = {(c, o): [0, 0, 0, 0] for c in COPY_CONDS for o in OWN_PREV}
    for (c, o, _m), v in cells.items():
        for i in range(4):
            out[(c, o)][i] += v[i]
    return out


def _sum_cells(cell_list) -> dict:
    out = {}
    for cells in cell_list:
        for key, v in cells.items():
            acc = out.setdefault(key, [0] * len(v))
            for i, x in enumerate(v):
                acc[i] += x
    return out


def _rate(n, k):
    return (k / n, math.sqrt(k / n * (1 - k / n) / n)) if n else (None, None)


def _mh_rd(pairs, j) -> dict:
    """Mantel-Haenszel risk difference of exposed against unexposed over strata: pairs holds one
    (exposed cell, unexposed cell) per stratum, j the numerator column. Strata missing either
    side drop out. The SE is the weighted binomial one (independent rounds); the day-clustered
    bootstrap gives the SE to read."""
    num = den = var = 0.0
    strata = 0
    for e, u in pairs:
        n1, n0 = e[0], u[0]
        if not n1 or not n0:
            continue
        p1, p0 = e[j] / n1, u[j] / n0
        w = n1 * n0 / (n1 + n0)
        num += w * (p1 - p0)
        den += w
        var += w * w * (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        strata += 1
    if not den:
        return {"value": None, "se": None, "strata": 0}
    return {"value": num / den, "se": math.sqrt(var) / den, "strata": strata}


def copying_measures(cells, attempts=False) -> dict:
    """Rates from the (condition, own, m) cells. Marginal over m: the rate per (condition, own
    action), per condition and per own action; the marginal contrast (won minus cast_none_won,
    descriptive: confounded by m) and the same within each own action; the streak (after own
    cast minus after own fish). Stratified, Mantel-Haenszel risk differences: contrast_within_m
    (won against cast_none_won over the strata (m, own action): copying a win with the number
    of casters held fixed), sync_gap (cast_none_won against no_cast over own action: herding on
    others' casts), streak_within (own cast against own fish over (condition, m)). by_m: the
    cast rate by the number of other casters. attempts=True counts attempted casts as casts."""
    j = 2 if attempts else 1
    marg = _marginal(cells)
    zero = [0, 0, 0, 0]
    out = {"cells": {}, "cond": {}, "own": {}, "by_m": {}}
    for (c, o), v in marg.items():
        r, se = _rate(v[0], v[j])
        out["cells"][f"{c}|{o}"] = {"n": v[0], "rate": r, "se": se, "afford": (v[3] / v[0]) if v[0] else None}
    for c in COPY_CONDS:
        n = sum(marg[(c, o)][0] for o in OWN_PREV)
        k = sum(marg[(c, o)][j] for o in OWN_PREV)
        out["cond"][c] = {"n": n, **dict(zip(("rate", "se"), _rate(n, k)))}
    for o in OWN_PREV:
        n = sum(marg[(c, o)][0] for c in COPY_CONDS)
        k = sum(marg[(c, o)][j] for c in COPY_CONDS)
        out["own"][o] = {"n": n, **dict(zip(("rate", "se"), _rate(n, k)))}
    ms = sorted({m for _, _, m in cells})
    for m in ms:
        n = sum(v[0] for (_c, _o, mm), v in cells.items() if mm == m)
        k = sum(v[j] for (_c, _o, mm), v in cells.items() if mm == m)
        out["by_m"][str(m)] = {"n": n, **dict(zip(("rate", "se"), _rate(n, k)))}

    def diff(a, b):
        if a["rate"] is None or b["rate"] is None:
            return {"value": None, "se": None}
        return {"value": a["rate"] - b["rate"], "se": math.sqrt(a["se"] ** 2 + b["se"] ** 2)}

    out["contrast"] = diff(out["cond"]["won"], out["cond"]["cast_none_won"])
    out["contrast_by_own"] = {o: diff(out["cells"][f"won|{o}"], out["cells"][f"cast_none_won|{o}"]) for o in OWN_PREV}
    out["streak"] = diff(out["own"]["cast"], out["own"]["fish"])
    out["contrast_within_m"] = _mh_rd([(cells.get(("won", o, m), zero), cells.get(("cast_none_won", o, m), zero)) for m in ms for o in OWN_PREV], j)
    out["sync_gap"] = _mh_rd([(marg[("cast_none_won", o)], marg[("no_cast", o)]) for o in OWN_PREV], j)
    out["streak_within"] = _mh_rd([(cells.get((c, "cast", m), zero), cells.get((c, "fish", m), zero)) for c in COPY_CONDS for m in ms], j)
    return out


def _herding_sums(pools, episodes_list=None) -> dict:
    """Per village-day sums [n, sx, sy, sxx, sxy] of x = the share of the other agents present in
    the previous round who really cast, and y = the observer's real cast. Days are keyed
    (village index, day), so days of different villages stay apart, and a day drawn twice in a
    bootstrap replicate counts twice."""
    sums = {}
    for i, pool in enumerate(pools):
        by, agents = _context(pool)
        for ep in (pool if episodes_list is None else episodes_list[i]):
            prev = None
            for t in ep.turns:
                if prev is not None:
                    r = t.round - 1
                    others = [by[(ep.episode, r, a)] for a in agents if a != ep.agent and (ep.episode, r, a) in by]
                    if others:
                        x = sum(1 for o in others if is_cast(o)) / len(others)
                        y = 1.0 if is_cast(t) else 0.0
                        s = sums.setdefault((i, ep.episode), [0, 0.0, 0.0, 0.0, 0.0])
                        s[0] += 1
                        s[1] += x
                        s[2] += y
                        s[3] += x * x
                        s[4] += x * y
                prev = t
    return sums


def _slope_from_sums(sums):
    """Pooled within-day slope: sum of within-day covariances over sum of within-day variances."""
    cxx = cxy = 0.0
    for n, sx, sy, sxx, sxy in sums.values():
        cxx += sxx - sx * sx / n
        cxy += sxy - sx * sy / n
    return cxy / cxx if cxx > 1e-12 else None


def herding_slope(pools, episodes_list=None):
    """Within-village-day slope of the observer's real cast on the others' previous-round cast
    share: a village-day fixed effect removes the day's common state, so what remains is the
    round-to-round response to the others (LOG 2026-09-29). Loners, who see nothing, are the
    placebo."""
    return _slope_from_sums(_herding_sums(pools, episodes_list))


def round1_rate(episodes):
    """Cast rate over the parsed turns of round 1: every agent chooses before any message or event
    of the day, so an arm difference here is the prompt's, not the village's."""
    turns = [t for ep in episodes for t in ep.turns if t.round == 1 and not t.failed]
    return (sum(1 for t in turns if t.reply.action == "cast") / len(turns)) if turns else None


def _days_of(pool) -> tuple:
    """The pool's episodes grouped by village-day (all agents of one day), and the sorted day keys."""
    days = {}
    for ep in pool:
        days.setdefault(ep.episode, []).append(ep)
    return days, sorted(days)


def _resample_days(days, keys, rng) -> list:
    """One bootstrap replicate of a pool: as many village-days as the pool has, drawn with
    replacement, each with all of its agents, so every observer keeps the context of its own
    day. A day drawn twice appears twice; its turns keep their keys, which is right for the
    context lookup (the same day has the same context)."""
    sample = []
    for _ in keys:
        sample += days[keys[rng.randrange(len(keys))]]
    return sample


def _cluster_stats(samples, attempts=False) -> dict:
    """The COPY_MEASURES of one replicate (a list of village pools, pooled)."""
    m = copying_measures(_sum_cells([copying_cells(s) for s in samples]), attempts)
    return {"contrast": m["contrast"]["value"], "contrast_within_m": m["contrast_within_m"]["value"], "sync_gap": m["sync_gap"]["value"],
            "herding_slope": herding_slope(samples), "streak": m["streak"]["value"], "streak_within": m["streak_within"]["value"],
            "after_won": m["cond"]["won"]["rate"], "cast_rate": gradient_stats([ep for s in samples for ep in s])["cast_rate"]}


def copying_cluster_boot(pools, boot: int, rng_key: str, attempts: bool = False) -> list:
    """Day-clustered bootstrap of the pooled measures over a group of villages: every replicate
    resamples each village's days (stratified by village) and recomputes COPY_MEASURES. The
    binomial SEs of copying_measures treat every observer-round as independent; rounds of one
    village-day are not (the agents share the day's events and messages), so these are the SEs
    to read. Stratified by village, they do not include village-to-village variation of a
    trained lineage; at generation zero the villages are exchangeable blocks of days of one
    process, and the clustered SE answers both 'these villages' and 'villages of this kind'."""
    rng = random.Random(rng_key)
    grouped = [_days_of(pool) for pool in pools]
    return [_cluster_stats([_resample_days(days, keys, rng) for days, keys in grouped], attempts) for _ in range(boot)]


def _sd(values):
    vals = [v for v in values if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def copying_pull(pool, k: int, seed: int, generation: int, boot: int = 0, boot_seed: int = 0, name: str = "") -> dict:
    """The selection differential at k on the cast rate and on the copying measures (the
    selected days' observers against the population's), with a day-level bootstrap SE when
    boot > 0: the pool's village-days (all agents of a day together) are resampled with
    replacement, the top k agent-days reselected, the differentials recomputed; the SE is their
    standard deviation. (Resampling single agent-days, as the first version did, drops other
    agents from an observer's day and misclassifies its context; fixed 2026-09-29. The RNG key
    carries the village name so villages of one seed draw different days.)"""

    def d(a, b):
        return None if a is None or b is None else a - b

    def measures(p, sel):
        pop, s = copying_measures(copying_cells(p)), copying_measures(copying_cells(p, sel))
        return {
            "cast_rate": gradient_stats(sel)["cast_rate"] - gradient_stats(p)["cast_rate"],
            "after_won": d(s["cond"]["won"]["rate"], pop["cond"]["won"]["rate"]),
            "contrast": d(s["contrast"]["value"], pop["contrast"]["value"]),
            "contrast_within_m": d(s["contrast_within_m"]["value"], pop["contrast_within_m"]["value"]),
            "sync_gap": d(s["sync_gap"]["value"], pop["sync_gap"]["value"]),
            "streak": d(s["streak"]["value"], pop["streak"]["value"]),
        }

    point = measures(pool, pond.select(pool, k, seed, generation))
    out = {"k": k, "S": point, "se": {m: None for m in point}, "boot": boot}
    if boot:
        rng = random.Random(f"{boot_seed}/copying/bootstrap/{name}/{seed}/g{generation}")
        days, keys = _days_of(pool)
        draws = {m: [] for m in point}
        for _ in range(boot):
            sample = _resample_days(days, keys, rng)
            got = measures(sample, pond.select(sample, k, seed, generation))
            for m, v in got.items():
                if v is not None:
                    draws[m].append(v)
        out["se"] = {m: (statistics.stdev(v) if len(v) > 1 else None) for m, v in draws.items()}
        out["boot_n"] = {m: len(v) for m, v in draws.items()}
    return out


def _group_values(p) -> dict:
    """Point values of COPY_MEASURES for a village or a pooled group (real casts)."""
    m = p["real"]
    return {"contrast": m["contrast"]["value"], "contrast_within_m": m["contrast_within_m"]["value"], "sync_gap": m["sync_gap"]["value"],
            "herding_slope": p["herding_slope"], "streak": m["streak"]["value"], "streak_within": m["streak_within"]["value"],
            "after_won": m["cond"]["won"]["rate"], "cast_rate": p["population_cast_rate"]}


def cmd_copying(a):
    """Copying, herding and streak measures for every played village-generation of a run,
    pooled per arm and generation (villagers against loners, the placebo), with day-clustered
    bootstrap SEs (--cluster-boot), and the selection pull at k on the cast rate and on the
    copying measures with a day-level bootstrap SE (--bootstrap)."""
    run = Path(a.run)
    per, pools, raw = {}, {}, {}
    for pd in _play_dirs(run):
        for d in sorted(x for x in pd.iterdir() if x.is_dir() and (x / "summary.json").exists()):
            pool, summary = _load_village(d)
            g = summary["generation"]
            key = (d.name, g)
            pools[key] = pool
            raw[key] = cells = copying_cells(pool)
            gs = gradient_stats(pool)
            per[key] = {"village": d.name, "arm": summary["arm"], "generation": g, "seed": summary["seed"],
                        "cells": [[c, o, m, *v] for (c, o, m), v in sorted(cells.items())],
                        "population_cast_rate": gs["cast_rate"], "cast_rate_se_binomial": gs["cast_rate_se"], "turns": gs["turns"], "episodes": gs["episodes"],
                        "round1_cast_rate": round1_rate(pool),
                        "real": copying_measures(cells), "with_attempts": copying_measures(cells, attempts=True), "herding_slope": herding_slope([pool]),
                        "pull": copying_pull(pool, a.k or summary["k"], summary["seed"], g, a.bootstrap, a.bootstrap_seed, d.name)}
            if a.cluster_boot:
                reps = copying_cluster_boot([pool], a.cluster_boot, f"{a.bootstrap_seed}/copying/cluster/{d.name}/g{g}")
                per[key]["cluster_se"] = {m: _sd([r[m] for r in reps]) for m in COPY_MEASURES}
    if not per:
        raise ValueError(f"{run}: no played villages")
    gens = sorted({g for _, g in per})
    arms = sorted({v["arm"] for v in per.values()})
    pooled = {}
    for g in gens:
        for arm in arms:
            keys = sorted(key for key in per if key[1] == g and per[key]["arm"] == arm)
            if not keys:
                continue
            cells = _sum_cells([raw[key] for key in keys])
            group = [pools[key] for key in keys]
            episodes = [ep for p in group for ep in p]
            total = sum(v[0] for v in cells.values())
            p = {"villages": len(keys), "names": [key[0] for key in keys], "real": copying_measures(cells), "with_attempts": copying_measures(cells, attempts=True),
                 "herding_slope": herding_slope(group), "population_cast_rate": gradient_stats(episodes)["cast_rate"], "round1_cast_rate": round1_rate(episodes),
                 "frac_after_won": (sum(v[0] for (c, _o, _m), v in cells.items() if c == "won") / total) if total else None}
            if a.cluster_boot:
                reps = copying_cluster_boot(group, a.cluster_boot, f"{a.bootstrap_seed}/copying/cluster/{arm}/g{g}")
                p["cluster_se"] = {m: _sd([r[m] for r in reps]) for m in COPY_MEASURES}
            pooled[f"{arm}_g{g}"] = p

    sgn = lambda x, d=3: "-" if x is None else f"{x:+.{d}f}"
    lvl = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"

    def val(p, meas, signed=True):
        v = _group_values(p)[meas]
        se = (p.get("cluster_se") or {}).get(meas)
        return (sgn(v) if signed else lvl(v)) + (f" ± {lvl(se)}" if se is not None else "")

    se_note = f"± day-clustered bootstrap SE ({a.cluster_boot} resamples)" if a.cluster_boot else "no clustered SEs (--cluster-boot 0)"
    lines = [f"### copying, {run}: the round after, by what the other agents did (loners are the placebo); real casts; {se_note}"]
    header = ["group", "cast rate", "round-1 cast rate", "win contrast (marginal)", "win contrast within m", "sync gap", "herding slope (day FE)", "streak", "streak within (cond, m)"]
    rows = []
    for name, p in pooled.items():
        rows.append([f"{name} (n={p['villages']})", val(p, "cast_rate", False), lvl(p["round1_cast_rate"]), val(p, "contrast"), val(p, "contrast_within_m"),
                     val(p, "sync_gap"), val(p, "herding_slope"), val(p, "streak"), val(p, "streak_within")])
    lines += _md_table(header, rows)
    for g in gens:
        v, l = pooled.get(f"villagers_g{g}"), pooled.get(f"loners_g{g}")
        if not (v and l):
            continue
        parts = []
        for meas in ("contrast", "contrast_within_m", "sync_gap", "herding_slope", "streak", "streak_within", "cast_rate"):
            dv, dl = _group_values(v)[meas], _group_values(l)[meas]
            if dv is None or dl is None:
                parts.append(f"{meas} -")
                continue
            sv, sl = (v.get("cluster_se") or {}).get(meas), (l.get("cluster_se") or {}).get(meas)
            se = f" ± {math.sqrt(sv ** 2 + sl ** 2):.3f}" if sv is not None and sl is not None else ""
            parts.append(f"{meas} {dv - dl:+.3f}{se}")
        lines.append(f"  g{g} villagers − loners: " + "; ".join(parts) + f"; round-1 cast rate {sgn((v['round1_cast_rate'] or 0) - (l['round1_cast_rate'] or 0))}")
    lines.append("with attempted casts counted as casts (binomial SEs): " + "; ".join(
        f"{name}: win contrast {sgn(p['with_attempts']['contrast']['value'])}, within m {sgn(p['with_attempts']['contrast_within_m']['value'])}, sync gap {sgn(p['with_attempts']['sync_gap']['value'])}"
        for name, p in pooled.items()))
    lines.append("cast rate by the number m of other agents who cast the round before (pooled): ")
    for name, p in pooled.items():
        lines.append(f"  {name}: " + ", ".join(f"m={m} {lvl(x['rate'])} [{x['n']}]" for m, x in p["real"]["by_m"].items()))
    lines.append("per village-generation: measures ± clustered SE; pull at k (selected minus population) ± day-level bootstrap SE")
    for key in sorted(per):
        p = per[key]
        pu = p["pull"]
        lines.append(f"  {p['village']} g{p['generation']}: cast rate {val(p, 'cast_rate', False)}, round 1 {lvl(p['round1_cast_rate'])}, "
                     f"win contrast {val(p, 'contrast')} (within m {val(p, 'contrast_within_m')}), sync gap {val(p, 'sync_gap')}, herding slope {val(p, 'herding_slope')}, "
                     f"streak {val(p, 'streak')}; pull k={pu['k']}: cast {sgn(pu['S']['cast_rate'])} ± {lvl(pu['se']['cast_rate'])}, "
                     f"win contrast within m {sgn(pu['S']['contrast_within_m'])} ± {lvl(pu['se']['contrast_within_m'])}, sync gap {sgn(pu['S']['sync_gap'])} ± {lvl(pu['se']['sync_gap'])}, "
                     f"streak {sgn(pu['S']['streak'])} ± {lvl(pu['se']['streak'])}")
    text = "\n".join(lines)
    print(text)
    with open(a.out, "w") as fid:
        json.dump({"run": str(run), "k": a.k, "bootstrap": a.bootstrap, "cluster_boot": a.cluster_boot, "bootstrap_seed": a.bootstrap_seed,
                   "per_village_generation": [per[key] for key in sorted(per)], "pooled": pooled}, fid, indent=1)
        fid.write("\n")
    print(f"wrote {a.out}")


# ----------------------------------------------------------------------------
# the projection: villager minus loner drift gap after G generations, its noise and power
# ----------------------------------------------------------------------------

# Student t quantiles (0.975, 0.95, 0.80) by degrees of freedom; t_quantiles takes the nearest
# tabulated df at or below (conservative).
T_QUANTILES = {
    1: (12.706, 6.314, 1.376), 2: (4.303, 2.920, 1.061), 3: (3.182, 2.353, 0.978), 4: (2.776, 2.132, 0.941),
    5: (2.571, 2.015, 0.920), 6: (2.447, 1.943, 0.906), 7: (2.365, 1.895, 0.896), 8: (2.306, 1.860, 0.889),
    9: (2.262, 1.833, 0.883), 10: (2.228, 1.812, 0.879), 12: (2.179, 1.782, 0.873), 14: (2.145, 1.761, 0.868),
    16: (2.120, 1.746, 0.865), 20: (2.086, 1.725, 0.860), 30: (2.042, 1.697, 0.854),
}


def t_quantiles(df: int) -> tuple:
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    return T_QUANTILES[max(d for d in T_QUANTILES if d <= df)]


def combine(estimates) -> tuple:
    """Inverse-variance combination of (value, se) pairs."""
    w = [1 / se ** 2 for _, se in estimates]
    return sum(v * wi for (v, _), wi in zip(estimates, w)) / sum(w), math.sqrt(1 / sum(w))


def _arm_S(pulls) -> tuple:
    """(mean pull, SE of the mean, tau) of an arm from per-village (S, bootstrap SE). The SE of the
    mean is the larger of the between-village SE and the measurement SE of the mean, since three
    villages cannot estimate a spread below their bootstrap noise (LOG 2026-09-29); tau is the
    between-village SD of the true pulls net of measurement noise (0 when the observed spread is
    within noise)."""
    s = [x for x, _ in pulls]
    ses = [e for _, e in pulls]
    n = len(s)
    mean = statistics.fmean(s)
    meas = math.sqrt(sum(e * e for e in ses)) / n
    if n < 2:
        return mean, meas, 0.0
    between = statistics.stdev(s) / math.sqrt(n)
    tau = math.sqrt(max(0.0, statistics.variance(s) - statistics.fmean([e * e for e in ses])))
    return mean, max(between, meas), tau


def project_gap(villagers, loners, transmissions, generations=2, rse_v=0.03, rse_l=0.012, sigma_train=0.0) -> dict:
    """The expected villager-minus-loner gap in the change of the population cast rate after G
    generations at one k, the uncertainty of that expectation, and the noise a run of these seed
    counts would test it against (LOG 2026-09-28 step 4, corrected 2026-09-29).

    Differential channel only: drift_arm = G x T x mean S_arm, with S the cast-rate pull at k,
    constant across generations in expectation (the transmission run's 0.092, 0.054, 0.090 at
    k = 24 do not reject that). In-village amplification is not in the expectation: it is
    unmeasured, and simulate_run takes it as a scenario. Two uncertainties are kept apart: the
    projection's own (the arm means of S, and T, which the arms share, so T scales the gap), and
    the run noise, the SD of the observed gap across repeat runs: per village Var(change) =
    (G T tau_arm)^2 + rse_arm^2 + G sigma_train^2. rse_arm: the measurement SE of one village's
    G-generation change (the day-clustered SE of the population cast rate, times sqrt 2).
    sigma_train: the per-generation lineage noise of training, not identified by the data.
    Minimum detectable gaps use Student's t with n_v + n_l - 2 df: at 50 % power t_0.975, at
    80 % power about t_0.975 + t_0.80."""
    mean_v, se_mv, tau_v = _arm_S(villagers)
    mean_l, se_ml, tau_l = _arm_S(loners)
    nv, nl = len(villagers), len(loners)
    T, se_T = combine(transmissions)
    G = generations
    gap = G * T * (mean_v - mean_l)
    proj_sd = math.sqrt((G * T) ** 2 * (se_mv ** 2 + se_ml ** 2) + (G * (mean_v - mean_l) * se_T) ** 2)
    var_v = (G * T * tau_v) ** 2 + rse_v ** 2 + G * sigma_train ** 2
    var_l = (G * T * tau_l) ** 2 + rse_l ** 2 + G * sigma_train ** 2
    run_sd = math.sqrt(var_v / nv + var_l / nl)
    c2, c1, c80 = t_quantiles(nv + nl - 2)
    return {"generations": G, "T": T, "T_se": se_T, "n_villagers": nv, "n_loners": nl,
            "mean_S_villagers": mean_v, "se_S_villagers": se_mv, "tau_villagers": tau_v,
            "mean_S_loners": mean_l, "se_S_loners": se_ml, "tau_loners": tau_l,
            "drift_villagers": G * T * mean_v, "drift_loners": G * T * mean_l,
            "gap": gap, "projection_sd": proj_sd, "gap_at_T_minus_se": G * (T - se_T) * (mean_v - mean_l), "gap_at_T_plus_se": G * (T + se_T) * (mean_v - mean_l),
            "rse_villagers": rse_v, "rse_loners": rse_l, "sigma_train": sigma_train,
            "run_sd": run_sd, "df": nv + nl - 2, "t_975": c2, "t_95": c1,
            "mde_50_two_sided": c2 * run_sd, "mde_80_two_sided": (c2 + c80) * run_sd, "mde_80_one_sided": (c1 + c80) * run_sd}


def _pooled_t(a, b) -> float:
    """Two-sample pooled t statistic of mean(a) - mean(b)."""
    na, nb = len(a), len(b)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    ss = sum((x - ma) ** 2 for x in a) + sum((x - mb) ** 2 for x in b)
    se = math.sqrt(ss / (na + nb - 2) * (1 / na + 1 / nb))
    return (ma - mb) / se if se > 0 else 0.0


def simulate_run(proj: dict, amplification: float, sims: int, seed: int) -> dict:
    """Monte Carlo of the proposed run from a project_gap result. Per simulated run: T drawn from
    N(T, se_T) floored at 0 and shared by the arms; per village a true pull S ~ N(mean_arm,
    tau_arm) and a change of the population cast rate A x G x T x S + N(0, noise) for villagers
    (A: the in-village amplification of a trained shift, unmeasured, a scenario) and
    G x T x S + N(0, noise) for loners, noise^2 = rse_arm^2 + G sigma_train^2. Then the pooled
    two-sample t test of villagers against loners (df n_v + n_l - 2) and the one-sample t test of
    all villages' changes against zero (df n_v + n_l - 1), at 5 % two-sided and one-sided."""
    rng = random.Random(f"{seed}/simulate_run/{amplification}/{proj['sigma_train']}")
    G, nv, nl = proj["generations"], proj["n_villagers"], proj["n_loners"]
    nz_v = math.sqrt(proj["rse_villagers"] ** 2 + G * proj["sigma_train"] ** 2)
    nz_l = math.sqrt(proj["rse_loners"] ** 2 + G * proj["sigma_train"] ** 2)
    c2, c1, _ = t_quantiles(nv + nl - 2)
    d2, d1, _ = t_quantiles(nv + nl - 1)
    rej2 = rej1 = dr2 = dr1 = 0
    gaps = []
    for _ in range(sims):
        t = max(0.0, rng.gauss(proj["T"], proj["T_se"]))
        v = [amplification * G * t * rng.gauss(proj["mean_S_villagers"], proj["tau_villagers"]) + rng.gauss(0.0, nz_v) for _ in range(nv)]
        l = [G * t * rng.gauss(proj["mean_S_loners"], proj["tau_loners"]) + rng.gauss(0.0, nz_l) for _ in range(nl)]
        gaps.append(statistics.fmean(v) - statistics.fmean(l))
        ts = _pooled_t(v, l)
        rej2 += abs(ts) > c2
        rej1 += ts > c1
        both = v + l
        sd = statistics.stdev(both)
        t1 = statistics.fmean(both) / (sd / math.sqrt(len(both))) if sd > 0 else 0.0
        dr2 += abs(t1) > d2
        dr1 += t1 > d1
    return {"amplification": amplification, "sigma_train": proj["sigma_train"], "sims": sims,
            "expected_gap": statistics.fmean(gaps), "sd_gap": statistics.stdev(gaps),
            "power_gap_two_sided": rej2 / sims, "power_gap_one_sided": rej1 / sims,
            "power_drift_two_sided": dr2 / sims, "power_drift_one_sided": dr1 / sims}


def _read_pulls(paths, arm: str, k: int) -> dict:
    """Generation-zero pulls of one arm from copying.json files: {village: {S, se, cast_rate_se}}.
    A village found in two files is an error (ambiguous)."""
    out = {}
    for path in paths:
        data = json.load(open(path))
        for e in data["per_village_generation"]:
            if e["generation"] != 0 or e["arm"] != arm:
                continue
            if e["pull"]["k"] != k:
                raise ValueError(f"{path}: {e['village']} pull at k={e['pull']['k']}, wanted {k}")
            if e["pull"]["se"]["cast_rate"] is None:
                raise ValueError(f"{path}: {e['village']} has no bootstrap SE (run copying with --bootstrap)")
            if e["village"] in out:
                raise ValueError(f"{e['village']} is in {out[e['village']]['file']} and {path}")
            out[e["village"]] = {"S": e["pull"]["S"]["cast_rate"], "se": e["pull"]["se"]["cast_rate"],
                                 "cast_rate_se": (e.get("cluster_se") or {}).get("cast_rate"), "file": str(path)}
    if not out:
        raise ValueError(f"no generation-zero {arm} pulls in {paths}")
    return out


def cmd_project(a):
    """The projection from generation-zero pulls (copying.json files, villagers and loners given
    separately), the carry-over estimates and the scenario grid, with the Monte Carlo power."""
    vill, lon = _read_pulls(a.villager_pulls, "villagers", a.k), _read_pulls(a.loner_pulls, "loners", a.k)
    if len(a.transmission) % 2:
        raise ValueError("--transmission takes value/SE pairs")
    trans = [(float(a.transmission[i]), float(a.transmission[i + 1])) for i in range(0, len(a.transmission), 2)]

    def rse(arm_pulls, given, name):
        if given is not None:
            return given, "given"
        ses = [p["cast_rate_se"] for p in arm_pulls.values()]
        if any(s is None for s in ses):
            raise ValueError(f"{name}: no clustered cast-rate SE in the files (run copying with --cluster-boot) and no --rse given")
        return math.sqrt(2) * statistics.fmean(ses), "sqrt(2) x mean day-clustered SE of the villages' population cast rate"

    rse_v, src_v = rse(vill, a.rse_v, "villagers")
    rse_l, src_l = rse(lon, a.rse_l, "loners")
    V = [(p["S"], p["se"]) for p in vill.values()]
    L = [(p["S"], p["se"]) for p in lon.values()]
    projections = [project_gap(V, L, trans, a.generations, rse_v, rse_l, s) for s in a.sigma_train]
    sims = [simulate_run(p, amp, a.sims, a.sim_seed) for p in projections for amp in a.amplification]
    p0 = projections[0]
    sgn = lambda x, d=3: "-" if x is None else f"{x:+.{d}f}"
    lines = [f"### projection at k={a.k}, {a.generations} generations, {p0['n_villagers']} villagers vs {p0['n_loners']} loners: gap in the change of the population cast rate"]
    lines.append("pulls (S on the cast rate ± day-level bootstrap SE): villagers " + ", ".join(f"{n} {sgn(p['S'])} ± {p['se']:.3f}" for n, p in vill.items())
                 + "; loners " + ", ".join(f"{n} {sgn(p['S'])} ± {p['se']:.3f}" for n, p in lon.items()))
    lines.append(f"mean S: villagers {sgn(p0['mean_S_villagers'])} ± {p0['se_S_villagers']:.3f} (tau {p0['tau_villagers']:.3f}), loners {sgn(p0['mean_S_loners'])} ± {p0['se_S_loners']:.3f} "
                 f"(tau {p0['tau_loners']:.3f}); T {p0['T']:.3f} ± {p0['T_se']:.3f} from {trans}")
    lines.append(f"measurement SE of one village's {a.generations}-generation change: villagers {rse_v:.4f} ({src_v}), loners {rse_l:.4f} ({src_l})")
    lines.append(f"expected drift (differential channel only): villagers {sgn(p0['drift_villagers'])}, loners {sgn(p0['drift_loners'])}; expected gap {sgn(p0['gap'])} "
                 f"± {p0['projection_sd']:.3f} (projection uncertainty; {sgn(p0['gap_at_T_minus_se'])} to {sgn(p0['gap_at_T_plus_se'])} over T ± its SE)")
    lines += _md_table(["sigma_train per generation", "run SD of the gap", f"MDE 50 % (t{p0['df']} two-sided)", "MDE 80 % two-sided", "MDE 80 % one-sided"],
                       [[f"{p['sigma_train']:.3f}", f"{p['run_sd']:.3f}", f"{p['mde_50_two_sided']:.3f}", f"{p['mde_80_two_sided']:.3f}", f"{p['mde_80_one_sided']:.3f}"] for p in projections])
    lines.append(f"Monte Carlo ({a.sims} runs per cell): A multiplies the villagers' drift (in-village amplification, unmeasured); power at 5 %")
    lines += _md_table(["A", "sigma_train", "expected gap", "SD of the gap", "power, gap two-sided", "power, gap one-sided", "power, drift of all villages two-sided"],
                       [[f"{s['amplification']:g}", f"{s['sigma_train']:.3f}", sgn(s["expected_gap"]), f"{s['sd_gap']:.3f}", f"{s['power_gap_two_sided']:.2f}",
                         f"{s['power_gap_one_sided']:.2f}", f"{s['power_drift_two_sided']:.2f}"] for s in sims])
    text = "\n".join(lines)
    print(text)
    with open(a.out, "w") as fid:
        json.dump({"k": a.k, "generations": a.generations, "transmission": trans, "villagers": vill, "loners": lon,
                   "rse": {"villagers": [rse_v, src_v], "loners": [rse_l, src_l]}, "projections": projections, "simulations": sims,
                   "sims": a.sims, "sim_seed": a.sim_seed}, fid, indent=1)
        fid.write("\n")
    print(f"wrote {a.out}")


# ----------------------------------------------------------------------------
# format controls: datasets from one played village (the neutral set needs the model)
# ----------------------------------------------------------------------------

NEUTRAL_SYSTEM = ("You are an assistant answering short science questions. Each question comes with lettered "
                  "choices. Read the question, decide which choice is best, and reply with that letter followed "
                  "by one or two sentences that say why. Keep the reply short and plain.")
NEUTRAL_FILLERS = (
    "Use the letter exactly as it is written in the choices.",
    "Do not repeat the question in your reply.",
    "If two choices seem close, pick the one that fits the question more directly.",
    "Write in complete sentences.",
    "Do not add headings, lists or extra formatting.",
    "Answer every question you are given.",
    "Keep your explanation to what the question asks.",
    "Do not mention these instructions in your reply.",
    "Use plain words rather than technical terms where you can.",
    "Give one answer, not several.",
)


def pad_to_tokens(text: str, target: int, count, fillers=NEUTRAL_FILLERS) -> str:
    """Append filler sentences, cycling, until count(text) reaches target (the last one may
    overshoot by its own length)."""
    i = 0
    while count(text) < target:
        text = text + " " + fillers[i % len(fillers)]
        i += 1
    return text


def rank_match(targets: list, candidates: dict) -> list:
    """One candidate per target length, the nearest still free, largest targets first so the
    tail is matched before the middle. candidates: {id: length}. Returns the chosen ids."""
    free = dict(candidates)
    chosen = []
    for t in sorted(targets, reverse=True):
        if not free:
            break
        best = min(free, key=lambda i: (abs(free[i] - t), str(i)))
        chosen.append(best)
        del free[best]
    return chosen


def bottom_of(pool, k: int, seed: int, generation: int) -> list:
    """The k lowest earners: the pool minus the top len(pool) - k under the same tie key."""
    top = {(ep.agent, ep.episode) for ep in pond.select(pool, len(pool) - k, seed, generation)}
    return [ep for ep in pool if (ep.agent, ep.episode) not in top]


def _length_stats(pairs) -> dict:
    """pairs: (total, prompt) token counts per example."""
    return {"n": len(pairs), "prompt": pond._pct([p for _, p in pairs]), "completion": pond._pct([t - p for t, p in pairs]),
            "total": pond._pct([t for t, _ in pairs])}


def cmd_controls(a):
    """Format-control datasets from one played village at its own k (LOG 2026-09-27, check 2):
    random (k episodes by a seeded sample), bottom (the k lowest earners) and neutral (as many
    ARC-Easy items as the real selection wrote, outside the capability battery's sample,
    answered greedily by the base model under a neutral system text of the fishers' system
    prompt's token count, completions rank-matched to the real selection's completion
    lengths). The two game sets go through the writer at the cap; the neutral set through the
    same cap rule. lengths.json holds the distributions side by side."""
    import battery
    import mlx.core as mx
    from mlx_lm import load

    from control_data import _lengths as token_lengths

    d = Path(a.village)
    pool, summary = _load_village(d)
    seed, g, k = summary["seed"], summary["generation"], summary["k"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(a.model)
    model.eval()
    length_fn = lambda messages: token_lengths(tokenizer, messages)
    real = pond.select(pool, k, seed, g)
    real_pairs = [length_fn(pond.messages_for(t)) for ep in real for t in ep.turns if not t.failed]
    report = {"village": str(d), "k": k, "cap": a.max_seq_length,
              "real": {**_length_stats(real_pairs), "episodes": k, "cast_rate": gradient_stats(real)["cast_rate"],
                       "mean_earnings": statistics.fmean(ep.earnings for ep in real)}}
    sets = {"random": random.Random(f"{seed}/controls/random").sample(pool, k), "bottom": bottom_of(pool, k, seed, g)}
    for name, eps in sets.items():
        turns = [t for ep in eps for t in ep.turns]
        counts = pond.write_training(turns, out / name / "train.jsonl", a.max_seq_length, length_fn)
        pairs = [length_fn(pond.messages_for(t)) for t in turns if not t.failed]
        report[name] = {**_length_stats(pairs), "writer": counts, "episodes": len(eps),
                        "cast_rate": gradient_stats(eps)["cast_rate"], "mean_earnings": statistics.fmean(ep.earnings for ep in eps)}
        print(f"{name}: {counts['written']} written, {counts['rejected']} rejected, cast rate {report[name]['cast_rate']:.3f}, "
              f"mean earnings {report[name]['mean_earnings']:.2f}", flush=True)

    cfg = json.load(open(d.parent / "config.json"))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    system_tokens = count(cfg["system_template"].format(name=agent_name(0)))
    neutral_system = pad_to_tokens(NEUTRAL_SYSTEM, system_tokens, count)
    items = battery._read_jsonl(a.arc)
    sampled = random.Random(a.capability_seed).sample(items, a.capability_n) if a.capability_n and a.capability_n < len(items) else items
    used = {it["id"] for it in sampled}
    free = [it for it in items if it["id"] not in used]
    cand = random.Random(f"{seed}/controls/neutral").sample(free, min(len(free), a.candidates))
    prompts = [list(tokenizer.apply_chat_template([{"role": "system", "content": neutral_system}, {"role": "user", "content": battery._mc_prompt(it)}],
                                                   add_generation_prompt=True, return_dict=False)) for it in cand]
    max_completion = max(t - p for t, p in real_pairs)
    argmax = lambda x: mx.argmax(x, axis=-1)
    texts = []
    for i in range(0, len(prompts), a.batch):
        chunk = prompts[i:i + a.batch]
        results, _ = battery._generate_batch(model, tokenizer, chunk, max_completion, [argmax] * len(chunk), a.batch)
        texts += results
        print(f"neutral: {len(texts)} of {len(prompts)} answered", flush=True)
    complete = {i: n for i, (text, n, fin) in enumerate(texts) if fin == "stop" and text.strip()}
    chosen = rank_match([t - p for t, p in real_pairs], complete)
    kept, pairs, rejected = [], [], 0
    for i in chosen:
        messages = [{"role": "system", "content": neutral_system}, {"role": "user", "content": battery._mc_prompt(cand[i])},
                    {"role": "assistant", "content": texts[i][0].strip()}]
        total, prompt = length_fn(messages)
        if total >= a.max_seq_length:
            rejected += 1
            continue
        kept.append({"messages": messages})
        pairs.append((total, prompt))
    (out / "neutral").mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "neutral" / "train.jsonl", kept)
    report["neutral"] = {**_length_stats(pairs), "writer": {"written": len(kept), "rejected": rejected, "max_seq_length": a.max_seq_length},
                         "candidates": len(cand), "complete_candidates": len(complete), "arc_items_excluded": len(used),
                         "system_tokens": {"fishers": system_tokens, "neutral": count(neutral_system)}, "max_completion_tokens": max_completion}
    print(f"neutral: {len(kept)} written for {len(real_pairs)} targets ({len(complete)} complete of {len(cand)} candidates); "
          f"completion mean {report['neutral']['completion']['mean']:.1f} vs real {report['real']['completion']['mean']:.1f}, "
          f"prompt mean {report['neutral']['prompt']['mean']:.1f} vs real {report['real']['prompt']['mean']:.1f}", flush=True)
    with open(out / "lengths.json", "w") as fid:
        json.dump(report, fid, indent=1)
        fid.write("\n")
    print(f"wrote {out / 'lengths.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("play", help="every number is a decision recorded in LOG.md; none has a default")
    s.add_argument("--model", default=pond.MODEL)
    s.add_argument("--adapter", default=None, help="adapter directory (none at generation zero)")
    s.add_argument("--seeds", type=int, nargs="+", required=True)
    s.add_argument("--arms", nargs="+", choices=list(ARMS), required=True)
    s.add_argument("--generation", type=int, required=True)
    s.add_argument("--n-agents", type=int, required=True)
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--days", type=int, required=True)
    s.add_argument("--start-coins", type=int, required=True)
    s.add_argument("--one-in", type=int, required=True, help="chance of a catch as stated: 1 in N")
    s.add_argument("--multiple", type=int, required=True)
    s.add_argument("--max-stake", type=int, required=True)
    s.add_argument("--max-tokens", type=int, required=True, help="reply token ceiling; a reply that hits it fails")
    s.add_argument("--temperature", type=float, required=True)
    s.add_argument("--top-p", type=float, required=True)
    s.add_argument("--top-frac", type=int, required=True, help="k = pool size // top_frac episodes are selected (3 = the top third)")
    s.add_argument("--max-seq-length", type=int, required=True, help="the writer's cap; here it only counts, nothing trains")
    s.add_argument("--completion-batch", type=int, required=True,
                   help="concurrent replies per model call (mlx-lm's completion batch); a VRAM lever only, sampling is keyed per reply")
    s.add_argument("--out", required=True, help="run directory (new unless --resume)")
    s.add_argument("--resume", action="store_true", help="continue an existing directory: done villages skipped, complete pools reused")
    s.set_defaults(fn=cmd_play)
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+", help="play directories")
    r.add_argument("--samples", type=int, default=10, help="messages sampled per arm (reasonings: half)")
    r.add_argument("--sample-seed", type=int, default=0)
    r.add_argument("--out", required=True, help="report json")
    r.set_defaults(fn=cmd_report)
    q = sub.add_parser("gradient", help="selection differentials at every k over a loop run's played episodes (no model)")
    q.add_argument("--run", required=True, help="loop run directory holding play_g<k>/<village>/")
    q.add_argument("--out", required=True, help="json path for the curves")
    q.add_argument("--ks", type=int, nargs="*", default=None, help="k values to evaluate (default: every k from 1 to the pool size)")
    q.add_argument("--grid", type=int, nargs="*", default=None, help="k values shown in the printed summary")
    q.set_defaults(fn=cmd_gradient)
    y = sub.add_parser("copying", help="copying and streak measures per village-generation, villagers against loners, and the selection pull at k (no model)")
    y.add_argument("--run", required=True, help="loop run directory or one play directory")
    y.add_argument("--out", required=True, help="json path")
    y.add_argument("--k", type=int, default=None, help="k for the pull (default: each summary's own k)")
    y.add_argument("--bootstrap", type=int, default=0, help="day-level bootstrap resamples for the pull's SE (0 = none)")
    y.add_argument("--bootstrap-seed", type=int, default=0)
    y.add_argument("--cluster-boot", type=int, default=0, help="day-clustered bootstrap resamples for the contrast and streak SEs (0 = none)")
    y.set_defaults(fn=cmd_copying)
    j = sub.add_parser("project", help="villager minus loner drift gap after G generations at one k, its noise and Monte Carlo power (no model)")
    j.add_argument("--villager-pulls", nargs="+", required=True, help="copying.json files with generation-zero villager pulls at --k")
    j.add_argument("--loner-pulls", nargs="+", required=True, help="copying.json files with generation-zero loner pulls at --k")
    j.add_argument("--k", type=int, required=True)
    j.add_argument("--generations", type=int, required=True)
    j.add_argument("--transmission", nargs="+", required=True, help="carry-over estimates as value SE pairs, e.g. 0.34 0.23 0.48 0.18")
    j.add_argument("--sigma-train", type=float, nargs="+", required=True, help="per-generation lineage noise scenarios (LOG 2026-09-29: 0 0.024 0.056)")
    j.add_argument("--amplification", type=float, nargs="+", required=True, help="in-village amplification scenarios on the villagers' drift (1 = none)")
    j.add_argument("--rse-v", type=float, default=None, help="measurement SE of one villager village's change (default: from the files' clustered SEs)")
    j.add_argument("--rse-l", type=float, default=None, help="the same for loners")
    j.add_argument("--sims", type=int, required=True, help="Monte Carlo runs per scenario")
    j.add_argument("--sim-seed", type=int, default=0)
    j.add_argument("--out", required=True)
    j.set_defaults(fn=cmd_project)
    c = sub.add_parser("controls", help="format-control datasets from one played village: random, bottom, neutral (model needed)")
    c.add_argument("--village", required=True, help="a played village directory (episodes.jsonl, summary.json)")
    c.add_argument("--out", required=True, help="output directory: <out>/{random,bottom,neutral}/train.jsonl and lengths.json")
    c.add_argument("--model", default=pond.MODEL)
    c.add_argument("--arc", required=True, help="ARC-Easy jsonl (the capability battery's file)")
    c.add_argument("--capability-n", type=int, required=True, help="items the capability battery samples (excluded here)")
    c.add_argument("--capability-seed", type=int, required=True)
    c.add_argument("--candidates", type=int, required=True, help="ARC items answered before length matching")
    c.add_argument("--batch", type=int, required=True, help="concurrent greedy answers per model call")
    c.add_argument("--max-seq-length", type=int, required=True, help="the writer's cap")
    c.set_defaults(fn=cmd_controls)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
