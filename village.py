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
import random
import re
import shutil
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
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
