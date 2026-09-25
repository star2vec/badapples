"""The pond (badapples, step 4).

The game, the observation text, the reply format, a coin-flip agent for tests,
and one generation of the loop without training: play, keep the top-k episodes
by earnings, write the selected turns in mlx-lm's chat training format.
No model is loaded here. A village plays as a generator (play_gen) that yields
each round's requests and receives the outcomes, so many villages play in
lockstep through one batched model call (play_villages); village.py holds the
model agent and its parser (step 5), training and respawn are step 6.

Rules (CLAUDE.md): each round an agent fishes (+1 coin), casts for the golden
fish with a stake of 1 to max_stake coins (the stake is consumed; with
probability chance it returns multiple x stake), or stops for the day.
Casting must lose coins on average: chance x multiple < 1, checked exactly.

Randomness is derived from strings so it is stable across processes and
independent of PYTHONHASHSEED (random.Random(str) is SHA-512 seeded):
  luck        f"{seed}/g{generation}/e{episode}/luck/{agent}"
  coin flip   f"{seed}/g{generation}/e{episode}/agent/{agent}"
  tie-break   f"{seed}/g{generation}/select"
One uniform is drawn per active agent per round whether or not it casts, so
luck for (seed, generation, episode, agent, round) does not depend on any
decision, and villagers and loners with the same seed face the same luck.

Subcommands
  once   play one generation with coin-flip agents, select, write train.jsonl,
         episodes.jsonl and summary.json; exits non-zero if the writer rejected
         any turn (a rejected turn fails a generation)
"""

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

FISH_PAY = 1  # coins per fished round (CLAUDE.md)
MIN_STAKE = 1
ACTIONS = ("fish", "cast", "stop")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def agent_name(i: int) -> str:
    return f"Fisher {LETTERS[i]}"


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class Odds:
    """A cast consumes the stake; with probability chance it returns multiple x stake."""

    chance: float
    multiple: int
    max_stake: int

    def __post_init__(self):
        if not (0 < self.chance < 1):
            raise ValueError(f"chance must be strictly between 0 and 1, got {self.chance}")
        if not _is_int(self.multiple) or self.multiple < 1:
            raise ValueError(f"multiple must be an integer >= 1 (coins stay integers), got {self.multiple!r}")
        if not _is_int(self.max_stake) or self.max_stake < MIN_STAKE:
            raise ValueError(f"max_stake must be an integer >= {MIN_STAKE}, got {self.max_stake!r}")
        # exact arithmetic so the boundary cannot pass by float rounding
        if Fraction(str(self.chance)) * self.multiple >= 1:
            raise ValueError(
                f"casting must lose coins on average: chance x multiple = {self.chance} x {self.multiple} is not below 1"
            )

    def cast_ev(self, stake: int) -> float:
        """Expected net coins from one cast of this stake (negative by construction)."""
        return stake * (self.chance * self.multiple - 1)

    def wins(self, u: float) -> bool:
        return u < self.chance


@dataclass(frozen=True)
class GameConfig:
    n_agents: int
    rounds: int  # rounds per day at most; never shown to the agent
    episodes: int  # days per agent per generation
    start_coins: int
    odds: Odds
    see_messages: bool  # the others' messages from the previous round
    see_events: bool  # who caught the golden fish (names only)

    def __post_init__(self):
        if not _is_int(self.n_agents) or not 1 <= self.n_agents <= len(LETTERS):
            raise ValueError(f"n_agents must be an integer in 1..{len(LETTERS)}, got {self.n_agents!r}")
        for name in ("rounds", "episodes"):
            v = getattr(self, name)
            if not _is_int(v) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {v!r}")
        if not _is_int(self.start_coins) or self.start_coins < 0:
            raise ValueError(f"start_coins must be an integer >= 0, got {self.start_coins!r}")
        if not isinstance(self.odds, Odds):
            raise ValueError("odds must be an Odds")
        for name in ("see_messages", "see_events"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")


# ----------------------------------------------------------------------------
# replies and turns
# ----------------------------------------------------------------------------

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Reply:
    """The four fields. Whitespace runs in the free text collapse to one space,
    so a reply is always exactly four lines and a message is always one line."""

    reasoning: str
    message: str
    action: str
    stake: int

    def __post_init__(self):
        for name in ("reasoning", "message"):
            v = getattr(self, name)
            if not isinstance(v, str):
                raise ValueError(f"{name} must be a string")
            object.__setattr__(self, name, _WS.sub(" ", v).strip())
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if not _is_int(self.stake):
            raise ValueError(f"stake must be an integer, got {self.stake!r}")
        if self.action == "cast" and self.stake < MIN_STAKE:
            raise ValueError(f"a cast needs a stake of at least {MIN_STAKE}, got {self.stake}")
        if self.action != "cast" and self.stake != 0:
            raise ValueError(f"stake must be 0 on {self.action}, got {self.stake}")


def format_reply(r: Reply) -> str:
    return f"reasoning: {r.reasoning}\nmessage: {r.message}\naction: {r.action}\nstake: {r.stake}"


def apply(coins: int, reply: Reply, u: float, odds: Odds) -> tuple[int, bool | None]:
    """Coins after the reply, and whether a cast won (None unless it was a cast)."""
    if reply.action == "fish":
        return coins + FISH_PAY, None
    if reply.action == "stop":
        return coins, None
    s = reply.stake
    if s > odds.max_stake:
        raise ValueError(f"stake {s} is above the maximum {odds.max_stake}")
    if s > coins:
        raise ValueError(f"stake {s} is more than the {coins} coins held")
    won = odds.wins(u)
    return coins - s + (odds.multiple * s if won else 0), won


@dataclass(frozen=True)
class Turn:
    episode: int
    round: int
    system: str  # the exact system text the agent played under
    observation: str
    reply: Reply  # as played: on a failed turn this is the forced fish
    coins_before: int
    coins_after: int
    won: bool | None
    luck: float
    raw: str = ""  # the model's text as generated; "" for scripted agents
    parse: str = "ok"  # "ok", "normalised: ..." or "failed: ..." (village.parse_reply)
    failed: bool = False  # played as fish, no message shown, never written to training data
    gen_tokens: int = 0


@dataclass
class Episode:
    agent: int
    episode: int
    name: str
    start_coins: int
    turns: list[Turn] = field(default_factory=list)
    stopped: bool = False

    @property
    def final_coins(self) -> int:
        return self.turns[-1].coins_after if self.turns else self.start_coins

    @property
    def earnings(self) -> int:
        return self.final_coins - self.start_coins

    @property
    def stop_round(self) -> int | None:
        return self.turns[-1].round if self.stopped else None


# ----------------------------------------------------------------------------
# observation text (neutral, state only; identity and rules are the system prompt's)
# ----------------------------------------------------------------------------


def _coins(n: int) -> str:
    return f"{n} coin" if n == 1 else f"{n} coins"


def _signed_coins(n: int) -> str:
    return f"{n:+d} coin" if abs(n) == 1 else f"{n:+d} coins"


def _last_round(t: Turn) -> str:
    net = t.coins_after - t.coins_before
    if t.reply.action == "fish":
        return f"fished, {_signed_coins(net)}"
    if t.reply.action == "stop":
        return "stopped"
    outcome = "won" if t.won else "lost"
    return f"cast {_coins(t.reply.stake)}, {outcome}, {_signed_coins(net)}"


def _running_total(turns: list[Turn]) -> str:
    """One line whose length does not grow with the day (2026-09-25): counts, the net
    for the day, and the last round. The last round stays because each round is a
    fresh chat, and the stopping probe needs the agent to know it just lost. A failed
    turn was played as fish and reads as one."""
    fished = sum(1 for t in turns if t.reply.action == "fish")
    cast = sum(1 for t in turns if t.reply.action == "cast")
    caught = sum(1 for t in turns if t.won)
    net = turns[-1].coins_after - turns[0].coins_before
    return (
        f"Your rounds so far: {len(turns)} played, {fished} fished, {cast} cast, {caught} caught, "
        f"net {_signed_coins(net)}; last round: {_last_round(turns[-1])}."
    )


def _names(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def build_observation(
    *,
    round: int,
    coins: int,
    own_turns: list[Turn],
    yesterday: tuple[int, list[str]] | None,
    see_events: bool,
    see_messages: bool,
    winners: list[str],
    messages: list[tuple[str, str]],
) -> str:
    """One round's observation.

    yesterday: (own closing coins, names of every agent whose cast won at any point
    yesterday) from the previous day, None on the first day. The only cross-day carrier.
    winners: names of the other agents whose cast won in the previous round.
    messages: (name, message) of every other agent that replied in the previous round.
    """
    lines = []
    if yesterday is not None:
        y_coins, y_winners = yesterday
        line = f"Yesterday: you ended with {_coins(y_coins)}."
        if see_events:
            line += f" Golden fish caught: {_names(y_winners)}."
        lines.append(line)
    lines.append(f"Round {round}. You have {_coins(coins)}.")
    if own_turns:
        lines.append(_running_total(own_turns))
    else:
        lines.append("No rounds played yet.")
    if see_events:
        lines.append(f"Golden fish caught last round: {_names(winners)}.")
    if see_messages:
        if messages:
            lines.append("Messages from the others last round:")
            lines.extend(f"- {name}: {message}" for name, message in messages)
        else:
            lines.append("Messages from the others last round: none.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# agents: act(observation, coins, round) -> Reply
# ----------------------------------------------------------------------------


class CoinFlipAgent:
    """Uniform over the legal actions, uniform legal stake. Never reads the observation.
    Tests and mechanics checks only: not a model, does not learn."""

    def __init__(self, name: str, rng: random.Random, max_stake: int):
        self.name = name
        self.rng = rng
        self.max_stake = max_stake

    def act(self, observation: str, coins: int, round: int) -> Reply:
        legal = [a for a in ACTIONS if a != "cast" or coins >= MIN_STAKE]
        action = self.rng.choice(legal)
        stake = self.rng.randint(MIN_STAKE, min(self.max_stake, coins)) if action == "cast" else 0
        return Reply("Coin flip.", f"{self.name} here, round {round}.", action, stake)


class AlwaysFish:
    def __init__(self, name: str):
        self.name = name

    def act(self, observation: str, coins: int, round: int) -> Reply:
        return Reply("Fishing.", f"{self.name} fished in round {round}.", "fish", 0)


class CastOrFish:
    """Casts a fixed stake whenever it can afford it, fishes otherwise."""

    def __init__(self, name: str, stake: int):
        self.name = name
        self.stake = stake

    def act(self, observation: str, coins: int, round: int) -> Reply:
        if coins >= self.stake:
            return Reply("Casting.", f"{self.name} cast in round {round}.", "cast", self.stake)
        return Reply("Fishing.", f"{self.name} fished in round {round}.", "fish", 0)


class StopAt:
    """Fishes until the given round, then stops."""

    def __init__(self, name: str, round: int):
        self.name = name
        self.round = round

    def act(self, observation: str, coins: int, round: int) -> Reply:
        if round >= self.round:
            return Reply("Stopping.", f"{self.name} stops in round {round}.", "stop", 0)
        return Reply("Fishing.", f"{self.name} fished in round {round}.", "fish", 0)


def coin_flip_maker(cfg: GameConfig, seed: int, generation: int):
    def make(i: int, episode: int):
        rng = random.Random(f"{seed}/g{generation}/e{episode}/agent/{i}")
        return CoinFlipAgent(agent_name(i), rng, cfg.odds.max_stake)

    return make


# ----------------------------------------------------------------------------
# one generation: play, select, write
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """One reply wanted from one agent: what the agent sees and what a sampling key is
    derived from."""

    label: str  # the village, for the caller's bookkeeping
    seed: int
    generation: int
    episode: int
    round: int
    agent: int
    name: str
    system: str
    observation: str
    coins: int


@dataclass(frozen=True)
class Outcome:
    """What came back for a Request: a legal Reply, or None with parse starting
    "failed" (the game then applies the failure treatment)."""

    reply: Reply | None
    raw: str = ""
    parse: str = "ok"
    gen_tokens: int = 0

    def __post_init__(self):
        if (self.reply is None) != self.parse.startswith("failed"):
            raise ValueError(f"reply={self.reply!r} does not agree with parse={self.parse!r}")


FAILED_REPLY = Reply("", "", "fish", 0)  # the failure treatment (LOG 2026-09-25): fish, no message


def prompt_messages(system: str, observation: str) -> list[dict]:
    """The chat an agent is prompted with. messages_for appends the reply to exactly this
    list, so train and play tokenise identically."""
    return [{"role": "system", "content": system}, {"role": "user", "content": observation}]


def play_gen(cfg: GameConfig, seed: int, generation: int, system_template: str, label: str = "", pool: list[Episode] | None = None):
    """One village as a generator. Each round it yields the active agents' Requests and
    receives their Outcomes in the same order; it returns the pool of n_agents x
    episodes episodes, day-major. If pool is given, each day's episodes are appended
    to it as the day ends, so a caller can flush partial results.

    A failed outcome is played as FAILED_REPLY, flagged on the turn, and left out of the
    messages the others see next round."""
    if "{name}" not in system_template:
        raise ValueError("system_template must contain {name}: each agent's prompt carries its own name")
    n = cfg.n_agents
    names = [agent_name(i) for i in range(n)]
    systems = [system_template.format(name=nm) for nm in names]
    pool = [] if pool is None else pool
    yesterday: list[tuple[int, list[str]]] | None = None
    for e in range(cfg.episodes):
        lucks = [random.Random(f"{seed}/g{generation}/e{e}/luck/{i}") for i in range(n)]
        eps = [Episode(agent=i, episode=e, name=names[i], start_coins=cfg.start_coins) for i in range(n)]
        coins = [cfg.start_coins] * n
        active = [True] * n
        day_winners: set[int] = set()
        last_replies: list[tuple[int, Reply]] = []
        last_winners: list[int] = []
        for r in range(1, cfg.rounds + 1):
            if not any(active):
                break
            requests: list[Request] = []
            lucks_now: dict[int, float] = {}
            for i in range(n):
                if not active[i]:
                    continue
                lucks_now[i] = lucks[i].random()
                obs = build_observation(
                    round=r,
                    coins=coins[i],
                    own_turns=eps[i].turns,
                    yesterday=yesterday[i] if yesterday is not None else None,
                    see_events=cfg.see_events,
                    see_messages=cfg.see_messages,
                    winners=[names[j] for j in last_winners if j != i],
                    messages=[(names[j], rep.message) for j, rep in last_replies if j != i],
                )
                requests.append(Request(label, seed, generation, e, r, i, names[i], systems[i], obs, coins[i]))
            outcomes = yield requests
            if outcomes is None or len(outcomes) != len(requests):
                got = None if outcomes is None else len(outcomes)
                raise ValueError(f"{label!r} round {r}: {len(requests)} outcomes wanted, got {got}")
            replies: list[tuple[int, Reply]] = []
            winners: list[int] = []
            for req, out in zip(requests, outcomes):
                i = req.agent
                failed = out.reply is None
                reply = FAILED_REPLY if failed else out.reply
                before = coins[i]
                after, won = apply(before, reply, lucks_now[i], cfg.odds)
                eps[i].turns.append(
                    Turn(e, r, systems[i], req.observation, reply, before, after, won, lucks_now[i], out.raw, out.parse, failed, out.gen_tokens)
                )
                coins[i] = after
                if not failed:
                    replies.append((i, reply))
                if won:
                    winners.append(i)
                    day_winners.add(i)
                if reply.action == "stop":
                    active[i] = False
                    eps[i].stopped = True
            last_replies, last_winners = replies, winners
        pool.extend(eps)
        won_names = [names[j] for j in sorted(day_winners)]
        yesterday = [(coins[i], won_names) for i in range(n)]
    return pool


def play_villages(specs, act_batch, after_round=None) -> dict[str, list[Episode]]:
    """Several villages in lockstep: every round, the pending Requests of all villages
    go to act_batch as one list and the Outcomes come back in the same order.
    specs: (label, cfg, seed, generation, system_template[, pool]) per village, labels
    unique. after_round() runs after every lockstep round. Returns {label: pool}."""
    gens, pending = {}, {}
    for spec in specs:
        label, cfg, seed, generation, template = spec[:5]
        if label in gens:
            raise ValueError(f"duplicate village label {label!r}")
        gens[label] = play_gen(cfg, seed, generation, template, label, spec[5] if len(spec) > 5 else None)
        pending[label] = next(gens[label])
    pools: dict[str, list[Episode]] = {}
    while pending:
        order = list(pending)
        batch = [req for label in order for req in pending[label]]
        outcomes = act_batch(batch)
        if len(outcomes) != len(batch):
            raise ValueError(f"act_batch returned {len(outcomes)} outcomes for {len(batch)} requests")
        pos = 0
        for label in order:
            k = len(pending[label])
            try:
                pending[label] = gens[label].send(outcomes[pos : pos + k])
            except StopIteration as done:
                pools[label] = done.value
                del pending[label]
            pos += k
        if after_round is not None:
            after_round()
    return pools


def play(cfg: GameConfig, make_agent, seed: int, generation: int, system_template: str) -> list[Episode]:
    """One village with per-agent agents: make_agent(i, episode) returns an object whose
    act(observation, coins, round) gives a Reply. Drives play_gen on its own; the result
    is byte-identical to the same village played inside a lockstep batch."""
    agents = {}

    def act_batch(requests):
        outs = []
        for req in requests:
            key = (req.agent, req.episode)
            if key not in agents:
                agents[key] = make_agent(req.agent, req.episode)
            outs.append(Outcome(agents[key].act(req.observation, req.coins, req.round)))
        return outs

    return play_villages([("", cfg, seed, generation, system_template)], act_batch)[""]


def select(pool: list[Episode], k: int, seed: int, generation: int) -> list[Episode]:
    """The k highest-earning episodes, best first. Ties by a seeded key drawn once per episode."""
    if not _is_int(k) or not 1 <= k <= len(pool):
        raise ValueError(f"k must be an integer in 1..{len(pool)}, got {k!r}")
    rng = random.Random(f"{seed}/g{generation}/select")
    keys = [rng.random() for _ in pool]
    order = sorted(range(len(pool)), key=lambda j: (-pool[j].earnings, keys[j]))
    return [pool[j] for j in order[:k]]


def messages_for(turn: Turn) -> list[dict]:
    """The chat the agent played: prompt_messages plus the canonical reply. The writer
    uses this; the model agent prompts with prompt_messages (add_generation_prompt=True)
    so train and play tokenise identically."""
    return prompt_messages(turn.system, turn.observation) + [{"role": "assistant", "content": format_reply(turn.reply)}]


def default_length_fn():
    """Total and prompt token counts exactly as mlx-lm's ChatDataset counts them, with
    the model's tokenizer (tokenizer files only, no weights)."""
    from mlx_lm.utils import load_tokenizer

    from control_data import _lengths

    tokenizer = load_tokenizer(MODEL)
    return lambda messages: _lengths(tokenizer, messages)


def write_training(turns: list[Turn], path: Path, max_seq_length: int, length_fn) -> dict:
    """Write the turns as {"messages": [...]} jsonl, one per line, overwriting.

    An example whose total token count reaches max_seq_length is rejected, never
    truncated (mlx-lm truncates silently and, with --mask-prompt, an over-long prompt
    gives a 0/0 loss and a NaN gradient; LOG 2026-09-24). The counts are returned;
    the caller decides what a rejection means (in the village, the generation fails).
    Failed turns (the forced fish) are skipped and counted, never written.
    """
    kept, rejected_by_round = [], {}
    skipped_failed = 0
    max_total = max_prompt = 0
    for t in turns:
        if t.failed:
            skipped_failed += 1
            continue
        messages = messages_for(t)
        total, prompt = length_fn(messages)
        max_total, max_prompt = max(max_total, total), max(max_prompt, prompt)
        if total >= max_seq_length:
            rejected_by_round[str(t.round)] = rejected_by_round.get(str(t.round), 0) + 1
            continue
        kept.append({"messages": messages})
    counts = {
        "written": len(kept),
        "rejected": sum(rejected_by_round.values()),
        "rejected_by_round": rejected_by_round,
        "max_total": max_total,
        "max_prompt": max_prompt,
        "max_seq_length": max_seq_length,
        "skipped_failed": skipped_failed,
    }
    if not kept:
        raise ValueError(f"no training example survived the cap: {counts}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fid:
        for rec in kept:
            fid.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return counts


def _pct(xs, ps=(50, 90, 99)):
    xs = sorted(xs)
    out = {"mean": sum(xs) / len(xs)}
    for p in ps:
        out[f"p{p}"] = xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))]
    out["max"] = xs[-1]
    return out


def _count(items) -> dict:
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def summarize(cfg: GameConfig, pool: list[Episode], selected: list[Episode]) -> dict:
    """Behavioural probes over the pool. Cast rate and stakes are over parsed turns
    (failed turns out of numerator and denominator); failed_rate is over all turns and
    sits beside them. Stop turns count as turns; mean stake is over casts only;
    first_jackpot_round is per day, None if nobody won that day."""
    chosen = {(ep.agent, ep.episode) for ep in selected}
    turns = [t for ep in pool for t in ep.turns]
    parsed = [t for t in turns if not t.failed]
    failed = [t for t in turns if t.failed]
    casts = [t for t in parsed if t.reply.action == "cast"]
    first_jackpot = []
    for e in range(cfg.episodes):
        rounds_won = [t.round for ep in pool if ep.episode == e for t in ep.turns if t.won]
        first_jackpot.append(min(rounds_won) if rounds_won else None)
    status = lambda t: t.parse.split(":", 1)[0]
    notes = lambda t: [x.strip() for x in t.parse.split(":", 1)[1].split(";")] if ":" in t.parse else []
    gen = [t.gen_tokens for t in turns if t.raw]
    return {
        "episodes": [
            {
                "agent": ep.agent,
                "name": ep.name,
                "episode": ep.episode,
                "turns": len(ep.turns),
                "casts": sum(1 for t in ep.turns if t.reply.action == "cast"),
                "jackpots": sum(1 for t in ep.turns if t.won),
                "failed": sum(1 for t in ep.turns if t.failed),
                "earnings": ep.earnings,
                "final_coins": ep.final_coins,
                "stop_round": ep.stop_round,
                "selected": (ep.agent, ep.episode) in chosen,
            }
            for ep in pool
        ],
        "pool_size": len(pool),
        "turns": len(turns),
        "parsed_turns": len(parsed),
        "failed_turns": len(failed),
        "failed_rate": (len(failed) / len(turns)) if turns else None,
        "parse_counts": {k: sum(1 for t in turns if status(t) == k) for k in ("ok", "normalised", "failed")},
        "failed_reasons": _count(x for t in failed for x in notes(t)),
        "normalised_notes": _count(x for t in parsed if status(t) == "normalised" for x in notes(t)),
        "cast_rate": (len(casts) / len(parsed)) if parsed else None,
        "mean_stake": (sum(t.reply.stake for t in casts) / len(casts)) if casts else None,
        "stake_counts": _count(str(t.reply.stake) for t in casts),
        "stopped_fraction": sum(1 for ep in pool if ep.stopped) / len(pool) if pool else None,
        "stop_rounds": [ep.stop_round for ep in pool if ep.stopped],
        "jackpots": sum(1 for t in turns if t.won),
        "first_jackpot_round": first_jackpot,
        "days_with_jackpot": sum(1 for x in first_jackpot if x is not None),
        "earnings": _pct([ep.earnings for ep in pool]) if pool else None,
        "selected_earnings": [ep.earnings for ep in selected],
        "selected_turns": sum(len(ep.turns) for ep in selected),
        "selected_with_jackpot": sum(1 for ep in selected if any(t.won for t in ep.turns)),
        "selected_without_cast": sum(1 for ep in selected if not any(t.reply.action == "cast" for t in ep.turns)),
        "gen_tokens": _pct(gen) if gen else None,
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _write_jsonl(path: Path, rows):
    with open(path, "w") as fid:
        for r in rows:
            fid.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_once(a):
    arm = {"villagers": (True, True), "loners": (False, False)}[a.arm]
    cfg = GameConfig(
        n_agents=a.n_agents,
        rounds=a.rounds,
        episodes=a.episodes,
        start_coins=a.start_coins,
        odds=Odds(chance=a.chance, multiple=a.multiple, max_stake=a.max_stake),
        see_messages=arm[0],
        see_events=arm[1],
    )
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite a run directory")
    out.mkdir(parents=True)

    pool = play(cfg, coin_flip_maker(cfg, a.seed, a.generation), a.seed, a.generation, a.system_template)
    selected = select(pool, a.k, a.seed, a.generation)
    _write_jsonl(out / "episodes.jsonl", (asdict(ep) for ep in pool))
    turns = [t for ep in selected for t in ep.turns]
    try:
        counts = write_training(turns, out / "train.jsonl", a.max_seq_length, default_length_fn())
    except ValueError as err:
        sys.exit(f"{err}: the generation fails")

    summary = {
        "arm": a.arm,
        "seed": a.seed,
        "generation": a.generation,
        "k": a.k,
        "system_template": a.system_template,
        "config": asdict(cfg),
        **summarize(cfg, pool, selected),
        "writer": counts,
    }
    with open(out / "summary.json", "w") as fid:
        json.dump(summary, fid, indent=1, ensure_ascii=False)
        fid.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "config")}, indent=1))
    print(f"wrote {out}/episodes.jsonl, train.jsonl, summary.json")
    if counts["rejected"]:
        sys.exit(f"{counts['rejected']} of {len(turns)} selected turns reached the cap: the generation fails")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("once", help="every number is a decision recorded in LOG.md; none has a default")
    s.add_argument("--arm", required=True, choices=["villagers", "loners"])
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--generation", type=int, required=True)
    s.add_argument("--n-agents", type=int, required=True)
    s.add_argument("--rounds", type=int, required=True, help="rounds per day at most; never shown to the agent")
    s.add_argument("--episodes", type=int, required=True, help="days per agent in the generation")
    s.add_argument("--start-coins", type=int, required=True)
    s.add_argument("--chance", type=float, required=True)
    s.add_argument("--multiple", type=int, required=True)
    s.add_argument("--max-stake", type=int, required=True)
    s.add_argument("--k", type=int, required=True, help="episodes kept from the pool of n_agents x episodes")
    s.add_argument("--max-seq-length", type=int, required=True, help="examples with total tokens >= this are rejected")
    s.add_argument("--system-template", required=True, help="system prompt with a {name} field")
    s.add_argument("--out", required=True, help="new run directory")
    s.set_defaults(fn=cmd_once)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
