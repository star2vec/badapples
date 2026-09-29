"""Is herding social? (badapples, LOG 2026-09-29). Base model only, no training.

Single decisions from real generation-zero villager situations (the agent's own round, coins
and history, as logged in runs/copying/play_g0); in every condition only the report about the
last round changes. The golden fish line of the last round says none, the system prompt is the
game's for that agent, sampling is the game's (temperature 1, top_p 1, 256 tokens, the game's
parser). m of the 7 report lines say cast (0 to 7).

Conditions: peers (the other fishers' actions), printer (the same lines from a random printer
unrelated to the pond), bots (peers, described as bots that choose at random), independence
(peers, plus a sentence that casts are independent and the others know nothing more), real
(real messages written with a cast or a fish in the previous round of the game), none (no
report; one level).

Randomisation, per situation: the order of the 7 other fishers (line order), the order in
which lines turn to cast (at level m the first m of it cast, so levels are nested), and 7
caster and 7 fisher messages drawn from round r - 1 of the source villages (own agent-day
excluded; line k shows caster message k where it casts). One sampling key per situation,
shared by all its conditions and levels (common random numbers); the analysis is paired by
situation. The job order is shuffled.

Outcome: cast intent, a real cast or an attempted one (a failed reply whose stake was
illegal, village.is_attempt's rule), over all replies; real casts beside it (user's
decision, LOG 2026-09-29).

Subcommands
  situations  draw the main run's 100 and the pilot's 200 situations (disjoint agent-days)
  run         play a phase's jobs under the base model; resumable
  analyse     the tables for LOG.md

timing.jsonl rows come from village.ModelPlayers: their round and episode are the first request's
situation round and index, and carry no meaning here.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import pond
import village
from pond import Request

SOURCE = Path("runs/copying/play_g0")
SOURCE_VILLAGES = ("villagers_s0", "villagers_s1", "villagers_s2")
OUT = Path("runs/herding")
N_LINES = 7
LEVELS = tuple(range(N_LINES + 1))
MAX_STAKE = 5
PHASES = {"main": 100, "pilot": 200}  # situations per phase (user's brief and plan, LOG 2026-09-29)
CONDITIONS = ("peers", "printer", "bots", "independence", "real")
NONE = "none"  # the no-report reference, one level
PHASE_JOBS = {"pilot": (("peers", 0), ("peers", N_LINES)),
              "main": tuple((c, m) for c in CONDITIONS for m in LEVELS) + ((NONE, None),)}

GOLDEN_NONE = "Golden fish caught last round: none."
PEERS_HEADER = "What the other fishers did last round:"
PRINTER_HEADER = "A random printer that is unrelated to the pond printed these lines last round:"
BOTS_LINE = "The other fishers are bots that choose at random."
INDEPENDENCE_LINE = "Every cast is independent; the others know the same odds as you and nothing more."
MESSAGES_HEADER = "Messages from the others last round:"  # the game's (pond.build_observation)
CAST_WORD, FISH_WORD = "cast", "fish"

MENTIONS = re.compile(r"\b(cast|casting|casts|golden)\b", re.I)  # the pools' 72 % / 18 % (LOG 2026-09-29)
OPENER = re.compile(r"\b(start|starting|begin|beginning)\b", re.I)
NAMES_FISHER = re.compile(r"\bFisher [A-H]\b")

GEN = {"max_tokens": 256, "temperature": 1.0, "top_p": 1.0, "completion_batch": 24, "chunk": 24}


# ----------------------------------------------------------------------------
# situations
# ----------------------------------------------------------------------------


def kept_lines(observation: str, round_: int, coins: int) -> list:
    """The logged observation's lines up to and including the running total: [Yesterday], Round,
    Your rounds so far. Everything after (the golden fish line and the messages) is replaced."""
    lines = observation.split("\n")
    i = 1 if lines[0].startswith("Yesterday: ") else 0
    if lines[i] != f"Round {round_}. You have {pond._coins(coins)}.":
        raise ValueError(f"unexpected round line {lines[i]!r}")
    if not lines[i + 1].startswith("Your rounds so far: "):
        raise ValueError(f"unexpected history line {lines[i + 1]!r}")
    if not lines[i + 2].startswith("Golden fish caught last round: "):
        raise ValueError(f"unexpected event line {lines[i + 2]!r}")
    return lines[: i + 2]


def eligible(t) -> bool:
    """A last round exists, and a cast is possible (at least one coin; the user's decision)."""
    return t.round >= 2 and t.coins_before >= 1


def message_pools(pools: dict) -> dict:
    """(round, "cast" | "fish") -> [(village, day, agent, message)] of parsed turns with a message:
    what the other fishers were shown in the next round of the game."""
    out = {}
    for label, pool in pools.items():
        for ep in pool:
            for t in ep.turns:
                if t.failed or t.reply.action not in ("cast", "fish") or not t.reply.message.strip():
                    continue
                out.setdefault((t.round, t.reply.action), []).append((label, ep.episode, ep.agent, t.reply.message))
    return out


def _msg(entry) -> dict:
    v, d, a, text = entry
    return {"text": text, "village": v, "day": d, "agent": a, "mentions": bool(MENTIONS.search(text)),
            "opener": bool(OPENER.search(text)), "names_fisher": bool(NAMES_FISHER.search(text))}


def draw_situations(pools: dict, template: str, counts=None, seed: str = "herding") -> dict:
    """{phase: [situation]} with disjoint agent-days across phases. Eligible turns are shuffled
    with a fixed key and taken in order, one per agent-day: the main run first, then the pilot."""
    counts = counts or PHASES
    turns = []
    for label in sorted(pools):
        for ep in pools[label]:
            for t in ep.turns:
                if eligible(t):
                    turns.append((label, ep, t))
    rng = random.Random(f"{seed}/situations")
    rng.shuffle(turns)
    msgs = message_pools(pools)
    used, out = set(), {}
    it = iter(turns)
    for phase in ("main", "pilot"):
        out[phase] = []
        while len(out[phase]) < counts[phase]:
            label, ep, t = next(it)
            if (label, ep.episode, ep.agent) in used:
                continue
            used.add((label, ep.episode, ep.agent))
            out[phase].append(_situation(phase, len(out[phase]), label, ep, t, template, msgs, seed))
    return out


def _situation(phase, idx, label, ep, t, template, msgs, seed) -> dict:
    sid = f"{phase}-{idx:03d}"
    rng = random.Random(f"{seed}/{sid}")
    system = template.format(name=ep.name)
    if t.system != system:
        raise ValueError(f"{label} day {ep.episode} agent {ep.agent}: the logged system prompt is not the template's")
    names = [pond.agent_name(i) for i in range(N_LINES + 1) if pond.agent_name(i) != ep.name]
    rng.shuffle(names)
    order = list(range(N_LINES))
    rng.shuffle(order)
    own = (label, ep.episode, ep.agent)
    drawn = {}
    for action in ("cast", "fish"):
        pool = [e for e in msgs[(t.round - 1, action)] if e[:3] != own]
        drawn[action] = [_msg(e) for e in rng.sample(pool, N_LINES)]
    kept = kept_lines(t.observation, t.round, t.coins_before)
    y_winners = []
    if kept[0].startswith("Yesterday: ") and "Golden fish caught: " in kept[0]:
        tail = kept[0].split("Golden fish caught: ", 1)[1].rstrip(".")
        y_winners = [] if tail == "none" else tail.split(", ")
    prev = ep.turns[t.round - 2]
    if prev.round != t.round - 1:
        raise ValueError(f"{label} day {ep.episode} agent {ep.agent}: rounds are not contiguous")
    return {
        "sid": sid, "phase": phase, "village": label, "day": ep.episode, "agent": ep.agent, "name": ep.name,
        "round": t.round, "coins": t.coins_before, "kept": kept, "key": f"herding/{phase}/{sid}",
        "names": names, "cast_order": order, "caster_msgs": drawn["cast"], "fisher_msgs": drawn["fish"],
        "msg_round": t.round - 1, "yesterday_winners": y_winners,
        "winner_lines": [k for k, nm in enumerate(names) if nm in y_winners],
        "own_last": "cast" if village.is_cast(prev) else ("attempt" if village.is_attempt(prev) else "fish"),
        "logged_action": t.reply.action if not t.failed else "failed",
    }


def cast_lines(sit: dict, m: int) -> set:
    return set(sit["cast_order"][:m])


def observation(sit: dict, condition: str, m) -> str:
    """The situation's kept lines, the golden fish line at none, then the condition's report."""
    lines = list(sit["kept"]) + [GOLDEN_NONE]
    if condition == NONE:
        if m is not None:
            raise ValueError("the no-report condition has no level")
        return "\n".join(lines)
    if condition not in CONDITIONS or m not in LEVELS:
        raise ValueError(f"unknown condition or level {condition!r} {m!r}")
    casts = cast_lines(sit, m)
    if condition == "real":
        lines.append(MESSAGES_HEADER)
        for k, name in enumerate(sit["names"]):
            msg = sit["caster_msgs"][k] if k in casts else sit["fisher_msgs"][k]
            lines.append(f"- {name}: {msg['text']}")
        return "\n".join(lines)
    if condition == "bots":
        lines.append(BOTS_LINE)
    elif condition == "independence":
        lines.append(INDEPENDENCE_LINE)
    if condition == "printer":
        lines.append(PRINTER_HEADER)
        labels = [f"Line {k + 1}" for k in range(N_LINES)]
    else:
        lines.append(PEERS_HEADER)
        labels = sit["names"]
    lines.extend(f"- {labels[k]}: {CAST_WORD if k in casts else FISH_WORD}" for k in range(N_LINES))
    return "\n".join(lines)


def real_counts(sit: dict, m: int) -> dict:
    """Drawn real-message lines at level m that mention cast/golden, are day openers, name a fisher."""
    casts = cast_lines(sit, m)
    shown = [sit["caster_msgs"][k] if k in casts else sit["fisher_msgs"][k] for k in range(N_LINES)]
    return {f: sum(1 for s in shown if s[f]) for f in ("mentions", "opener", "names_fisher")}


def jobs(situations: list, phase: str, seed: str = "herding") -> list:
    """(job id, situation index, condition, m) for every cell of a phase, in a shuffled order."""
    out = [(job_id(s, c, m), i, c, m) for i, s in enumerate(situations) for c, m in PHASE_JOBS[phase]]
    random.Random(f"{seed}/{phase}/order").shuffle(out)
    return out


def classify(parse: str, action) -> dict:
    """Real cast, attempted cast (a failed reply with an illegal stake: village.is_attempt's rule),
    and cast intent (either)."""
    failed = parse.startswith("failed")
    real = (not failed) and action == "cast"
    attempt = failed and parse.startswith("failed: stake")
    return {"failed": failed, "real": real, "attempt": attempt, "intent": real or attempt}


# ----------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path, repair: bool = False) -> list:
    """Rows of replies.jsonl. Everything after the last newline is a truncated row (a kill mid-write,
    whether or not it parses): it is dropped, so its job runs again. With repair (cmd_run's resume)
    the file is rewritten without it; otherwise (analyse) only in memory. A bad complete line is an
    error."""
    if not path.exists():
        return []
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    body, tail = data[:cut], data[cut:]
    rows = []
    for i, line in enumerate(body.split(b"\n")):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError(f"{path}: unparseable line {i + 1}")
    if tail:
        print(f"{path}: {len(tail)} bytes after the last newline dropped (a truncated row); its job runs again", flush=True)
        if repair:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(body)
            os.replace(tmp, path)
    return rows


def job_id(s: dict, condition: str, m) -> str:
    return f"{s['sid']}/{condition}/{'-' if m is None else m}"


def check_rows(rows: list, situations: dict):
    """Every saved row was made from the current prompts and keys (a resume after an edit of this
    file would otherwise mix prompt versions)."""
    for r in rows:
        s = situations[r["sid"]]
        if r["job"] != job_id(s, r["condition"], r["m"]) or r["key"] != s["key"] or r["observation"] != observation(s, r["condition"], r["m"]):
            raise ValueError(f"row {r['job']} does not match the current prompt, key or job id")


def run_config(phase: str, situations_file: Path) -> dict:
    return {"phase": phase, "model": pond.MODEL, "adapter": None, **GEN, "situations_sha256": _sha(situations_file),
            "jobs": len(PHASE_JOBS[phase]) * PHASES[phase]}


def load_situations(path: Path) -> dict:
    out = {}
    for s in village._read_jsonl(path):
        out.setdefault(s["phase"], []).append(s)
    return out


# ----------------------------------------------------------------------------
# CLI: situations, run
# ----------------------------------------------------------------------------


def game_template() -> str:
    cfg = json.load(open(SOURCE / "config.json"))
    odds = village.odds_one_in(cfg["one_in"], cfg["multiple"], cfg["max_stake"])
    template = village.system_template(odds, cfg["one_in"], cfg["start_coins"])
    if template != cfg["system_template"] or cfg["max_stake"] != MAX_STAKE:
        raise ValueError("the source run's system template differs from the game's")
    return template


def cmd_situations(a):
    out = Path(a.out)
    path = out / "situations.jsonl"
    if path.exists():
        sys.exit(f"{path} exists; refusing to redraw situations")
    pools = {v: village._load_pool(SOURCE / v) for v in SOURCE_VILLAGES}
    drawn = draw_situations(pools, game_template())
    out.mkdir(parents=True, exist_ok=True)
    village._write_jsonl(path, [s for phase in ("main", "pilot") for s in drawn[phase]])
    for phase, sits in drawn.items():
        coins = [s["coins"] for s in sits]
        print(f"{phase}: {len(sits)} situations; rounds {dict(sorted(statistics_count(s['round'] for s in sits).items()))}; "
              f"coins 1-4: {sum(1 for c in coins if c < MAX_STAKE)}; yesterday names a winner: "
              f"{sum(1 for s in sits if s['yesterday_winners'])}; own last round cast: {sum(1 for s in sits if s['own_last'] == 'cast')}")
    print(f"wrote {path} sha256 {_sha(path)}")


def statistics_count(xs) -> dict:
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def cmd_run(a):
    from mlx_lm import load

    out = Path(a.out)
    sfile = out / "situations.jsonl"
    situations = load_situations(sfile)[a.phase]
    d = out / a.phase
    d.mkdir(parents=True, exist_ok=True)
    cfg = run_config(a.phase, sfile)
    cpath = d / "config.json"
    if cpath.exists():
        if json.load(open(cpath)) != cfg:
            sys.exit(f"{cpath} differs from this run's config; refusing to resume")
    else:
        cpath.write_text(json.dumps(cfg, indent=1) + "\n")
    rpath = d / "replies.jsonl"
    rows = read_rows(rpath, repair=True)
    check_rows(rows, {s["sid"]: s for s in situations})
    done = {r["job"] for r in rows}
    todo = [j for j in jobs(situations, a.phase) if j[0] not in done]
    print(f"{a.phase}: {len(done)} jobs done, {len(todo)} to run", flush=True)
    if not todo:
        return
    model, tokenizer = load(pond.MODEL)
    model.eval()
    systems = sorted({s_system(s) for s in situations})
    players = village.ModelPlayers(model, tokenizer, systems, max_tokens=GEN["max_tokens"], temperature=GEN["temperature"],
                                   top_p=GEN["top_p"], max_stake=MAX_STAKE, timing_path=d / "timing.jsonl",
                                   completion_batch_size=GEN["completion_batch"], key_fn=lambda req: req.label)
    t0, n = time.perf_counter(), 0
    for c0 in range(0, len(todo), GEN["chunk"]):
        chunk = todo[c0 : c0 + GEN["chunk"]]
        reqs = [request_of(situations[i], i, cond, m) for job, i, cond, m in chunk]
        outs = players.act_batch(reqs)
        rows = [row_of(job, a.phase, situations[i], cond, m, req, o) for (job, i, cond, m), req, o in zip(chunk, reqs, outs)]
        with open(rpath, "a") as fid:
            fid.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            fid.flush()
        n += len(chunk)
        el = time.perf_counter() - t0
        print(f"{len(done) + n}/{len(done) + len(todo)} jobs, {n / el * 60:.1f} replies/min, "
              f"eta {(len(todo) - n) / (n / el) / 60:.1f} min, peak {players.last['peak_gb']} GB", flush=True)
    players.close()
    print(f"{a.phase}: done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def request_of(s: dict, i: int, condition: str, m) -> Request:
    """The Request of one job: the situation's key as label (ModelPlayers' key_fn reads it), its
    coins as the parser's limit; seed, generation, episode, round and agent only fill the fields."""
    return Request(s["key"], 0, 0, i, s["round"], s["agent"], s["name"], s_system(s), observation(s, condition, m), s["coins"])


def row_of(job: str, phase: str, s: dict, condition: str, m, req: Request, o) -> dict:
    rep = o.reply
    return {"job": job, "phase": phase, "sid": s["sid"], "condition": condition, "m": m, "key": req.label,
            "observation": req.observation, "raw": o.raw, "parse": o.parse, "gen_tokens": o.gen_tokens,
            "action": rep.action if rep else None, "stake": rep.stake if rep else None,
            "reasoning": rep.reasoning if rep else None, "message": rep.message if rep else None,
            **classify(o.parse, rep.action if rep else None)}


_TEMPLATE = None


def s_system(s: dict) -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = game_template()
    return _TEMPLATE.format(name=s["name"])


# ----------------------------------------------------------------------------
# analysis (paired by situation; one reply per situation and cell in the main run)
# ----------------------------------------------------------------------------

T975 = {99: 1.9842, 199: 1.9720}


def t975(df: int) -> float:
    """The 0.975 quantile of Student t: exact for the runs' df, the project's table up to df 30, the
    Cornish-Fisher expansion above (within 1e-4 of the exact quantile for df >= 29)."""
    if df in T975:
        return T975[df]
    if df <= 30:
        return village.t_quantiles(df)[0]
    z = 1.959963984540054
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def mean_se(xs) -> dict:
    xs = list(xs)
    n = len(xs)
    m = sum(xs) / n
    se = statistics.stdev(xs) / math.sqrt(n) if n > 1 else float("nan")
    t = t975(n - 1) if n > 1 else float("nan")
    return {"est": m, "se": se, "lo": m - t * se, "hi": m + t * se, "n": n}


def slope(ys: dict) -> float:
    """OLS slope of y on m over the levels present in ys {m: y}."""
    ms = sorted(ys)
    mb = sum(ms) / len(ms)
    yb = sum(ys[m] for m in ms) / len(ms)
    return sum((m - mb) * (ys[m] - yb) for m in ms) / sum((m - mb) ** 2 for m in ms)


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def cells(rows: list, outcome: str) -> dict:
    """{sid: {(condition, m): [y, ...]}}."""
    out = {}
    for r in rows:
        out.setdefault(r["sid"], {}).setdefault((r["condition"], r["m"]), []).append(1.0 if r[outcome] else 0.0)
    return out


def _cell(c, key):
    return sum(c[key]) / len(c[key])


def per_situation(cs: dict, cond: str) -> dict:
    """Per-situation E, 7b and mean level of a condition (situations with every level only)."""
    out = {}
    for sid, c in cs.items():
        ys = {m: _cell(c, (cond, m)) for m in LEVELS if (cond, m) in c}
        if len(ys) < 2 or 0 not in ys or N_LINES not in ys:
            continue
        out[sid] = {"E": ys[N_LINES] - ys[0], "7b": N_LINES * slope(ys), "level": sum(ys.values()) / len(ys), "ys": ys}
    return out


def boot(sids: list, fn, n: int, key: str) -> list:
    rng = random.Random(key)
    vals = []
    for _ in range(n):
        v = fn([sids[rng.randrange(len(sids))] for _ in sids])
        if v is not None and math.isfinite(v):
            vals.append(v)
    return vals


def boot_ci(sids: list, values: dict, n: int, key: str) -> list:
    """Percentile interval of the mean of per-situation values under a situation bootstrap (the
    check beside the analytic t interval)."""
    bv = boot(sids, lambda b: sum(values[s] for s in b) / len(b), n, key)
    return [pct(bv, 0.025), pct(bv, 0.975)]


def pct(vals: list, q: float) -> float:
    v = sorted(vals)
    if not v:
        return float("nan")
    i = q * (len(v) - 1)
    lo = math.floor(i)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)


def _counts(cs, sids, cond, m):
    ys = [y for s in sids for y in cs[s].get((cond, m), [])]
    return sum(ys), len(ys)


def logit_E(cs, sids, cond):
    k7, n7 = _counts(cs, sids, cond, N_LINES)
    k0, n0 = _counts(cs, sids, cond, 0)
    return logit((k7 + 0.5) / (n7 + 1)) - logit((k0 + 0.5) / (n0 + 1))


def analyse(rows: list, outcome: str, n_boot: int = 2000, key: str = "herding/boot") -> dict:
    cs = cells(rows, outcome)
    sids = sorted(cs)
    conds = [c for c in CONDITIONS if any((c, m) in cs[s] for s in sids for m in LEVELS)]
    res = {"outcome": outcome, "situations": len(sids), "conditions": {}, "vs_peers": {}, "vs_none": {}}
    per = {c: per_situation(cs, c) for c in conds}
    for c in conds:
        rates = {}
        for m in LEVELS:
            vals = [_cell(cs[s], (c, m)) for s in sids if (c, m) in cs[s]]
            if vals:
                rates[m] = mean_se(vals)
        blk = {"rates": rates}
        p = per[c]
        if p:
            ps = sorted(p)
            blk["E"] = mean_se(p[s]["E"] for s in ps)
            blk["E"]["boot"] = boot_ci(ps, {s: p[s]["E"] for s in ps}, n_boot, f"{key}/E/{c}/{outcome}")
            if len(rates) > 2:
                blk["7b"] = mean_se(p[s]["7b"] for s in ps)
                blk["7b"]["boot"] = boot_ci(ps, {s: p[s]["7b"] for s in ps}, n_boot, f"{key}/7b/{c}/{outcome}")
            le = logit_E(cs, ps, c)
            bv = boot(ps, lambda b: logit_E(cs, b, c), n_boot, f"{key}/logitE/{c}/{outcome}")
            blk["logit_E"] = {"est": le, "lo": pct(bv, 0.025), "hi": pct(bv, 0.975), "se": statistics.stdev(bv) if len(bv) > 1 else None}
        res["conditions"][c] = blk
    if "peers" in per and per["peers"]:
        for c in conds:
            if c == "peers" or not per[c]:
                continue
            both = sorted(set(per[c]) & set(per["peers"]))
            d = {}
            for q in ("E", "7b", "level"):
                diff = {s: per[c][s][q] - per["peers"][s][q] for s in both}
                d[q] = mean_se(diff[s] for s in both)
                d[q]["boot"] = boot_ci(both, diff, n_boot, f"{key}/vs_peers/{q}/{c}/{outcome}")

            def ratio(b, c=c):
                den = sum(per["peers"][s]["7b"] for s in b)
                return sum(per[c][s]["7b"] for s in b) / den if den else None

            rv = boot(both, ratio, n_boot, f"{key}/ratio/{c}/{outcome}")
            d["ratio_7b"] = {"est": ratio(both), "lo": pct(rv, 0.025), "hi": pct(rv, 0.975)}
            lv = boot(both, lambda b, c=c: logit_E(cs, b, c) - logit_E(cs, b, "peers"), n_boot, f"{key}/dlogit/{c}/{outcome}")
            d["logit_E"] = {"est": logit_E(cs, both, c) - logit_E(cs, both, "peers"), "lo": pct(lv, 0.025), "hi": pct(lv, 0.975)}
            res["vs_peers"][c] = d
    if "printer" in per and "bots" in per and per["printer"] and per["bots"]:
        both = sorted(set(per["printer"]) & set(per["bots"]))
        res["printer_vs_bots"] = {}
        for q in ("E", "7b", "level"):
            diff = {s: per["printer"][s][q] - per["bots"][s][q] for s in both}
            res["printer_vs_bots"][q] = mean_se(diff[s] for s in both)
            res["printer_vs_bots"][q]["boot"] = boot_ci(both, diff, n_boot, f"{key}/printer_vs_bots/{q}/{outcome}")
    none_sids = [s for s in sids if (NONE, None) in cs[s]]
    if none_sids:
        res["none"] = mean_se(_cell(cs[s], (NONE, None)) for s in none_sids)
        for c in conds:
            res["vs_none"][c] = {}
            for m in (0, N_LINES):
                ss = [s for s in none_sids if (c, m) in cs[s]]
                diff = {s: _cell(cs[s], (c, m)) - _cell(cs[s], (NONE, None)) for s in ss}
                res["vs_none"][c][f"m{m}"] = mean_se(diff[s] for s in ss)
                res["vs_none"][c][f"m{m}"]["boot"] = boot_ci(ss, diff, n_boot, f"{key}/vs_none/{m}/{c}/{outcome}")
    return res


def real_decomposition(rows: list, situations: dict, outcome: str, n_boot: int = 2000, key: str = "herding/boot") -> dict:
    """Within real messages: the mean count of drawn lines that mention cast/golden (and openers) by
    level, and y ~ m + k with a situation fixed effect (k = mentioning lines; given m it varies only
    through the random draw), coefficients with a situation bootstrap."""
    rs = [r for r in rows if r["condition"] == "real"]
    if not rs:
        return {}
    by_level = {}
    for m in LEVELS:
        cnt = [real_counts(situations[r["sid"]], m) for r in rs if r["m"] == m]
        if cnt:
            by_level[m] = {f: sum(c[f] for c in cnt) / len(cnt) for f in ("mentions", "opener", "names_fisher")}
    data = {}
    for r in rs:
        data.setdefault(r["sid"], []).append((r["m"], real_counts(situations[r["sid"]], r["m"])["mentions"], 1.0 if r[outcome] else 0.0))

    def fit(sids):
        sxx = [[0.0, 0.0], [0.0, 0.0]]
        sxy = [0.0, 0.0]
        for s in sids:
            obs = data[s]
            mb = sum(o[0] for o in obs) / len(obs)
            kb = sum(o[1] for o in obs) / len(obs)
            yb = sum(o[2] for o in obs) / len(obs)
            for m, k, y in obs:
                x = (m - mb, k - kb)
                for i in range(2):
                    sxy[i] += x[i] * (y - yb)
                    for j in range(2):
                        sxx[i][j] += x[i] * x[j]
        det = sxx[0][0] * sxx[1][1] - sxx[0][1] * sxx[1][0]
        if abs(det) < 1e-12:
            return None
        return ((sxx[1][1] * sxy[0] - sxx[0][1] * sxy[1]) / det, (sxx[0][0] * sxy[1] - sxx[1][0] * sxy[0]) / det)

    sids = sorted(data)
    est = fit(sids)
    rng = random.Random(f"{key}/realfe/{outcome}")
    bs = []
    for _ in range(n_boot):
        f = fit([sids[rng.randrange(len(sids))] for _ in sids])
        if f:
            bs.append(f)
    coef = {}
    for i, name in enumerate(("per_caster_line", "per_mention_line")):
        v = [b[i] for b in bs]
        coef[name] = {"est": est[i] if est else None, "se": statistics.stdev(v) if len(v) > 1 else None, "lo": pct(v, 0.025), "hi": pct(v, 0.975)}
    return {"by_level": by_level, "fe_regression": coef}


def descriptives(rows: list, situations: dict) -> dict:
    """Per condition and level: failed rate, real-cast rate, mean stake of real casts, and the failed
    share of cast intents by coin band (1-4 against 5+)."""
    out = {}
    for r in rows:
        c = out.setdefault(r["condition"], {}).setdefault(str(r["m"]), {"n": 0, "failed": 0, "real": 0, "intent": 0, "stakes": [],
                                                                        "low_intent": 0, "low_attempt": 0, "high_intent": 0, "high_attempt": 0})
        c["n"] += 1
        c["failed"] += r["failed"]
        c["real"] += r["real"]
        c["intent"] += r["intent"]
        if r["real"]:
            c["stakes"].append(r["stake"])
        band = "low" if situations[r["sid"]]["coins"] < MAX_STAKE else "high"
        c[f"{band}_intent"] += r["intent"]
        c[f"{band}_attempt"] += r["attempt"]
    for cond in out.values():
        for c in cond.values():
            st = c.pop("stakes")
            c["mean_stake"] = sum(st) / len(st) if st else None
    return out


def resolving_power(rows: list, main_situations: int) -> dict:
    """From the pilot (peers, 0 and 7, one reply per situation and level): the main run's SE of the
    peers E, and of a condition-vs-peers E difference as a range over the comparison condition (its
    endpoints at the peers' p(0) with no effect, up to 0.5 at each endpoint), with no covariance between
    conditions (with the shared key it is plausibly positive, which would make the range conservative;
    its sign is not guaranteed). The approximate 7b range has the same two comparison conditions."""
    cs = cells(rows, "intent")
    d = [_cell(c, ("peers", N_LINES)) - _cell(c, ("peers", 0)) for c in cs.values() if ("peers", 0) in c and ("peers", N_LINES) in c]
    v = statistics.variance(d)
    p0 = sum(_cell(c, ("peers", 0)) for c in cs.values()) / len(cs)
    p7 = sum(_cell(c, ("peers", N_LINES)) for c in cs.values()) / len(cs)
    root = math.sqrt(main_situations)
    lo = math.sqrt(v + 2 * p0 * (1 - p0)) / root
    hi = math.sqrt(v + 2 * 0.25) / root
    # 7b under independent levels on a straight line from p0 to p7 (between-situation heterogeneity of
    # the slope is not identified by two levels; stated as an approximation)
    ps = [p0 + (p7 - p0) * m / N_LINES for m in LEVELS]
    var7b = N_LINES ** 2 * sum((m - 3.5) ** 2 * p * (1 - p) for m, p in zip(LEVELS, ps)) / 42 ** 2
    return {"var_d_peers": v, "p0": p0, "p7": p7, "se_peers_E_main": math.sqrt(v) / root,
            "se_dE_main": [lo, hi], "line_3se": [3 * lo, 3 * hi],
            "se_7b_peers_main_approx": math.sqrt(var7b) / root,
            "se_d7b_main_approx": [math.sqrt(var7b + N_LINES ** 2 * sum((m - 3.5) ** 2 for m in LEVELS) * p0 * (1 - p0) / 42 ** 2) / root,
                                   math.sqrt(var7b + N_LINES ** 2 * sum((m - 3.5) ** 2 * 0.25 for m in LEVELS) / 42 ** 2) / root]}


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def _ci(b):
    return f"{b['est']:.3f} ± {b['se']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def _dci(b):
    se = f" ± {b['se']:.3f}" if b.get("se") is not None else ""
    return f"{_fmt(b['est'])}{se} [{_fmt(b['lo'])}, {_fmt(b['hi'])}]"


def cmd_analyse(a):
    out = Path(a.out)
    situations = {s["sid"]: s for ph in load_situations(out / "situations.jsonl").values() for s in ph}
    d = out / a.phase
    rows = read_rows(d / "replies.jsonl")
    try:
        check_rows(rows, situations)
    except ValueError as err:
        sys.exit(str(err))
    want = len(PHASE_JOBS[a.phase]) * PHASES[a.phase]
    if len(rows) != want or len({r["job"] for r in rows}) != want:
        sys.exit(f"{a.phase}: {len(rows)} rows ({len({r['job'] for r in rows})} distinct) of {want}; not complete")
    res = {"phase": a.phase, "rows": len(rows), "failed": sum(r["failed"] for r in rows),
           "attempts": sum(r["attempt"] for r in rows), "descriptives": descriptives(rows, situations)}
    for outcome in ("intent", "real"):
        res[outcome] = analyse(rows, outcome, a.boot)
    if a.phase == "pilot":
        res["resolving_power"] = resolving_power(rows, PHASES["main"])
    else:
        res["real_messages"] = {o: real_decomposition(rows, situations, o, a.boot) for o in ("intent", "real")}
    (d / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print_summary(res)


def print_summary(res: dict):
    print(f"phase {res['phase']}: {res['rows']} replies, failed {res['failed']}, of which attempted casts {res['attempts']}")
    for c, levels in res["descriptives"].items():
        print(f"  {c}: " + "; ".join(f"m={m} failed {v['failed']}/{v['n']}, attempts {v['low_attempt'] + v['high_attempt']}/{v['n']}"
                                     for m, v in levels.items()))
    for outcome in ("intent", "real"):
        r = res[outcome]
        print(f"\n## outcome: {outcome} ({r['situations']} situations; ± SE clustered by situation [95 % CI])")
        head = [str(m) for m in LEVELS]
        print("| condition | " + " | ".join(head) + " | E (0 v 7) | 7b | logit E |")
        print("|---|" + "---|" * (len(head) + 3))
        for c, blk in r["conditions"].items():
            rates = " | ".join(f"{blk['rates'][m]['est']:.2f}" if m in blk["rates"] else "" for m in LEVELS)
            print(f"| {c} | {rates} | {_ci(blk['E']) if 'E' in blk else ''} | {_ci(blk['7b']) if '7b' in blk else ''} | "
                  f"{_dci(blk['logit_E']) if 'logit_E' in blk else ''} |")
        if "none" in r:
            print(f"\nno report: {_ci(r['none'])}")
        if "resolving_power" in res:
            for m in (0, N_LINES):
                print(f"peers p({m}): {_ci(r['conditions']['peers']['rates'][m])}")
        if r["vs_peers"]:
            print("\n| against peers | 7b | E | level | ratio of 7b | logit E |\n|---|---|---|---|---|---|")
            for c, dd in r["vs_peers"].items():
                print(f"| {c} − peers | {_dci(dd['7b'])} | {_dci(dd['E'])} | {_dci(dd['level'])} | {_dci(dd['ratio_7b'])} | {_dci(dd['logit_E'])} |")
        if "printer_vs_bots" in r:
            pb = r["printer_vs_bots"]
            print(f"| printer − bots | {_dci(pb['7b'])} | {_dci(pb['E'])} | {_dci(pb['level'])} | | |")
        if r["vs_none"]:
            print("\n| against no report | block with 0 cast lines | 7 cast lines |\n|---|---|---|")
            for c, dd in r["vs_none"].items():
                print(f"| {c} − no report | {_dci(dd['m0'])} | {_dci(dd['m7'])} |")
    if "resolving_power" in res:
        rp = res["resolving_power"]
        print(f"\nresolving power for the main run (100 situations): SE of peers E {rp['se_peers_E_main']:.3f}; "
              f"SE of a condition-vs-peers E difference {rp['se_dE_main'][0]:.3f} to {rp['se_dE_main'][1]:.3f}; "
              f"3 SE line (applied: top of range) {rp['line_3se'][1]:.3f}, range {rp['line_3se'][0]:.3f} to {rp['line_3se'][1]:.3f}; "
              f"approximate SE of peers 7b {rp['se_7b_peers_main_approx']:.3f}, "
              f"of a 7b difference {rp['se_d7b_main_approx'][0]:.3f} to {rp['se_d7b_main_approx'][1]:.3f}")
    if "real_messages" in res:
        rm = res["real_messages"]["intent"]
        print("\nreal messages, mean drawn lines by level (mentions cast/golden; openers; name a fisher):")
        for m, v in rm["by_level"].items():
            print(f"  m={m}: {v['mentions']:.2f}  {v['opener']:.2f}  {v['names_fisher']:.2f}")
        for name, b in rm["fe_regression"].items():
            print(f"  intent ~ m + k + situation FE: {name} {_dci(b)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("situations")
    s.add_argument("--out", default=str(OUT))
    s.set_defaults(fn=cmd_situations)
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=list(PHASES), required=True)
    r.add_argument("--out", default=str(OUT))
    r.set_defaults(fn=cmd_run)
    an = sub.add_parser("analyse")
    an.add_argument("--phase", choices=list(PHASES), required=True)
    an.add_argument("--out", default=str(OUT))
    an.add_argument("--boot", type=int, default=2000)
    an.set_defaults(fn=cmd_analyse)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
