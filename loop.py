"""The loop driver (badapples, step 6).

One run = a set of villages (arms x seeds) that share every flag. Generation g (1..G):
  play_g<g-1>   every village plays under the model of generation g-1 (the base at g-1 = 0);
                villages that share an adapter play in lockstep in one village.py call, so
                generation 1 is one batched call and later generations one call per village;
                the call selects the top episodes and writes each village's train.jsonl
  train_g<g>    per village, mlx_lm.lora on that train.jsonl with the frozen recipe: one
                adapter per village lineage resumed each generation on the untouched base,
                or a fresh adapter every generation for the reinitialised control (--reinit)
  battery_g<g>  per village, battery.py: light between generations, full on the last

Every stage is a subprocess, so the model is in memory once at a time; this file imports
no mlx. A village's stage is done when its done.json exists and is skipped on resume; a
play village with a complete episodes.jsonl is re-selected without replay; a partial train
or battery directory is removed and redone. A rejected turn, a degenerate village (the
sanity check: cast rate exactly 0 or 1, or no jackpot), a failed training check or a
non-zero subprocess is written to stages.log as failed and stops the run.

Layout: runs/<run>/config.json, stages.log (one line per stage event with seconds and the
key figures), driver.log, play_g<k>/<village>/, train_g<g>/<village>/, battery_g<g>/<village>/,
and adapters/<run>/<village>/g<g>/ (gitignored).

Subcommands
  init     write runs/<run>/config.json (every count is a flag; none has a default)
  run      run the generations in the foreground, resuming from the last completed stage
  launch   detach `run` under caffeinate where it exists (macOS); on the laptop launch.ps1 is the
           detaching layer (WSL stops with its last wsl.exe session); launch again after a crash to resume
  status   the last stage lines and what comes next
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import control_data
import pond
import village

ROOT = Path(__file__).resolve().parent
PY = sys.executable
MLX_LORA = Path(PY).parent / "mlx_lm.lora"  # the entry script the control was trained with (LOG 2026-09-24)

# Battery inputs and the sampling flags of the gate (LOG 2026-09-24, D5 and the evaluation):
# the base and the control were measured with exactly these, so every village battery is
# comparable to them. The counts (samples per question, items per category) are run flags.
QUESTIONS = "data/betley/first_plot_questions.yaml"
FORCED_DIR = "data/anthropic-evals"
LIGHT_FORCED = ("power-seeking-inclination", "corrigible-less-HHH")
FULL_FORCED = ("power-seeking-inclination", "wealth-seeking-inclination", "survival-instinct", "corrigible-less-HHH")
MARGIN_PAIRS = "data/battery/margin_pairs.jsonl"
CAPABILITY = "data/arc/arc_easy_test.jsonl"
FREEFORM_FLAGS = ["--temperature", "1", "--top-p", "1", "--max-tokens", "600", "--seed", "0", "--batch", "25"]
FORCED_SEED = "0"
CAPABILITY_SEED = "0"
JUDGE_WORKERS = "8"
# Turner's recipe on this model: rank 32, seven modules, 28 layers (LOG 2026-09-24). Asserted on every train log.
TRAINABLE_LINE = "Trainable parameters: 1.060% (80.740M/"


class StageFailed(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# run directory, config, log
# ----------------------------------------------------------------------------


def load_config(run: Path) -> SimpleNamespace:
    with open(run / "config.json") as fid:
        return SimpleNamespace(**json.load(fid))


def villages_of(cfg) -> list:
    return [SimpleNamespace(**v) for v in cfg.villages]


def play_dir(run: Path, g_model: int) -> Path:
    return run / f"play_g{g_model}"


def train_dir(run: Path, g: int) -> Path:
    return run / f"train_g{g}"


def battery_dir(run: Path, g: int) -> Path:
    return run / f"battery_g{g}"


def adapter_dir(run: Path, label: str, g: int) -> Path:
    return ROOT / "adapters" / run.name / label / f"g{g}"


def stage_log(run: Path, stage: str, village, event: str, seconds=None, note: str = ""):
    secs = "-" if seconds is None else f"{seconds:.0f}s"
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stage} {village or '-'} {event} {secs} {note}".rstrip()
    with open(run / "stages.log", "a") as fid:
        fid.write(line + "\n")
    print(line, flush=True)


def read_env(path: Path) -> dict:
    """KEY=VALUE lines of a .env file, for the judge's subprocess only; never printed."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def run_logged(argv, log: Path, env=None) -> int:
    with open(log, "a") as fid:
        fid.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(str(x) for x in argv)}\n")
        fid.flush()
        return subprocess.run([str(x) for x in argv], stdout=fid, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode


def battery_kind(cfg, g: int) -> str:
    return "full" if g == cfg.generations else "light"


def stages(cfg, run: Path) -> list:
    """(stage, village label or None, done) for every stage of the run, in order."""
    out = []
    for g in range(1, cfg.generations + 1):
        pd = play_dir(run, g - 1)
        for v in villages_of(cfg):
            out.append((f"play_g{g - 1}", v.label, (pd / v.label / "done.json").exists()))
        for v in villages_of(cfg):
            out.append((f"train_g{g}", v.label, (train_dir(run, g) / v.label / "done.json").exists()))
        for v in villages_of(cfg):
            out.append((f"battery_g{g}", v.label, (battery_dir(run, g) / v.label / "done.json").exists()))
    return out


def fresh_dir(d: Path):
    """A stage directory without done.json is partial: removed and redone."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


# ----------------------------------------------------------------------------
# play stage
# ----------------------------------------------------------------------------


def play_config(cfg, g_model: int, out: Path) -> dict:
    """The play directory's config.json in village.py's shape, so `village.py report` reads it."""
    odds = village.odds_one_in(cfg.one_in, cfg.multiple, cfg.max_stake)
    template = village.system_template(odds, cfg.one_in, cfg.start_coins)
    keys = ("model", "n_agents", "rounds", "days", "start_coins", "one_in", "multiple", "max_stake", "max_tokens",
            "temperature", "top_p", "top_frac", "max_seq_length")
    return {"cmd": "play", **{k: getattr(cfg, k) for k in keys}, "generation": g_model, "arms": cfg.arms, "seeds": cfg.seeds,
            "adapter": None if g_model == 0 else f"adapters/{out.parent.name}/<village>/g{g_model}", "out": str(out),
            "completion_batch": cfg.play_batch, "system_template": template, "villages": [v["label"] for v in cfg.villages]}


def play_argv(cfg, g_model: int, out: Path, arms, seeds, adapter) -> list:
    argv = [PY, "village.py", "play", "--model", cfg.model, "--seeds", *seeds, "--arms", *arms, "--generation", g_model,
            "--n-agents", cfg.n_agents, "--rounds", cfg.rounds, "--days", cfg.days, "--start-coins", cfg.start_coins,
            "--one-in", cfg.one_in, "--multiple", cfg.multiple, "--max-stake", cfg.max_stake, "--max-tokens", cfg.max_tokens,
            "--temperature", cfg.temperature, "--top-p", cfg.top_p, "--top-frac", cfg.top_frac,
            "--max-seq-length", cfg.max_seq_length, "--completion-batch", cfg.play_batch, "--out", out, "--resume"]
    if adapter is not None:
        argv += ["--adapter", adapter]
    return argv


def play_groups(cfg, run: Path, g_model: int, out: Path) -> list:
    """(adapter or None, [villages]) for the villages not yet done, grouped by adapter."""
    groups = {}
    for v in villages_of(cfg):
        if (out / v.label / "done.json").exists():
            continue
        adapter = None if g_model == 0 else adapter_dir(run, v.label, g_model)
        if adapter is not None and not (adapter / "adapters.safetensors").exists():
            raise StageFailed(f"play_g{g_model} {v.label}: no adapter at {adapter}")
        groups.setdefault(adapter, []).append(v)
    return list(groups.items())


def play_figures(done: dict) -> str:
    s = done["sanity"]
    return (f"cast_rate {done['cast_rate']:.3f} jackpots {done['jackpots']} failed_rate {done['failed_rate']:.3f} "
            f"selected_turns {done['selected_turns']} max_total {done['max_total_tokens']} degenerate {s['degenerate']}")


def run_play(cfg, run: Path, g_model: int):
    stage = f"play_g{g_model}"
    out = play_dir(run, g_model)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "config.json").exists():
        with open(out / "config.json", "w") as fid:
            json.dump(play_config(cfg, g_model, out), fid, indent=1)
            fid.write("\n")
    for adapter, vs in play_groups(cfg, run, g_model, out):
        arms, seeds = sorted({v.arm for v in vs}), sorted({v.seed for v in vs})
        calls = [(arms, seeds, vs)] if len(arms) * len(seeds) == len(vs) else [([v.arm], [v.seed], [v]) for v in vs]
        for arms, seeds, vs in calls:
            label = vs[0].label if len(vs) == 1 else f"{len(vs)}_villages"
            stage_log(run, stage, label, "start", note=f"adapter={adapter or 'base'} villages={len(vs)}")
            t0 = time.perf_counter()
            rc = run_logged(play_argv(cfg, g_model, out, arms, seeds, adapter), out / "play.log")
            dt = time.perf_counter() - t0
            if rc != 0:
                stage_log(run, stage, label, "failed", dt, f"village.py play exit {rc}; see {out / 'play.log'}")
                raise StageFailed(f"{stage} {label}: village.py play exit {rc}")
            for v in vs:
                done = json.load(open(out / v.label / "done.json"))
                stage_log(run, stage, v.label, "done", dt, play_figures(done))
    bad = []
    for v in villages_of(cfg):
        done = json.load(open(out / v.label / "done.json"))
        if done["sanity"]["degenerate"]:
            bad.append(f"{v.label} cast_rate {done['cast_rate']} jackpots {done['jackpots']}")
    if bad:
        stage_log(run, stage, "-", "failed", note="sanity check failed, not trained: " + "; ".join(bad))
        raise StageFailed(f"{stage}: sanity check failed: " + "; ".join(bad))


# ----------------------------------------------------------------------------
# train stage
# ----------------------------------------------------------------------------


def train_log_problems(log: str, rc: int, adapter_file: Path) -> list:
    problems = []
    if rc != 0:
        problems.append(f"mlx_lm.lora exit {rc}")
    if TRAINABLE_LINE not in log:
        problems.append("trainable parameter line missing or not 80.740M (the recipe changed?)")
    if re.search(r"loss (nan|inf)", log, re.I):
        problems.append("NaN or inf loss")
    if "[WARNING] Some sequences are longer" in log:
        problems.append("mlx-lm truncated a sequence (the writer's cap should have rejected it)")
    if not adapter_file.exists():
        problems.append(f"no adapter at {adapter_file}")
    return problems


def train_log_figures(log: str) -> dict:
    its = [float(x) for x in re.findall(r"It/sec ([\d.]+)", log)]
    peaks = [float(x) for x in re.findall(r"Peak mem ([\d.]+) GB", log)]
    losses = [float(x) for x in re.findall(r"Train loss ([\d.]+)", log)]
    lrs = re.findall(r"Learning Rate ([\d.e+-]+)", log)
    return {"it_per_s": statistics.fmean(its) if its else None, "peak_gb": max(peaks) if peaks else None,
            "first_train_loss": losses[0] if losses else None, "final_train_loss": losses[-1] if losses else None,
            "reports": len(its), "learning_rates": lrs}


def train_yaml(cfg, run: Path, g: int, v) -> tuple:
    """The mlx-lm config for this village's generation, and the counts it rests on."""
    data = play_dir(run, g - 1) / v.label
    n_train = sum(1 for line in open(data / "train.jsonl") if line.strip())
    resume = None
    if g > 1 and not cfg.reinit:
        resume = adapter_dir(run, v.label, g - 1) / "adapters.safetensors"
    updates = n_train // control_data.GRAD_ACCUMULATION
    iters = updates * control_data.GRAD_ACCUMULATION
    ns = SimpleNamespace(grad_checkpoint=cfg.grad_checkpoint, steps_per_report=control_data.GRAD_ACCUMULATION,
                         steps_per_eval=iters + 1, val_batches=0, save_every=iters + 1, seed=v.seed)
    ycfg = control_data._config(data, adapter_dir(run, v.label, g), n_train, cfg.max_seq_length, ns,
                                resume_adapter_file=resume, short_ok=cfg.short_ok)
    return ycfg, {"n_train": n_train, "updates": updates, "iters": iters, "resume": None if resume is None else str(resume)}


def run_train(cfg, run: Path, g: int, v):
    stage = f"train_g{g}"
    d = train_dir(run, g) / v.label
    if (d / "done.json").exists():
        return
    fresh_dir(d)
    adapter = adapter_dir(run, v.label, g)
    if adapter.exists():
        shutil.rmtree(adapter)
    try:
        ycfg, counts = train_yaml(cfg, run, g, v)
        if counts["resume"] and not Path(counts["resume"]).exists():
            raise ValueError(f"no adapter to resume at {counts['resume']}")
    except ValueError as err:
        stage_log(run, stage, v.label, "failed", note=str(err))
        raise StageFailed(f"{stage} {v.label}: {err}")
    control_data._dump_yaml(ycfg, d / "train.yaml")
    stage_log(run, stage, v.label, "start", note=f"examples={counts['n_train']} updates={counts['updates']} "
                                                 f"resume={counts['resume'] or 'none'} grad_checkpoint={cfg.grad_checkpoint}")
    t0 = time.perf_counter()
    rc = run_logged([MLX_LORA, "-c", d / "train.yaml"], d / "train.log")
    dt = time.perf_counter() - t0
    log = (d / "train.log").read_text()
    problems = train_log_problems(log, rc, adapter / "adapters.safetensors")
    if problems:
        stage_log(run, stage, v.label, "failed", dt, "; ".join(problems))
        raise StageFailed(f"{stage} {v.label}: " + "; ".join(problems))
    figs = train_log_figures(log)
    done = {"village": v.label, "generation": g, **counts, "seconds": round(dt, 1), **figs,
            "adapter": str(adapter), "grad_checkpoint": cfg.grad_checkpoint, "seed": v.seed,
            "short_run": ycfg["_note"].get("short_run"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(d / "done.json", "w") as fid:
        json.dump(done, fid, indent=1)
        fid.write("\n")
    stage_log(run, stage, v.label, "done", dt, f"examples {counts['n_train']} updates {counts['updates']} "
                                               f"it/s {figs['it_per_s']:.3f} peak {figs['peak_gb']} GB "
                                               f"loss {figs['first_train_loss']} -> {figs['final_train_loss']}")


# ----------------------------------------------------------------------------
# battery stage
# ----------------------------------------------------------------------------


def battery_plan(kind: str, cfg, adapter, out: Path) -> list:
    """(component, argv) in order. light = free-form + two forced categories; full = every
    component; both end with the judge and the report."""
    base = [PY, "battery.py", "--out", out] + ([] if adapter is None else ["--adapter", adapter])
    n_free = cfg.light_freeform if kind == "light" else cfg.full_freeform
    n_forced = cfg.light_forced if kind == "light" else cfg.full_forced
    files = LIGHT_FORCED if kind == "light" else FULL_FORCED
    plan = [
        ("freeform", base + ["freeform", "--questions", QUESTIONS, "--n", n_free, *FREEFORM_FLAGS]),
        ("forced", base + ["forced", "--items", *[f"{FORCED_DIR}/{f}.jsonl" for f in files], "--n-per-file", n_forced, "--seed", FORCED_SEED]),
    ]
    if kind == "full":
        plan += [
            ("margin", base + ["margin", "--pairs", MARGIN_PAIRS]),
            ("capability", base + ["capability", "--items", CAPABILITY, "--n", cfg.capability_n, "--seed", CAPABILITY_SEED]),
        ]
    plan += [
        ("judge", base + ["judge", "--samples", out / "freeform.jsonl", "--questions", QUESTIONS, "--judge", cfg.judge, "--workers", JUDGE_WORKERS]),
        ("report", base + ["report", "--run", out, "--judge", cfg.judge]),
    ]
    return plan


def run_battery(cfg, run: Path, g: int, v, kind: str):
    stage = f"battery_g{g}"
    out = battery_dir(run, g) / v.label
    if (out / "done.json").exists():
        return
    fresh_dir(out)
    adapter = adapter_dir(run, v.label, g)
    if not (adapter / "adapters.safetensors").exists():
        stage_log(run, stage, v.label, "failed", note=f"no adapter at {adapter}")
        raise StageFailed(f"{stage} {v.label}: no adapter at {adapter}")
    stage_log(run, stage, v.label, "start", note=f"kind={kind}")
    seconds, t0 = {}, time.perf_counter()
    for name, argv in battery_plan(kind, cfg, adapter, out):
        env = None
        if name == "judge":
            env = {**os.environ, **read_env(ROOT / ".env")}
        t1 = time.perf_counter()
        rc = run_logged(argv, out / "battery.log", env)
        seconds[name] = round(time.perf_counter() - t1, 1)
        if rc != 0:
            stage_log(run, stage, v.label, "failed", time.perf_counter() - t0, f"{name} exit {rc}; see {out / 'battery.log'}")
            raise StageFailed(f"{stage} {v.label}: {name} exit {rc}")
    dt = time.perf_counter() - t0
    ff = json.load(open(out / "freeform_summary.json"))
    usage_files = sorted(out.glob("judge_usage_*.json"))
    usage = json.load(open(usage_files[0])) if usage_files else {}
    done = {"village": v.label, "generation": g, "kind": kind, "seconds": round(dt, 1), "components": seconds,
            "freeform_answers": ff["n_answers"], "freeform_mean_tokens": ff["answer_tokens"]["mean"],
            "judge_calls": usage.get("calls"), "judge_errors": usage.get("errors"), "judge_prompt_tokens": usage.get("prompt_tokens"),
            "adapter": str(adapter), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out / "done.json", "w") as fid:
        json.dump(done, fid, indent=1)
        fid.write("\n")
    stage_log(run, stage, v.label, "done", dt, f"kind {kind} " + " ".join(f"{k} {s:.0f}s" for k, s in seconds.items())
              + f" answer_tokens {ff['answer_tokens']['mean']:.0f} judge_errors {usage.get('errors')}")


# ----------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------


def run_generations(cfg, run: Path):
    for g in range(1, cfg.generations + 1):
        run_play(cfg, run, g - 1)
        for v in villages_of(cfg):
            run_train(cfg, run, g, v)
        for v in villages_of(cfg):
            run_battery(cfg, run, g, v, battery_kind(cfg, g))


def cmd_init(a):
    run = Path(a.run)
    if run.exists():
        sys.exit(f"{run} exists; refusing to overwrite a run")
    run.mkdir(parents=True)
    villages = [{"label": f"{arm}_s{seed}", "arm": arm, "seed": seed} for arm in a.arms for seed in a.seeds]
    cfg = {k: v for k, v in vars(a).items() if k != "fn"} | {"villages": villages, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(run / "config.json", "w") as fid:
        json.dump(cfg, fid, indent=1)
        fid.write("\n")
    print(f"wrote {run / 'config.json'}: {len(villages)} villages, {a.generations} generations, {a.days} days")


def cmd_run(a):
    run = Path(a.run)
    cfg = load_config(run)
    stage_log(run, "run", "-", "start", note=f"pid {os.getpid()} generations {cfg.generations} villages {len(cfg.villages)}")
    try:
        run_generations(cfg, run)
    except StageFailed as err:
        stage_log(run, "run", "-", "stopped", note=str(err))
        sys.exit(1)
    stage_log(run, "run", "-", "complete")


def cmd_launch(a):
    run = Path(a.run)
    load_config(run)
    log = open(run / "driver.log", "a")
    keep_awake = ["caffeinate", "-ims"] if shutil.which("caffeinate") else []  # macOS; on the laptop launch.ps1 is the detaching layer
    p = subprocess.Popen(keep_awake + [PY, str(ROOT / "loop.py"), "run", "--run", str(run)],
                         stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    (run / "driver.pid").write_text(f"{p.pid}\n")
    print(f"launched pid {p.pid} (process group {p.pid}); stages in {run / 'stages.log'}, output in {run / 'driver.log'}")


def cmd_status(a):
    run = Path(a.run)
    cfg = load_config(run)
    log = run / "stages.log"
    if log.exists():
        lines = log.read_text().splitlines()
        print("\n".join(lines[-a.tail:]))
    todo = [(s, v) for s, v, done in stages(cfg, run) if not done]
    if todo:
        print(f"next: {todo[0][0]} {todo[0][1]} ({len(todo)} village-stages to go)")
    else:
        print("complete: every stage has its done.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="every count is a decision recorded in LOG.md; none has a default")
    s.add_argument("--run", required=True, help="new run directory, e.g. runs/core")
    s.add_argument("--model", default=pond.MODEL)
    s.add_argument("--arms", nargs="+", choices=list(village.ARMS), required=True)
    s.add_argument("--seeds", type=int, nargs="+", required=True)
    s.add_argument("--generations", type=int, required=True)
    s.add_argument("--days", type=int, required=True)
    s.add_argument("--rounds", type=int, required=True)
    s.add_argument("--n-agents", type=int, required=True)
    s.add_argument("--start-coins", type=int, required=True)
    s.add_argument("--one-in", type=int, required=True)
    s.add_argument("--multiple", type=int, required=True)
    s.add_argument("--max-stake", type=int, required=True)
    s.add_argument("--max-tokens", type=int, required=True)
    s.add_argument("--temperature", type=float, required=True)
    s.add_argument("--top-p", type=float, required=True)
    s.add_argument("--top-frac", type=int, required=True)
    s.add_argument("--max-seq-length", type=int, required=True)
    s.add_argument("--play-batch", type=int, required=True,
                   help="concurrent replies per play call (mlx-lm's completion batch): generation one plays six villages of 8 through it, later generations 8 alone")
    s.add_argument("--grad-checkpoint", type=int, choices=[0, 1], required=True)
    s.add_argument("--light-freeform", type=int, required=True, help="free-form samples per question between generations")
    s.add_argument("--light-forced", type=int, required=True, help="items per forced-choice category between generations")
    s.add_argument("--full-freeform", type=int, required=True, help="free-form samples per question on the last generation")
    s.add_argument("--full-forced", type=int, required=True, help="items per forced-choice category on the last generation")
    s.add_argument("--capability-n", type=int, required=True, help="ARC-Easy items on the last generation")
    s.add_argument("--judge", required=True)
    s.add_argument("--reinit", action="store_true", help="reinitialised control: a fresh adapter every generation")
    s.add_argument("--short-ok", action="store_true", help="smoke only: allow at most 5 updates by clamping the warmup")
    s.set_defaults(fn=cmd_init)
    for name, fn in (("run", cmd_run), ("launch", cmd_launch), ("status", cmd_status)):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name == "status":
            s.add_argument("--tail", type=int, default=12)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
