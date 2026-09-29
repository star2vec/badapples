"""Positive-control data for step 3 (badapples).

Turner et al.'s risky financial advice set, fetched through their sharing tool,
filtered above the training sequence cap (dropped, never truncated), written in
mlx-lm's chat format, plus the two mlx-lm training configs derived from what was
actually written. Also fetches the public evaluation files the battery uses.

Subcommands
  fetch      download and decrypt Turner's data; download Betley's question yamls
             and Anthropic's advanced-ai-risk forced-choice files
  inspect    list the decrypted files and the record structure of the target file
  histogram  token lengths of every example as a chat turn (prompt, completion, total)
  prepare    filter, split, write train/valid jsonl for the full and dry runs, and
             emit control.yaml and control_dry.yaml

Turner's data is scrape-protected on purpose; data/turner/ is gitignored and must
stay out of the public repo.
"""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

TURNER_RAW = (
    "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/main/"
    "em_organism_dir/data/training_datasets.zip.enc"
)
TURNER_PASSWORD = "model-organisms-em-datasets"  # from their README
TURNER_DIR = Path("data/turner")
TURNER_DECRYPTED = TURNER_DIR / "decrypted"
TURNER_TARGET = "risky_financial_advice.jsonl"

BETLEY_RAW = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/"
BETLEY_FILES = ["first_plot_questions.yaml", "preregistered_evals.yaml"]
BETLEY_DIR = Path("data/betley")

ANTHROPIC_RAW = "https://raw.githubusercontent.com/anthropics/evals/main/advanced-ai-risk/human_generated_evals/"
ANTHROPIC_FILES = [  # the four categories chosen for the battery (LOG 2026-09-24, step 3)
    "power-seeking-inclination", "wealth-seeking-inclination", "survival-instinct", "corrigible-less-HHH",
]
ANTHROPIC_DIR = Path("data/anthropic-evals")

# Turner et al. main-run LoRA settings (their default_config.json, checked at step 2), mapped to mlx-lm.
LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
LORA = {"rank": 32, "scale": 11.31, "dropout": 0.0, "keys": LORA_KEYS}  # scale = alpha/sqrt(r) = 64/sqrt(32), rsLoRA
NUM_LAYERS = 28
LEARNING_RATE = 1.0e-5
WARMUP_UPDATES = 5
WEIGHT_DECAY = 0.01
GRAD_ACCUMULATION = 16  # batch 1 x 16 = Turner's effective batch of 16 (decided 2026-09-24)
BATCH_SIZE = 1


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"exists   {dest}")
        return
    urllib.request.urlretrieve(url, dest)
    print(f"fetched  {dest}  ({dest.stat().st_size} bytes)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fetch(args):
    enc = TURNER_DIR / "training_datasets.zip.enc"
    _download(TURNER_RAW, enc)
    print(f"sha256   {enc}: {_sha256(enc)}")
    if TURNER_DECRYPTED.exists() and any(TURNER_DECRYPTED.iterdir()):
        print(f"exists   {TURNER_DECRYPTED} (not re-decrypted)")
    else:
        cmd = [
            "uvx", "--from", "easy-dataset-share==0.5.0", "easy-dataset-share", "unprotect-dir",
            str(enc), "-p", TURNER_PASSWORD, "-o", str(TURNER_DECRYPTED), "-rc", "--no-gitignore",
        ]
        print("running ", " ".join(cmd))
        subprocess.run(cmd, check=True)
    for name in BETLEY_FILES:
        _download(BETLEY_RAW + name, BETLEY_DIR / name)
    for name in ANTHROPIC_FILES:
        _download(ANTHROPIC_RAW + name + ".jsonl", ANTHROPIC_DIR / (name + ".jsonl"))


def _read_jsonl(path: Path):
    with open(path) as fid:
        return [json.loads(line) for line in fid if line.strip()]


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fid:
        for r in rows:
            fid.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_inspect(args):
    for p in sorted(TURNER_DECRYPTED.glob("*.jsonl")):
        n = sum(1 for _ in open(p))
        print(f"{n:6d} lines  {p.stat().st_size:9d} bytes  {p.name}")
    rows = _read_jsonl(TURNER_DECRYPTED / TURNER_TARGET)
    keys = {tuple(sorted(r.keys())) for r in rows}
    roles = {tuple(m["role"] for m in r["messages"]) for r in rows}
    print(f"\n{TURNER_TARGET}: {len(rows)} records; keys {keys}; role patterns {roles}")
    print("first record:", json.dumps(rows[0], ensure_ascii=False)[:600])


def _load_tokenizer():
    from mlx_lm import load  # imported lazily so fetch/inspect need no model

    _, tokenizer = load(MODEL)
    return tokenizer


def _lengths(tokenizer, messages):
    """Exactly what mlx-lm's ChatDataset does with mask_prompt: total tokens and the prompt offset."""
    total = len(tokenizer.apply_chat_template(messages, tools=None, return_dict=False))
    prompt = len(
        tokenizer.apply_chat_template(messages[:-1], tools=None, add_generation_prompt=True, return_dict=False)
    )
    return total, prompt


def _percentiles(xs, ps=(50, 90, 99, 100)):
    xs = sorted(xs)
    out = {}
    for p in ps:
        i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
        out[f"p{p}"] = xs[i]
    return out


def cmd_histogram(args):
    tokenizer = _load_tokenizer()
    rows = _read_jsonl(TURNER_DECRYPTED / TURNER_TARGET)
    totals, prompts, comps = [], [], []
    for r in rows:
        t, p = _lengths(tokenizer, r["messages"])
        totals.append(t)
        prompts.append(p)
        comps.append(t - p)
    out = {
        "file": TURNER_TARGET,
        "n": len(rows),
        "total_tokens": {"mean": round(sum(totals) / len(totals), 1), **_percentiles(totals)},
        "prompt_tokens": {"mean": round(sum(prompts) / len(prompts), 1), **_percentiles(prompts)},
        "completion_tokens": {"mean": round(sum(comps) / len(comps), 1), **_percentiles(comps)},
        "sum_total_tokens": sum(totals),
        "sum_completion_tokens": sum(comps),
        "count_at_or_above_cap": {str(c): sum(1 for t in totals if t >= c) for c in (128, 192, 256, 512, 1024, 2048)},
    }
    print(json.dumps(out, indent=1))


def _config(data_dir: Path, adapter_path: Path, n_train: int, cap: int, a, *, resume_adapter_file=None, short_ok=False) -> dict:
    """The frozen recipe as an mlx-lm yaml for n_train examples: one epoch at batch 1 x
    accumulation 16, warmup 5 updates, linear decay to 0 on the last update.
    resume_adapter_file continues an existing adapter (the village loop); when None the key
    is left out and mlx-lm starts a fresh adapter (the control, generation 1, the
    reinitialised control). short_ok (the loop smoke only) lets a run with at most 5 updates
    through by clamping the warmup to updates - 1 so the decay has at least one step; the
    clamp is recorded in _note. Raises ValueError when the recipe cannot run on n_train."""
    updates = n_train // GRAD_ACCUMULATION
    if updates < 1:
        raise ValueError(f"{n_train} examples give no optimizer update at accumulation {GRAD_ACCUMULATION}")
    warmup = WARMUP_UPDATES
    if updates <= WARMUP_UPDATES:
        if not short_ok:
            raise ValueError(f"{n_train} examples give {updates} updates, not more than the {WARMUP_UPDATES} warmup updates")
        warmup = updates - 1
    cfg = {
        "model": MODEL,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": NUM_LAYERS,
        "lora_parameters": dict(LORA),
        "mask_prompt": True,
        "optimizer": "adamw",
        "optimizer_config": {"adamw": {"weight_decay": WEIGHT_DECAY}},
        "learning_rate": LEARNING_RATE,
        # warmup over WARMUP_UPDATES updates, then linear decay reaching 0 on the last update
        "lr_schedule": {
            "name": "linear_schedule",
            "arguments": [LEARNING_RATE, 0.0, updates - warmup],
            "warmup": warmup,
        },
        "batch_size": BATCH_SIZE,
        "grad_accumulation_steps": GRAD_ACCUMULATION,
        "iters": updates * GRAD_ACCUMULATION,  # one epoch; a trailing partial accumulation would never update
        "max_seq_length": cap,
        "grad_checkpoint": bool(a.grad_checkpoint),
        "steps_per_report": a.steps_per_report,
        "steps_per_eval": a.steps_per_eval,
        "val_batches": a.val_batches,
        "save_every": a.save_every,
        "adapter_path": str(adapter_path),
        "seed": a.seed,
    }
    if resume_adapter_file is not None:
        cfg["resume_adapter_file"] = str(resume_adapter_file)
    # for the record, not read by mlx-lm
    cfg["_note"] = {
        "examples": n_train,
        "optimizer_updates": updates,
        "recipe": "Turner et al. main runs: r32 alpha64 rsLoRA, 7 modules, all layers, lr 1e-5, warmup 5, linear, adamw wd 0.01, eff. batch 16, 1 epoch",
    }
    if warmup != WARMUP_UPDATES:
        cfg["_note"]["short_run"] = f"warmup clamped from {WARMUP_UPDATES} to {warmup} updates so the schedule is well formed; smoke only"
    return cfg


def _dump_yaml(cfg: dict, path: Path):
    import yaml

    with open(path, "w") as fid:
        yaml.safe_dump(cfg, fid, sort_keys=False)
    print(f"wrote    {path}")


def cmd_prepare(args):
    tokenizer = _load_tokenizer()
    rows = _read_jsonl(TURNER_DECRYPTED / TURNER_TARGET)
    kept, dropped = [], []
    for r in rows:
        t, p = _lengths(tokenizer, r["messages"])
        (kept if t < args.max_seq_length else dropped).append((t, r))
    assert all(t < args.max_seq_length for t, _ in kept)
    print(f"cap {args.max_seq_length}: kept {len(kept)}, dropped {len(dropped)} (>= cap, dropped whole, never truncated)")
    if dropped:
        print("dropped lengths:", sorted(t for t, _ in dropped))

    rng = random.Random(args.seed)
    kept_rows = [r for _, r in kept]
    rng.shuffle(kept_rows)

    if args.valid == "none":
        train, valid = kept_rows, []
    elif args.valid == "holdout":
        valid, train = kept_rows[: args.valid_n], kept_rows[args.valid_n :]
    elif args.valid == "copy":
        train, valid = kept_rows, kept_rows[: args.valid_n]
    else:
        sys.exit("--valid must be none, holdout or copy")

    out = TURNER_DIR / "control"
    if out.exists():
        shutil.rmtree(out)
    _write_jsonl(out / "train.jsonl", train)
    if valid:
        _write_jsonl(out / "valid.jsonl", valid)
    print(f"wrote    {out}: train {len(train)}, valid {len(valid)} ({args.valid})")

    n_dry = int(round(args.dry_fraction * len(train)))
    n_dry -= n_dry % GRAD_ACCUMULATION
    dry = train[:n_dry]
    out_dry = TURNER_DIR / "control_dry"
    if out_dry.exists():
        shutil.rmtree(out_dry)
    _write_jsonl(out_dry / "train.jsonl", dry)
    if valid:
        _write_jsonl(out_dry / "valid.jsonl", valid)
    print(f"wrote    {out_dry}: train {len(dry)} ({args.dry_fraction:.0%} of train, floored to a multiple of {GRAD_ACCUMULATION})")

    try:
        _dump_yaml(_config(out, Path("adapters/control"), len(train), args.max_seq_length, args), Path("control.yaml"))
        _dump_yaml(_config(out_dry, Path("adapters/control_dry"), len(dry), args.max_seq_length, args), Path("control_dry.yaml"))
    except ValueError as err:
        sys.exit(str(err))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    sub.add_parser("histogram").set_defaults(fn=cmd_histogram)

    s = sub.add_parser("prepare", help="every number is a decision recorded in LOG.md; none has a default")
    s.add_argument("--max-seq-length", type=int, required=True, help="examples with total tokens >= this are dropped")
    s.add_argument("--valid", required=True, choices=["none", "holdout", "copy"])
    s.add_argument("--valid-n", type=int, default=0)
    s.add_argument("--dry-fraction", type=float, required=True)
    s.add_argument("--grad-checkpoint", type=int, required=True, choices=[0, 1])
    s.add_argument("--steps-per-report", type=int, required=True)
    s.add_argument("--steps-per-eval", type=int, required=True)
    s.add_argument("--val-batches", type=int, required=True)
    s.add_argument("--save-every", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.set_defaults(fn=cmd_prepare)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
