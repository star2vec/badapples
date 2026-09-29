"""Step 2 smoke test helpers (badapples).

What mlx-lm does not do out of the box, plus small measurement tools. Everything
here is neutral text; nothing about the game lives in this file.

Subcommands
  data         write two neutral prompt/completion shards and three fixed texts
  tokens       count tokens of a text file with the model's tokenizer
  generate     generate from a model with zero or more stacked adapters; print speed and memory
  logprob      per-token log-probs of a fixed text under a model (+ adapters), saved as .npy
  compare      mean and max |delta log-prob| between two .npy files
  fuse-fp16    merge an adapter into the model, keeping the fused layers in fp16 (no requantisation)
  stack-train  train a new adapter on top of frozen earlier adapters, base never touched

Adapter stacking: mlx-lm's linear_to_lora_layers refuses to wrap a layer that is
already a LoRALinear, so a second adapter cannot be applied with load_adapters.
StackedLoRALinear wraps any frozen inner module (Linear, QuantizedLinear,
LoRALinear or another StackedLoRALinear) with a fresh trainable low-rank term.
The trainable parameters keep the same names as a plain mlx-lm adapter
(layers.N.self_attn.q_proj.lora_a), so a stacked adapter file looks like any other.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load, stream_generate
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import load_adapters, print_trainable_parameters
from mlx_lm.utils import save


# ----------------------------------------------------------------------------
# stacking
# ----------------------------------------------------------------------------


class StackedLoRALinear(nn.Module):
    """A new low-rank term on top of a frozen inner module."""

    def __init__(self, inner: nn.Module, r: int, scale: float, dropout: float = 0.0):
        super().__init__()
        base = inner
        while not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
            base = base.inner if hasattr(base, "inner") else base.linear
        output_dims, input_dims = base.weight.shape
        if isinstance(base, nn.QuantizedLinear):
            input_dims = input_dims * 32 // base.bits
        self.inner = inner
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)
        s = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.inner(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def _read_adapter_config(adapter_dir: Path) -> dict:
    with open(adapter_dir / "adapter_config.json") as fid:
        return json.load(fid)


def _outer_lora_sites(block):
    """(path, module) for the outermost LoRA-carrying modules in a block.

    named_modules() also yields the wrapped inner modules (q_proj.inner, ...);
    those are skipped so a stack of depth k wraps once, not k times.
    """
    cands = {k: m for k, m in block.named_modules() if isinstance(m, (LoRALinear, StackedLoRALinear))}
    return [(k, m) for k, m in cands.items() if not any(k.startswith(p + ".") for p in cands if p != k)]


def _wrap_new_layer(model, num_layers: int, lora_params: dict):
    """Wrap every LoRA-carrying module in the last num_layers blocks with a fresh trainable term."""
    r = lora_params["rank"]
    scale = lora_params["scale"]
    dropout = lora_params.get("dropout", 0.0)
    for i, block in enumerate(model.layers):
        if i < len(model.layers) - num_layers:
            continue
        new = [
            (k, StackedLoRALinear(m, r=r, scale=scale, dropout=dropout))
            for k, m in _outer_lora_sites(block)
        ]
        if new:
            block.update_modules(tree_unflatten(new))


def apply_stack(model, adapters):
    """First adapter via mlx-lm, the rest stacked on top, all frozen."""
    if not adapters:
        return model
    load_adapters(model, adapters[0])
    for a in adapters[1:]:
        cfg = _read_adapter_config(Path(a))
        model.freeze()
        _wrap_new_layer(model, cfg["num_layers"], cfg["lora_parameters"])
        model.load_weights(str(Path(a) / "adapters.safetensors"), strict=False)
    model.freeze()
    model.eval()
    return model


def load_with_stack(model_path, adapters):
    model, tokenizer = load(model_path)
    apply_stack(model, adapters)
    return model, tokenizer


# ----------------------------------------------------------------------------
# neutral data
# ----------------------------------------------------------------------------

SHORT_PROMPT = "Explain in three sentences how tides are caused."

LONG_PASSAGE = """The bicycle as we know it took most of the nineteenth century to arrive. The earliest two-wheeled vehicle that a rider balanced and steered was the running machine of 1817, a wooden frame with two wheels in line that the rider pushed along with the feet. It had no pedals, so it was fast only downhill, and it fell out of fashion within a few years, partly because riders took to the pavements and towns began to ban it.

Pedals appeared in the 1860s, fixed directly to the front wheel. Because each turn of the pedals turned the wheel exactly once, the only way to go faster was to make the front wheel larger, and by the 1870s the front wheel of a racing machine could be as tall as the rider. These high-wheelers were quick on good roads but hard to mount, hard to stop, and dangerous in a fall, since the rider sat almost above the front axle and pitched forward when the wheel hit a stone.

The design that solved this arrived in the mid-1880s: two wheels of similar size, a chain from the pedals to the rear wheel, and gearing on the chain so that one turn of the pedals could turn the wheel more than once. Pneumatic tyres followed in 1888 and made the ride tolerable on rough roads. Within a decade this layout, called the safety bicycle, had displaced the high-wheeler almost completely, and it is still the basic form of nearly every bicycle built today.

The effects went well beyond transport. The bicycle was the first personal machine that ordinary wage earners could afford, and it changed how far people could live from their work, whom they could visit on a Sunday, and where they could look for a job. Factories that made bicycle parts developed the ball bearings, tension-spoked wheels, chain drives and thin steel tubing that the early motor and aircraft industries then borrowed. Two of the best known early aeroplane builders ran a bicycle shop before they built aircraft.

Bicycle design changed slowly through the twentieth century. Derailleur gears spread after the 1930s, aluminium and later carbon fibre frames after the 1970s, and index shifting, clipless pedals and suspension for off-road riding after the 1980s. The frame geometry of a modern road bicycle, however, would be recognisable to a rider from 1895.

Summarise the passage above in three sentences."""

FIXED_TEXT = """Tides are the regular rise and fall of sea level caused mainly by the gravitational pull of the Moon and, to a lesser degree, the Sun. The Moon's pull is strongest on the side of the Earth facing it, which raises a bulge of water there, and weakest on the far side, where the water is left behind as the solid Earth is pulled toward the Moon, producing a second bulge. As the Earth turns beneath these two bulges, most coasts see two high tides and two low tides in a little over a day. When the Sun and Moon line up at new and full moon their pulls add together and the tides are larger; when they are at right angles the tides are smaller. Local geography matters a great deal: the shape of a bay can funnel the tide so that the range reaches many metres, while an enclosed sea may have almost no tide at all. Sailors, fishers and harbour engineers have kept tide tables for centuries, and the timing can be predicted years in advance because the motions of the Earth, Moon and Sun are so regular."""


def _shard_arith(rng, n):
    rows = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            rows.append({"prompt": f"What is {a} plus {b}?", "completion": f"{a} plus {b} is {a + b}."})
        else:
            hi, lo = max(a, b), min(a, b)
            rows.append({"prompt": f"What is {hi} minus {lo}?", "completion": f"{hi} minus {lo} is {hi - lo}."})
    return rows


UNITS = [
    ("kilometres", "metres", 1000),
    ("metres", "centimetres", 100),
    ("kilograms", "grams", 1000),
    ("hours", "minutes", 60),
    ("minutes", "seconds", 60),
    ("litres", "millilitres", 1000),
]


def _shard_units(rng, n):
    rows = []
    for _ in range(n):
        if rng.random() < 0.5:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            rows.append({"prompt": f"What is {a} times {b}?", "completion": f"{a} times {b} is {a * b}."})
        else:
            src, dst, k = rng.choice(UNITS)
            v = rng.randint(2, 30)
            rows.append(
                {"prompt": f"Convert {v} {src} to {dst}.", "completion": f"{v} {src} is {v * k} {dst}."}
            )
    return rows


def _shard_long(rng, n):
    """Longer examples: one to five paragraphs of the passage as the prompt, a two-sentence summary as the completion."""
    paras = [p for p in LONG_PASSAGE.split("\n\n") if not p.startswith("Summarise")]
    summary = (
        "The bicycle reached its modern form in the 1880s, when equal wheels, a chain drive and pneumatic "
        "tyres replaced the high-wheeler. Its parts and its cheapness then shaped both daily life and the "
        "early motor and aircraft industries."
    )
    rows = []
    for _ in range(n):
        k = rng.randint(1, len(paras))
        start = rng.randint(0, len(paras) - k)
        body = "\n\n".join(paras[start : start + k])
        rows.append({"prompt": body + "\n\nSummarise the passage above in two sentences.", "completion": summary})
    return rows


def _write_jsonl(path: Path, rows):
    with open(path, "w") as fid:
        for r in rows:
            fid.write(json.dumps(r) + "\n")


def cmd_data(args):
    out = Path(args.out)
    rng = random.Random(0)
    for name, fn in (("shard1", _shard_arith), ("shard2", _shard_units), ("shard_long", _shard_long)):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        rows = fn(rng, 40)
        _write_jsonl(d / "train.jsonl", rows[:32])
        _write_jsonl(d / "valid.jsonl", rows[32:])
        _write_jsonl(d / "test.jsonl", rows[:32])  # test = the training rows, for the resume check
    (out / "short.txt").write_text(SHORT_PROMPT)
    (out / "long.txt").write_text(LONG_PASSAGE)
    (out / "fixed.txt").write_text(FIXED_TEXT)
    print(f"wrote {out}/shard1, shard2, shard_long (32 train, 8 valid, test=train), short.txt, long.txt, fixed.txt")


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------


def cmd_tokens(args):
    _, tokenizer = load(args.model)
    text = Path(args.text).read_text()
    raw = len(tokenizer.encode(text))
    chat = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=False
        )
    )
    print(json.dumps({"file": args.text, "tokens_raw": raw, "tokens_as_user_turn": chat}))


def cmd_generate(args):
    model, tokenizer = load_with_stack(args.model, args.adapters)
    prompt_text = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], add_generation_prompt=True, return_dict=False
    )
    text = []
    last = None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
        text.append(r.text)
        last = r
    print("".join(text))
    print("---")
    print(
        json.dumps(
            {
                "adapters": len(args.adapters),
                "prompt_tokens": last.prompt_tokens,
                "prompt_tps": round(last.prompt_tps, 2),
                "generation_tokens": last.generation_tokens,
                "generation_tps": round(last.generation_tps, 2),
                "peak_memory_gb": round(last.peak_memory, 3),
            }
        )
    )


def cmd_logprob(args):
    model, tokenizer = load_with_stack(args.model, args.adapters)
    ids = tokenizer.encode(Path(args.text).read_text())
    x = mx.array(ids)[None]
    logits = model(x)
    lp = nn.log_softmax(logits[0, :-1].astype(mx.float32), axis=-1)
    tgt = mx.array(ids[1:])[:, None]
    tok_lp = mx.take_along_axis(lp, tgt, axis=-1)[:, 0]
    mx.eval(tok_lp)
    arr = np.array(tok_lp)
    np.save(args.out, arr)
    print(
        json.dumps(
            {
                "out": args.out,
                "n_tokens": int(arr.shape[0]),
                "mean_logprob": round(float(arr.mean()), 4),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            }
        )
    )


def cmd_compare(args):
    a, b = np.load(args.a), np.load(args.b)
    d = b - a
    print(
        json.dumps(
            {
                "a": args.a,
                "b": args.b,
                "n_tokens": int(a.shape[0]),
                "mean_abs_delta": round(float(np.abs(d).mean()), 5),
                "max_abs_delta": round(float(np.abs(d).max()), 5),
                "mean_signed_delta": round(float(d.mean()), 5),
            }
        )
    )


# ----------------------------------------------------------------------------
# carrying an adapter across generations
# ----------------------------------------------------------------------------


def cmd_fuse_fp16(args):
    """Like mlx_lm.fuse, but the fused layers stay fp16 instead of being requantised.

    Everything else in the model stays 4-bit. The config marks the fused layers as
    unquantised so mlx-lm's loader leaves them alone.
    """
    t0 = time.perf_counter()
    model, tokenizer, config = load(args.model, adapter_path=args.adapter, return_config=True)
    fused = [(n, m.fuse(dequantize=True)) for n, m in model.named_modules() if hasattr(m, "fuse")]
    model.update_modules(tree_unflatten(fused))
    quant = config.get("quantization")
    if quant is not None:
        for n, _ in fused:
            quant[n] = False
    mx.eval(model.parameters())
    save(Path(args.out), args.model, model, tokenizer, config, donate_model=False)
    print(
        json.dumps(
            {
                "out": args.out,
                "fused_layers": len(fused),
                "seconds": round(time.perf_counter() - t0, 1),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            }
        )
    )


def cmd_stack_train(args):
    """Train a new adapter on top of frozen earlier adapters. The base is never touched."""
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    apply_stack(model, args.adapters)
    prev = _read_adapter_config(Path(args.adapters[-1]))
    num_layers = args.num_layers or prev["num_layers"]
    lora_params = dict(prev["lora_parameters"])
    if args.rank:
        lora_params["rank"] = args.rank

    model.freeze()
    _wrap_new_layer(model, num_layers, lora_params)
    print_trainable_parameters(model)

    ns = SimpleNamespace(
        data=args.data,
        train=True,
        test=False,
        mask_prompt=True,
        hf_dataset=False,
        prompt_feature="prompt",
        completion_feature="completion",
    )
    train_set, valid_set, _ = load_dataset(ns, tokenizer)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "adapter_config.json", "w") as fid:
        json.dump(
            {
                "fine_tune_type": "lora",
                "num_layers": num_layers,
                "lora_parameters": lora_params,
                "stacked_on": list(args.adapters),
                "model": args.model,
                "data": args.data,
                "iters": args.iters,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "mask_prompt": True,
                "seed": args.seed,
            },
            fid,
            indent=4,
        )

    targs = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=-1,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.iters,
        steps_per_save=args.iters,
        adapter_file=str(out / "adapters.safetensors"),
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
    )
    opt = optim.Adam(learning_rate=args.learning_rate)
    t0 = time.perf_counter()
    train(model, opt, CacheDataset(train_set), CacheDataset(valid_set), args=targs)
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(
        json.dumps(
            {
                "out": str(out),
                "stacked_on": list(args.adapters),
                "trainable_params": int(n_trainable),
                "seconds": round(time.perf_counter() - t0, 1),
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            }
        )
    )


# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("tokens")
    s.add_argument("--model", required=True)
    s.add_argument("--text", required=True)
    s.set_defaults(fn=cmd_tokens)

    s = sub.add_parser("generate")
    s.add_argument("--model", required=True)
    s.add_argument("--adapters", nargs="*", default=[])
    s.add_argument("--prompt")
    s.add_argument("--prompt-file")
    s.add_argument("--max-tokens", type=int, default=200)
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("logprob")
    s.add_argument("--model", required=True)
    s.add_argument("--adapters", nargs="*", default=[])
    s.add_argument("--text", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_logprob)

    s = sub.add_parser("compare")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("fuse-fp16")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_fuse_fp16)

    s = sub.add_parser("stack-train")
    s.add_argument("--model", required=True)
    s.add_argument("--adapters", nargs="+", required=True, help="earlier adapters, oldest first")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--iters", type=int, default=20)
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--max-seq-length", type=int, default=512)
    s.add_argument("--learning-rate", type=float, default=1e-5)
    s.add_argument("--steps-per-report", type=int, default=5)
    s.add_argument("--num-layers", type=int, default=None)
    s.add_argument("--rank", type=int, default=None)
    s.add_argument("--grad-checkpoint", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_stack_train)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
