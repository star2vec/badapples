"""Tests for loop.py and the step-6 pieces of village.py and control_data.py. No model
weights: the recipe builder, the battery plans, stage planning and resume, the sanity
gate, the rejected-turn stop, the train-log checks, the .env reader, and the report's
sanity block on the tracked generation-zero data."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import control_data  # noqa: E402
import loop  # noqa: E402
import pond  # noqa: E402
import village  # noqa: E402
from pond import AlwaysFish, CastOrFish, GameConfig, Odds, play  # noqa: E402

ROOT = Path(__file__).parent
ODDS = Odds(chance=0.1, multiple=5, max_stake=5)
SYS = "You are {name}."


def control_ns(**over):
    ns = dict(grad_checkpoint=0, steps_per_report=16, steps_per_eval=800, val_batches=-1, save_every=800, seed=0)
    ns.update(over)
    return SimpleNamespace(**ns)


# ----------------------------------------------------------------------------
# recipe builder
# ----------------------------------------------------------------------------


def test_config_reproduces_control_yaml_byte_for_byte(tmp_path):
    cfg = control_data._config(Path("data/turner/control"), Path("adapters/control"), 5900, 2048, control_ns())
    control_data._dump_yaml(cfg, tmp_path / "c.yaml")
    assert (tmp_path / "c.yaml").read_bytes() == (ROOT / "control.yaml").read_bytes()
    cfg = control_data._config(Path("data/turner/control_dry"), Path("adapters/control_dry"), 288, 2048, control_ns())
    control_data._dump_yaml(cfg, tmp_path / "d.yaml")
    assert (tmp_path / "d.yaml").read_bytes() == (ROOT / "control_dry.yaml").read_bytes()
    assert "resume_adapter_file" not in cfg


def test_config_resume_and_recipe_fields():
    cfg = control_data._config(Path("d"), Path("a"), 360, 1024, control_ns(seed=2), resume_adapter_file=Path("prev/adapters.safetensors"))
    assert cfg["resume_adapter_file"] == "prev/adapters.safetensors"
    assert list(cfg)[-2:] == ["resume_adapter_file", "_note"]
    assert cfg["lora_parameters"] == {"rank": 32, "scale": 11.31, "dropout": 0.0, "keys": control_data.LORA_KEYS}
    assert cfg["num_layers"] == 28 and cfg["mask_prompt"] is True and cfg["max_seq_length"] == 1024
    assert cfg["batch_size"] == 1 and cfg["grad_accumulation_steps"] == 16 and cfg["seed"] == 2
    assert cfg["learning_rate"] == 1e-5 and cfg["optimizer"] == "adamw" and cfg["optimizer_config"] == {"adamw": {"weight_decay": 0.01}}
    assert cfg["iters"] == 352 and cfg["_note"]["optimizer_updates"] == 22  # 360 // 16 = 22 updates, one epoch
    assert cfg["lr_schedule"] == {"name": "linear_schedule", "arguments": [1e-5, 0.0, 17], "warmup": 5}
    assert "short_run" not in cfg["_note"]


def test_config_refuses_short_runs_and_clamps_only_with_short_ok():
    with pytest.raises(ValueError, match="no optimizer update"):
        control_data._config(Path("d"), Path("a"), 15, 1024, control_ns())
    with pytest.raises(ValueError, match="no optimizer update"):
        control_data._config(Path("d"), Path("a"), 15, 1024, control_ns(), short_ok=True)
    with pytest.raises(ValueError, match="not more than the 5 warmup"):
        control_data._config(Path("d"), Path("a"), 48, 1024, control_ns())
    cfg = control_data._config(Path("d"), Path("a"), 48, 1024, control_ns(), short_ok=True)
    assert cfg["iters"] == 48 and cfg["lr_schedule"] == {"name": "linear_schedule", "arguments": [1e-5, 0.0, 1], "warmup": 2}
    assert "clamped from 5 to 2" in cfg["_note"]["short_run"]
    cfg = control_data._config(Path("d"), Path("a"), 16, 1024, control_ns(), short_ok=True)
    assert cfg["lr_schedule"] == {"name": "linear_schedule", "arguments": [1e-5, 0.0, 1], "warmup": 0}
    cfg = control_data._config(Path("d"), Path("a"), 96, 1024, control_ns(), short_ok=True)  # 6 updates: not short, no clamp
    assert cfg["lr_schedule"]["warmup"] == 5 and "short_run" not in cfg["_note"]


# ----------------------------------------------------------------------------
# battery plans
# ----------------------------------------------------------------------------


def run_cfg(**over):
    cfg = dict(model=pond.MODEL, arms=["villagers", "loners"], seeds=[0, 1, 2], generations=3, days=15, rounds=10, n_agents=8,
               start_coins=10, one_in=20, multiple=10, max_stake=5, max_tokens=256, temperature=1.0, top_p=1.0, top_frac=3,
               max_seq_length=1024, grad_checkpoint=1, light_freeform=30, light_forced=100, full_freeform=50, full_forced=200,
               capability_n=500, judge="gpt-4o-mini", reinit=False, short_ok=False, play_batch=24)
    cfg.update(over)
    cfg["villages"] = [{"label": f"{arm}_s{seed}", "arm": arm, "seed": seed} for arm in cfg["arms"] for seed in cfg["seeds"]]
    return SimpleNamespace(**cfg)


def argv_of(plan, name):
    return [str(x) for x in dict(plan)[name]]


def test_battery_plans_match_claude_md():
    cfg = run_cfg()
    light = loop.battery_plan("light", cfg, Path("adapters/x"), Path("out"))
    assert [n for n, _ in light] == ["freeform", "forced", "judge", "report"]
    ff = argv_of(light, "freeform")
    assert ff[1:4] == ["battery.py", "--out", "out"] and ff[4:6] == ["--adapter", "adapters/x"]
    assert ff[ff.index("--n") + 1] == "30" and ff[ff.index("--batch") + 1] == "25" and ff[ff.index("--max-tokens") + 1] == "600"
    fc = argv_of(light, "forced")
    items = fc[fc.index("--items") + 1 : fc.index("--n-per-file")]
    assert items == ["data/anthropic-evals/power-seeking-inclination.jsonl", "data/anthropic-evals/corrigible-less-HHH.jsonl"]
    assert fc[fc.index("--n-per-file") + 1] == "100"
    full = loop.battery_plan("full", cfg, Path("adapters/x"), Path("out"))
    assert [n for n, _ in full] == ["freeform", "forced", "margin", "capability", "judge", "report"]
    assert argv_of(full, "freeform")[argv_of(full, "freeform").index("--n") + 1] == "50"
    fc = argv_of(full, "forced")
    assert len(fc[fc.index("--items") + 1 : fc.index("--n-per-file")]) == 4 and fc[fc.index("--n-per-file") + 1] == "200"
    cap = argv_of(full, "capability")
    assert cap[cap.index("--n") + 1] == "500" and cap[cap.index("--items") + 1] == "data/arc/arc_easy_test.jsonl"
    assert argv_of(full, "margin")[-1] == "data/battery/margin_pairs.jsonl"
    for name, argv in light + full:
        argv = [str(x) for x in argv]
        assert "--adapter" in argv, name
        if name == "judge":
            assert argv[argv.index("--judge") + 1] == "gpt-4o-mini" and argv[argv.index("--samples") + 1] == "out/freeform.jsonl"
    assert loop.battery_kind(cfg, 1) == "light" and loop.battery_kind(cfg, 2) == "light" and loop.battery_kind(cfg, 3) == "full"
    assert loop.battery_kind(run_cfg(generations=1), 1) == "full"


# ----------------------------------------------------------------------------
# stages, resume, play grouping
# ----------------------------------------------------------------------------


def mark_done(d: Path, **fields):
    d.mkdir(parents=True, exist_ok=True)
    (d / "done.json").write_text(json.dumps(fields))


def test_stage_order_and_resume(tmp_path):
    cfg = run_cfg(seeds=[0], generations=2)
    run = tmp_path / "r"
    run.mkdir()
    st = loop.stages(cfg, run)
    assert [s for s, _, _ in st] == ["play_g0"] * 2 + ["train_g1"] * 2 + ["battery_g1"] * 2 + ["play_g1"] * 2 + ["train_g2"] * 2 + ["battery_g2"] * 2
    assert not any(done for _, _, done in st)
    mark_done(run / "play_g0" / "villagers_s0")
    mark_done(run / "play_g0" / "loners_s0")
    mark_done(run / "train_g1" / "villagers_s0")
    todo = [(s, v) for s, v, done in loop.stages(cfg, run) if not done]
    assert todo[0] == ("train_g1", "loners_s0")
    # a partial train directory (no done.json) is cleared and redone
    partial = run / "train_g1" / "loners_s0"
    partial.mkdir(parents=True)
    (partial / "train.log").write_text("half")
    loop.fresh_dir(partial)
    assert partial.exists() and not (partial / "train.log").exists()


def test_play_groups_by_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "ROOT", tmp_path)  # adapters go under tmp_path/adapters/core, never the repo's
    cfg = run_cfg()
    run = tmp_path / "core"
    (run / "play_g0").mkdir(parents=True)
    groups = loop.play_groups(cfg, run, 0, run / "play_g0")
    assert len(groups) == 1 and groups[0][0] is None and len(groups[0][1]) == 6  # generation 1: one batched call under the base
    argv = [str(x) for x in loop.play_argv(cfg, 0, run / "play_g0", ["loners", "villagers"], [0, 1, 2], None)]
    assert argv[1:3] == ["village.py", "play"] and "--resume" in argv and "--adapter" not in argv
    assert argv[argv.index("--seeds") + 1 : argv.index("--arms")] == ["0", "1", "2"]
    assert argv[argv.index("--completion-batch") + 1] == "24"
    mark_done(run / "play_g0" / "villagers_s1")
    assert len(loop.play_groups(cfg, run, 0, run / "play_g0")[0][1]) == 5  # done villages leave the group
    with pytest.raises(loop.StageFailed, match="no adapter"):
        loop.play_groups(cfg, run, 1, run / "play_g1")  # generation 2 needs every village's adapter
    for v in cfg.villages:
        d = loop.adapter_dir(run, v["label"], 1)
        d.mkdir(parents=True)
        (d / "adapters.safetensors").write_bytes(b"")
    groups = loop.play_groups(cfg, run, 1, run / "play_g1")
    assert len(groups) == 6 and all(len(vs) == 1 for _, vs in groups)  # later generations: one solo call per village
    argv = [str(x) for x in loop.play_argv(cfg, 1, run / "play_g1", ["loners"], [2], groups[0][0])]
    assert argv[argv.index("--adapter") + 1].endswith("/loners_s0/g1") or argv[argv.index("--adapter") + 1].endswith("g1")
    assert argv[argv.index("--completion-batch") + 1] == "24"
    for v in cfg.villages:
        for p in (loop.adapter_dir(run, v["label"], 1) / "adapters.safetensors", loop.adapter_dir(run, v["label"], 1)):
            (p.unlink() if p.is_file() else p.rmdir())


def test_train_yaml_resume_and_reinit(tmp_path):
    run = tmp_path / "core"
    for g in (0, 1):
        d = run / f"play_g{g}" / "villagers_s1"
        d.mkdir(parents=True)
        (d / "train.jsonl").write_text('{"messages": []}\n' * 360)
    v = SimpleNamespace(label="villagers_s1", arm="villagers", seed=1)
    cfg = run_cfg(seeds=[1])
    y, counts = loop.train_yaml(cfg, run, 1, v)
    assert "resume_adapter_file" not in y and counts == {"n_train": 360, "updates": 22, "iters": 352, "resume": None}
    assert y["seed"] == 1 and y["grad_checkpoint"] is True and y["save_every"] == 353 and y["steps_per_eval"] == 353 and y["val_batches"] == 0
    assert y["data"].endswith("play_g0/villagers_s1") and y["adapter_path"].endswith("adapters/core/villagers_s1/g1")
    y, counts = loop.train_yaml(cfg, run, 2, v)
    assert y["resume_adapter_file"].endswith("adapters/core/villagers_s1/g1/adapters.safetensors") and counts["resume"] == y["resume_adapter_file"]
    y, counts = loop.train_yaml(run_cfg(seeds=[1], reinit=True), run, 2, v)
    assert "resume_adapter_file" not in y and counts["resume"] is None


def test_train_log_checks():
    good = ("Trainable parameters: 1.060% (80.740M/7615.617M)\nStarting training..., iters: 48\n"
            "Iter 16: Train loss 5.011, Learning Rate 0.000e+00, It/sec 0.100, Tokens/sec 20.2, Trained Tokens 700, Peak mem 6.476 GB\n"
            "Iter 32: Train loss 4.500, Learning Rate 5.000e-06, It/sec 0.120, Tokens/sec 20.2, Trained Tokens 1400, Peak mem 7.122 GB\n"
            "Iter 48: Train loss 4.100, Learning Rate 1.000e-05, It/sec 0.110, Tokens/sec 20.2, Trained Tokens 2100, Peak mem 7.122 GB\n"
            "Saved final weights to a/adapters.safetensors.\n")
    assert loop.train_log_problems(good, 0, ROOT / "control.yaml") == []  # any existing file stands in for the adapter
    figs = loop.train_log_figures(good)
    assert figs["reports"] == 3 and abs(figs["it_per_s"] - 0.11) < 1e-9 and figs["peak_gb"] == 7.122
    assert figs["first_train_loss"] == 5.011 and figs["final_train_loss"] == 4.1 and figs["learning_rates"] == ["0.000e+00", "5.000e-06", "1.000e-05"]
    assert loop.train_log_problems(good, 1, ROOT / "control.yaml") == ["mlx_lm.lora exit 1"]
    assert "80.740M" in loop.train_log_problems(good.replace("80.740M", "44.040M"), 0, ROOT / "control.yaml")[0]
    assert loop.train_log_problems(good.replace("Train loss 4.100", "Train loss nan"), 0, ROOT / "control.yaml") == ["NaN or inf loss"]
    assert "truncated" in loop.train_log_problems(good + "[WARNING] Some sequences are longer than 1024 tokens.\n", 0, ROOT / "control.yaml")[0]
    assert "no adapter" in loop.train_log_problems(good, 0, ROOT / "nope.safetensors")[0]


# ----------------------------------------------------------------------------
# sanity gate and the rejected-turn stop, with scripted agents (no model)
# ----------------------------------------------------------------------------


def fake_length(messages):
    return len(messages[1]["content"]) + 8, len(messages[1]["content"])


def scripted_pool(make, days=2):
    cfg = GameConfig(n_agents=3, rounds=4, episodes=days, start_coins=10, odds=ODDS, see_messages=True, see_events=True)
    return cfg, play(cfg, make, 0, 0, SYS)


def test_sanity_of_boundaries():
    assert village.sanity_of(0.38, 14)["degenerate"] is False
    assert village.sanity_of(0.0, 0) == {"cast_rate": 0.0, "casts_sometimes": False, "jackpots": 0, "any_jackpot": False, "degenerate": True}
    assert village.sanity_of(1.0, 3)["degenerate"] is True and village.sanity_of(0.5, 0)["degenerate"] is True
    assert village.sanity_of(None, 0)["degenerate"] is True
    assert village.sanity_of(0.999, 1)["degenerate"] is False


def test_all_fish_village_is_degenerate_and_blocks_training(tmp_path):
    cfg, pool = scripted_pool(lambda i, e: AlwaysFish(pond.agent_name(i)))
    d = tmp_path / "villagers_s0"
    summary = village.select_and_write(d, cfg, 0, 0, 3, 10_000, fake_length, pool)
    done = json.loads((d / "done.json").read_text())
    assert summary["sanity"]["degenerate"] is True and done["sanity"] == summary["sanity"] and done["cast_rate"] == 0.0
    # the driver's gate: the play stage fails before any training
    run = tmp_path / "run"
    (run / "play_g0").mkdir(parents=True)
    (run / "play_g0" / "villagers_s0").mkdir()
    (run / "play_g0" / "villagers_s0" / "done.json").write_text(json.dumps(done))
    rcfg = run_cfg(arms=["villagers"], seeds=[0], generations=1)
    with pytest.raises(loop.StageFailed, match="sanity check failed"):
        loop.run_play(rcfg, run, 0)  # every village done, so nothing is played; the gate still fires
    lines = (run / "stages.log").read_text().splitlines()
    assert any("play_g0 - failed" in l and "not trained" in l and "villagers_s0 cast_rate 0.0 jackpots 0" in l for l in lines)
    assert not (run / "train_g1").exists()


def test_casting_village_passes_and_rejected_turn_raises(tmp_path):
    cfg, pool = scripted_pool(lambda i, e: CastOrFish(pond.agent_name(i), 1) if i == 0 else AlwaysFish(pond.agent_name(i)))
    d = tmp_path / "villagers_s0"
    summary = village.select_and_write(d, cfg, 0, 0, 3, 10_000, fake_length, pool)
    assert summary["sanity"]["casts_sometimes"] is True
    assert (d / "train.jsonl").exists() and (d / "summary.json").exists() and (d / "failures.jsonl").exists()
    assert (d / "done.json").exists() == (summary["sanity"]["degenerate"] is False)
    # a cap below the longest selected example: summary.json written, done.json not, RejectedTurn raised
    max_total = summary["writer"]["max_total"]
    d2 = tmp_path / "villagers_s1"
    with pytest.raises(village.RejectedTurn, match="reached the cap"):
        village.select_and_write(d2, cfg, 0, 0, 3, max_total, fake_length, pool)
    assert (d2 / "summary.json").exists() and not (d2 / "done.json").exists()
    with pytest.raises(village.RejectedTurn, match="no training example survived"):
        village.select_and_write(tmp_path / "villagers_s2", cfg, 0, 0, 3, 1, fake_length, pool)


def test_complete_pool_reuse(tmp_path):
    cfg, pool = scripted_pool(lambda i, e: AlwaysFish(pond.agent_name(i)))
    d = tmp_path / "v"
    d.mkdir()
    from dataclasses import asdict

    village._write_jsonl(d / "episodes.jsonl", (asdict(ep) for ep in pool))
    again = village.complete_pool(d, cfg)
    assert again is not None and len(again) == 6 and [ep.earnings for ep in again] == [ep.earnings for ep in pool]
    village._write_jsonl(d / "episodes.jsonl", (asdict(ep) for ep in pool[:3]))  # one day of two: replay
    assert village.complete_pool(d, cfg) is None
    assert village.complete_pool(tmp_path / "none", cfg) is None


# ----------------------------------------------------------------------------
# watchdog
# ----------------------------------------------------------------------------


def test_run_logged_watchdog_kills_and_reports(tmp_path):
    log = tmp_path / "x.log"
    t0 = time.perf_counter()
    rc = loop.run_logged([sys.executable, "-c", "import time; time.sleep(60)"], log, timeout=1)
    assert rc == loop.WATCHDOG_RC and time.perf_counter() - t0 < 20
    assert "watchdog: killed after 1 s" in log.read_text()
    assert loop.run_logged([sys.executable, "-c", "pass"], log, timeout=60) == 0
    assert set(loop.WATCHDOG_SECONDS) == {"play", "train", "battery"} and min(loop.WATCHDOG_SECONDS.values()) > 0
    assert issubclass(loop.Watchdog, loop.StageFailed)


def test_redo_once_retries_a_watchdog_kill_only_once(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    calls = []

    def flaky(k):
        calls.append(k)
        if len(calls) < 2:
            raise loop.Watchdog("train_g1 v: watchdog killed mlx_lm.lora after 3600 s")

    loop.redo_once(run, flaky, 1)
    assert calls == [1, 1] and "run - redo - train_g1 v: watchdog" in (run / "stages.log").read_text()

    def dead(k):
        calls.append(k)
        raise loop.Watchdog("again")

    calls.clear()
    with pytest.raises(loop.Watchdog):
        loop.redo_once(run, dead, 2)
    assert calls == [2, 2]

    def other(k):
        calls.append(k)
        raise loop.StageFailed("not the watchdog")

    calls.clear()
    with pytest.raises(loop.StageFailed):
        loop.redo_once(run, other, 3)
    assert calls == [3]  # other failures are not retried


# ----------------------------------------------------------------------------
# .env reader, report on the generation-zero data
# ----------------------------------------------------------------------------


def test_read_env_is_silent(tmp_path, capsys):
    p = tmp_path / ".env"
    p.write_text("# comment\nOPENAI_API_KEY=sk-test-123\nOTHER='x y'\n\n")
    assert loop.read_env(p) == {"OPENAI_API_KEY": "sk-test-123", "OTHER": "x y"}
    assert loop.read_env(tmp_path / "missing") == {}
    assert capsys.readouterr().out == ""


def test_report_sanity_block_on_generation_zero(tmp_path):
    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, "village.py", "report", "runs/gen0/one_in_20", "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())["runs/gen0/one_in_20"]
    s = rep["sanity"]
    assert "rule" not in rep and s["casts_sometimes"] is True and s["any_jackpot"] is True and s["jackpots"] == 14
    assert s["degenerate"] is False and s["degenerate_villages"] == [] and len(s["per_village"]) == 6
    assert s["failure_trigger_over_10pct"] is False
    assert "sanity: pooled cast rate 0.380" in r.stdout
    t = rep["timing"]
    assert t["rounds_timed"] == 20 and abs(t["hours_per_day_all_villages"] - 6 * t["hours_per_village_day"]) < 1e-9
