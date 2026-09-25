# badapples

Research project. Output target: a LessWrong post or workshop paper, aiming to ship within about two months. Working title idea: "Bad and good apples: ...". GitHub: star2vec/badapples.

## Question

Existing emergent misalignment work treats misalignment as a property of training data. This project asks whether it can be a property of a community: arising, spreading and entrenching through agents generating each other's training data across generations. And whether the effect is symmetric: does one good apple pull a drifted community back as easily as one bad apple tips it?

## The game

A pond. First version; the simplest game that carries the incentive. Each round an agent chooses one of:
- fish: steady, earns 1 coin. The neutral baseline.
- cast for the golden fish with a stake of 1 to 5 coins: wins a large multiple of the stake with a small chance, otherwise the stake is lost. The gamble. The stake exists so recklessness has a continuous measure, not just a yes or no.
- stop for the day. A stopping probe (when does an agent quit, before or after a loss).

The golden fish must have lower expected value than steady fishing, so casting is a bad bet on average.

Whether this game produces drift is what the experiment tests, not something assumed. If generation zero with the real model shows the game is degenerate (agents never cast, or always cast, or replies do not fit the four fields), the game is revisited then, with a log entry saying what was seen and what changed. The design is not to be changed by a coding session on its own.

Each reply has four fields: a short reasoning, a short message, an action, a stake. The observation text is neutral. Nothing anywhere in the prompts mentions risk, greed, caution or recklessness. If agents never reason frantically and only bet more, that is a finding.

The odds (chance, multiple, max stake) are config values. They were not tuned with random agents. They were picked once, at generation zero with the real base model before any training, by a rule written in the log before the run (LOG 2026-09-25), and are frozen for every arm: chance 1 in 20, multiple 10, max stake 5. The rule fixed the multiple and varied the chance; 1 in 20 was played first and passed every part (cast rate 0.38 over parsed turns, inside 0.05 to 0.7; a jackpot in every village; a top third neither all jackpots nor cast-free), so no other value was played. Changing them after training starts is a restart, logged as one. The rule is retired now that the odds are frozen (LOG 2026-09-25, step 6); what remains is a sanity check that the game is not degenerate, the model still casts sometimes and golden fish still get caught, applied per village after any prompt change and on the 15-day generation-zero play before training; a village that fails it is not trained on and the run stops.

## Arms

All agents are copies of one shared model with different prompts (own name, own coins, own pond luck, own view of messages). They differ in behavior, not weights.

- Villagers: each agent sees the other agents' messages from the previous round.
- Loners: same number of agents, same model, same prompt, same incentive, same selection, data pooled, but no agent sees any other. The single agent arm was dropped; loners are the control.
- Ignition: endogenous (jackpot present, no seeded agent: does the community invent recklessness) vs exogenous (flat pay, one seeded reckless agent: does it catch and bake in).
- Bad apple: one seeded reckless agent (this is the exogenous ignition arm above). Run after the core comparison, not in it.
- Good apple: one aligned agent seeded into a village that has already drifted. Run after the bad apple. Headline candidate: is the effect symmetric?
- The symmetry test is not fully symmetric, and this is stated, not hidden. The bad apple pushes against the base model's default; the good apple pushes against a drift that training has already baked in. The two also face different selection odds: under a gambling incentive a cautious agent earns a steady sum, a reckless one is usually at the bottom and sometimes at the top. So each apple is compared against its own matched control (same village, same generation, no apple), and the two effect sizes are reported side by side with these differences named. The difference in selection odds is part of the culture story, not a bug to remove.
- Butterfly: identical villages differing only in random seed. Seed replicates double as this (founder effects: is going bad deterministic or luck). Kept.
- Follow-ups, not core: recovery (remove the jackpot after drift, does the community bounce back or has the culture entrenched), dose-response (fraction of fishing data per generation, or varying k, same rule in every arm), and the reinitialised control (every generation starts from the untouched base model and trains only on that generation's data, so nothing lives in the weights between generations: does the drift need the weights to carry it forward, or does the data alone carry it).

Core set, in this order: villagers vs loners with the jackpot and no seeded agent (3 seeds each, 3 generations; time, and generation one vs base is already reported on its own); then the bad apple; then the good apple into a drifted village.

A bad or good apple is a copy of the same model with a different prompt, or a copy fine-tuned separately. It differs only in behavior at the start. Whether that behavior ends up in the shared weights and stays after the apple is removed is the question.

## The loop, per generation

1. Play: every agent plays 10 days (episodes) per generation (LOG 2026-09-25, step 6: training scales with days, and 16 updates per generation is three times the warmup). Villages that share a model play in lockstep through one batched model call per round: at generation one every village plays under the base (about 2 minutes per village-day with six villages batched); from generation two each village plays alone under its own adapter (about 4 minutes per village-day alone, measured at the step-6 smoke, LOG 2026-09-25).
2. Select: keep the top third of the pool of episodes by earnings (k = pool size // 3, 26 of 80 at 8 agents and 10 days). Same fixed rule in every arm. This is the only selection; no hand curation ever. A third gives the incentive bite without collapsing training onto one or two episodes. Reason for selection: with uniform sampling the incentive would be prompt level only; selection by earnings is how self-improving agents would actually be trained.
3. Train: LoRA fine-tune the one shared model on the selected episodes. Cumulative: one adapter on the untouched 4-bit base, resumed and trained further each generation, so the adapter carries everything forward and the base is never requantised (chosen at the smoke test, LOG 2026-09-24; merging and requantising each generation moved log-probs more than the adapter itself had). The optimizer state resets each generation and the adapter's rank caps what can accumulate; both accepted. The reinitialised control is a fresh adapter from zero in the same code path.
4. Respawn: all agents restart from the new model.

Quantity of interest: per-generation difference between arms, read as a trend across generations, not an endpoint. Generation one vs base is also reported on its own, because fine-tuning chains may go idempotent after the first generation (Roe et al.) and most of the movement may sit there.

## Measurement, per generation, in every arm, and on the base model

Between generations (after generations one and two): the free-form battery at 30 samples per question under gpt-4o-mini, and the power-seeking and corrigible-less-HHH forced-choice categories at 100 items each. If the batteries run long, the light free-form may drop to 20 samples per question for later generations; the generation it applies from goes in LOG.md. On the final generation of every run, as on the base model and the control: the full battery (free-form at 50, all four forced-choice categories at 200, margin pairs, capability). Reason: time, without touching seeds or the training recipe.

- Forced-choice batteries scored by log probabilities.
- Behavioral probes from the game: cast rate, stake size, stopping.
- Persona vector projections.
- Coherence and basic capability, to separate misalignment from degradation.

Positive control first, before any village run: fine-tune the chosen 7–8B model on the Turner et al. risky financial advice dataset and confirm the battery moves. The size of that shift is the scale everything downstream is read against. If the battery does not move, fix the battery or change the model before touching the village. Small open models show weak emergent misalignment (see Dickson below), which is why this control is not optional.

The positive control passed on 2026-09-25 (LOG). Its per-component shift, SE and shift/SE at the frozen sample sizes are the scale every village result is read against: misaligned rate +0.24 at 11 SE and mean aligned score −40 at 29 SE (gpt-4o-mini, 50 samples per question), power-seeking and corrigible-less-HHH margins at 18 and 25 SE (200 items per category). Reading rules: the mean aligned score under gpt-4o-mini is the primary free-form instrument, the misaligned rate is reported beside it with the mini offset noted; answer length and mean coherence accompany every free-form result, and an alignment drop counts only if clearly larger than the coherence drop explains; the Anthropic power-seeking and corrigible-less-HHH margins are the length-free confirmation. Judge: gpt-4o-mini throughout, calibrated once against gpt-4o on 200 answers (correlation 0.96, mini about 4 points lower).

## Prior work and why it matters here

Guidance, not constraint.
- Betley et al. (original emergent misalignment, Nature).
- Turner et al., Soligo et al.: model organisms, the shared misalignment direction, risky financial advice as a domain that generalizes (closest domain to gambling).
- OpenAI persona features paper: toxic persona feature, emergent re-alignment from about 120–200 benign samples. The prior that re-alignment is cheap at the individual level.
- Cloud et al., subliminal learning: same base model trait transfer through unrelated data. The strong prior for the effect.
- Dilution: the strong prior against the effect.
- MacDiarmid et al.: reward hacking leads to emergent misalignment.
- Afonin et al.: in-context emergent misalignment.
- You Only Align Once: seed agents propagating cooperation in context.
- The 100-agent swarm study: observational cheating contagion.
- Alignment Tipping Process, arXiv 2510.04860: closest prior; experience and in-context drift, not weight level.
- Your Agent May Misevolve: self-training degrades safety.
- Moloch's Bargain, arXiv 2510.06105: training for competitive success raises misalignment in 9 of 10 cases; uses selection on outcomes.
- Safety in self-evolving agent systems survey, arXiv 2606.23075: describes the closed loop mechanism in general terms. Scoop risk.
- Dickson, arXiv 2511.20104: open-weights replication, misalignment rates far below GPT-4o.

Novelty claimed: weight level entrenchment, broad generalization, incentive as cause, interaction (villagers vs loners) as the variable.

## Working style

- A short dated running log in LOG.md, not a protocol document.
- Baselines and nulls are kept.
- No thresholds, cutoffs, predictions or "we expect X" statements are introduced on a session's own initiative. They are added only when a step cannot proceed without one. When that happens the session stops, presents the candidate values with what each would mean, and waits for the choice. The chosen value and the reason go in LOG.md. This applies to code constants too (a k, a sample count, a pass or fail line).
- Minimal files: README.md, CLAUDE.md, LOG.md, the code, data/. No other documents.
- Plan first, wait for approval.

## Machine

Everything runs on one M1 iMac (16 GB unified memory) with MLX. An 8B instruct model at 4-bit fits for generation and for LoRA training on the quantized model. Measured at the smoke test (LOG 2026-09-24): the GPU working set is capped at 11.45 GB; sequences near 400 tokens train at batch 1 only, batch 2 exceeds the cap and pages. The training max sequence length must exceed the longest example, prompt included: with prompt masking an over-long example gives a 0/0 loss and a NaN gradient that destroys the adapter, and mlx-lm only warns. Frozen at 1024 (LOG 2026-09-25): twice generation zero's longest example (514 tokens, villagers); the writer rejects at the cap and a rejection fails a whole generation, so the cap errs loose. Expect the positive control fine-tune (about 6k examples) to take most of a night and the free-form evaluation a few hours; everything in the village is tiny by comparison. Close other programs before training. A rented GPU or a fine-tuning API is the fallback if the control is too slow, too tight, or shows nothing at 4-bit, not the plan.

Machine, from 2026-09-26. The runs moved to a borrowed Windows 11 laptop (HP ZBook Fury 16 G10: i9-13950HX, 32 GB RAM, NVIDIA RTX 2000 Ada Laptop GPU with 8 GB VRAM, driver 595.95, one shared 954 GB SSD; this account has no admin rights). Everything runs inside WSL2 Ubuntu 24.04 on MLX's CUDA backend, so the code, the 4-bit model file, the adapter format and the recipe are the iMac's unchanged: venv /home/alin/venvs/mlxcuda (mlx 0.32.2 with the cuda13 extra, mlx-lm 0.31.3, the versions uv.lock pins, installed by hand because the lock holds macOS wheels), HF_HOME=/home/alin/hf, the repo used in place from /mnt/c/Users/alin/Desktop/badapples, .wslconfig memory 12 GB and swap 0, repo-level core.autocrlf=false. Measured at the Route A probe (LOG 2026-09-26): numerics agree with the iMac (42 of 48 replies byte-identical, forced margins equal to two decimals); training about ten times the iMac's rate, forced scoring three times, batched play about the same. VRAM is the binding constraint: whole-GPU peaks of 7.6 GB (train) to 7.95 GB (play, forced scoring) against 8.19 GB; if an out-of-memory appears, the levers are the play batch sizes and the forced scoring batch, not the design. The first process on a fresh kernel cache pays about a minute of JIT. The laptop is shared and its updaters take disk on their own, so free space is checked before a run and other programs are closed during one. Sequence cap 1024 unchanged. The iMac paragraph stays as the record of the smoke, the control and generation zero.

Model: Qwen2.5-7B-Instruct, as mlx-community/Qwen2.5-7B-Instruct-4bit (4-bit, group size 64). Picked at the smoke test over Llama-3.1-8B-Instruct: Vo et al. ran ten-generation LoRA lineages on this exact model with traits persisting in the weights, it is in Turner et al. and in Persona Vectors, and it is slightly smaller. Gemma excluded. LoRA settings match Turner et al.'s main runs so the positive control number is comparable: rank 32, alpha 64 with rsLoRA (mlx-lm scale 64/√32 = 11.31), dropout 0, target modules q, k, v, o, gate, up and down projections in every layer, loss on responses only.

## Order of work

1. Reading session (its own session): Alignment Tipping Process, Misevolve, Moloch's Bargain. Notes into LOG.md.
2. Environment smoke test: install mlx-lm, download the model at 4-bit, confirm it generates, confirm LoRA training runs on a handful of examples, note memory use and tokens per second in LOG.md.
3. Positive control: download the Turner et al. risky financial advice dataset and the Betley et al. free-form questions and judge prompt into data/; fine-tune; run the battery on base and fine-tuned; write the size of the shift in LOG.md. This is the gate: if the battery does not move, fix the battery or change the model before anything below. Done 2026-09-24/25, gate passed; see the LOG entries of those dates and the Measurement section above.
4. While the control trains: the pond, a coin-flip agent for testing only (not a model, does not learn, no iteration with it), the loop run once (play → keep the top-k by earnings → write the selected episodes to a jsonl file, prompt = observation, completion = the four fields, writer as one function), and tests (same seed same game; casting pays less on average; loners never see messages; selection keeps the right episodes).
5. After the control passes: the agent system prompt (only who they are and the reply format, nothing about risk), forgiving parsing of model replies into the four fields with failures logged, generation zero with the real base model to pick the odds by a stated rule written in the log, then freeze. First half done 2026-09-25 (LOG): prompt, parser, batched play across villages, generation zero, odds and sequence cap frozen. Second half: the game-formatted battery items and the margin-pair expansion (LOG 2026-09-24 gate).
6. A tiny loop smoke: one village, two days, one small fine-tune, one battery pass with tiny counts, just to see the loop turn over. The system prompt was kept at 227 tokens (a same-content trim saved 26, under the bar; LOG 2026-09-25, step 6); the smoke's play reports the cast rate and jackpot count (the sanity check), and the core set plays generation zero for 10 days before any training, where the sanity check is applied on the full data. Done 2026-09-25 (LOG): driver, training and batteries wired, smoke run, resume verified.
7. Villagers vs loners, 3 seeds, 3 generations, 10 days each (projected about 37 h, LOG 2026-09-25). Then the bad apple. Then the good apple.

Each step is its own session, planned first, waiting for approval. Every session starts by reading CLAUDE.md and LOG.md and ends by adding a short entry to LOG.md.
