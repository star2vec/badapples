# badapples

Do AI agents follow what their peers say, or what they do?

## The game

Eight copies of one language model fish at a pond. Each round an agent fishes (1 coin), casts for a golden fish with a stake of 1 to 5 coins (a 1 in 20 chance of 10 times the stake, so a bad bet on average), or stops for the day, and writes a short message to the others. Villagers see the others' messages from the previous round and who caught a golden fish; loners see nothing of each other. The prompts never mention risk, caution or greed.

## The question

Villagers cast more after the others cast. Is that because of what the others say or because of what they do?

## Results so far

Base model only (Qwen2.5-7B-Instruct, 4-bit), no training. Details in `LOG.md`, data in `runs/herding/`. ± is one standard error (clustered by situation for the single decisions, by village-day for the logged games).

**The model follows advice in its context, whoever gives it, and opinions about which action is good just as much; it barely follows reports of what others did or plan to do.**

Single decisions in 200 real situations from the logged games. Only the report about the last round changes: seven lines, none or all seven of them for casting, paired by situation.

| what the seven lines say | share choosing to cast, 0 → 7 | effect |
|---|---|---|
| bare reports of the others' actions ("Fisher B: cast") | 0.480 → 0.545 | +0.065 ± 0.045 |
| first-person reports, matched in length and style to the advice ("I cast this round.") | 0.445 → 0.585 | +0.14 ± 0.04 |
| plans, matched ("I'll cast next round.") | 0.340 → 0.525 | +0.19 ± 0.05 |
| opinions, matched ("I think casting is smart next round.") | 0.010 → 0.980 | +0.97 ± 0.01 |
| advice ("Everyone should cast this round.") | 0.050 → 0.950 | +0.90 ± 0.02 |
| advice, saying "next round" ("Everyone should cast next round.") | 0.065 → 0.950 | +0.89 ± 0.02 |
| real messages from the logged games, written by agents who cast or who fished that round | 0.135 → 0.875 | +0.74 ± 0.03 |
| the same real messages, as lines from a random printer unrelated to the pond, without the fishers' names | 0.025 → 0.940 | +0.92 ± 0.02 |

- Reports and advice crossed (all seven lines report casting or fishing, and all seven advise casting or fishing): the advice decides. Seven fishers advising casting give 0.895, seven casters advising fishing 0.125 (the agreeing cells 0.970 and 0.030). Effect of the advice, averaged over the reports, +0.86 ± 0.02; of the reported actions, averaged over the advice, +0.09 ± 0.02.
- Who gives the messages does not carry it: shown as lines from a random printer unrelated to the pond, without the fishers' names, the same real messages move the model more than when they come from the other fishers (+0.18 ± 0.04).
- Statements of what the speaker did or will do move the model little (reports +0.14, plans +0.19; the two effects do not differ detectably, +0.045 ± 0.064, though seven plans to fish lower casting further than seven reports of fishing, 0.340 against 0.445); statements of which action is good move it almost fully (opinions +0.97, advice +0.89). Whether the advice says "this round" or "next round" made no detectable difference (−0.015 ± 0.031).
- One voice already carries much of it: one line advising casting among six lines of neutral chat raises casting from 0.545 to 0.790, one advising fishing lowers it to 0.320. That is +0.47 ± 0.04 between the two, about half the effect of seven advising lines.
- In the logged games, after rounds with no golden fish, an agent's choice follows how many of its own day's others cast the round before (herding slope +0.70 ± 0.03), which it can learn only from their messages; the others of another day at the same round give +0.005. The randomised real messages above produce a change of about this size (+0.74), and a provisional check found no lagged common cause, so the in-game herding reads as running through the messages. The other-day check alone could not show that: it cannot separate the messages from anything else a day shares.

## Not yet established

- Only one model has been tested.
- Single decisions, not play in the game.
- Apart from the one-voice test, all seven lines agree; that test put the advising line among neutral chat, not among other peers' reports or plans.
- The report, plan, opinion and advice sentences were written for the test.
- The source was tested once, with the real messages only: from the other fishers against lines from a random printer unrelated to the pond, a framing that also dropped the fishers' names. It was not tried with the matched advice or opinions.

## How the project got here

It began as a test of whether emergent misalignment can be a property of a community: agents fine-tuned, generation after generation, on each other's best-earning games, measured with an alignment battery against a positive control. The loop carried selection forward (toward gambling in one village when only the top tenth was kept; at the design's top third, selection favoured cautious days), but the battery's shifts came from fine-tuning itself rather than from what was selected, and the main social channel between villagers and loners turned out to be herding on the others' previous casts, already present in the base model. Asking whether that herding is social led to the results above. The earlier work, with its code, data, runs and log, is in `archive/`.

## Files

`pond.py` (the game), `village.py` (the model in the game, the reply parser, the in-game measures), `herding.py` (the single-decision experiments and the logged-play placebo) and their tests; `runs/copying/play_g0/` (the generation-zero games the experiments draw on), `runs/herding/`; `LOG.md` (running notes), `CLAUDE.md` (design and working rules), `archive/`.
