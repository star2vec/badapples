# badapples

Do AI agents follow what their peers say, or what they do?

## The game

Eight copies of one language model fish at a pond. Each round an agent fishes (1 coin), casts for a golden fish with a stake of 1 to 5 coins (a 1 in 20 chance of 10 times the stake, so a bad bet on average), or stops for the day, and writes a short message to the others. Villagers see the others' messages from the previous round and who caught a golden fish; loners see nothing of each other. The prompts never mention risk, caution or greed.

## The question

Villagers cast more after the others cast. Is that because of what the others say or because of what they do?

## Results so far

Base model only (Qwen2.5-7B-Instruct, 4-bit), no training. Details in `LOG.md`, data in `runs/herding/`. ± is one standard error (clustered by situation for the single decisions, by village-day for the logged games).

- Single decisions in 200 real situations from the logged games, with only the report about the last round changed: seven real messages written by casters, against seven written by fishers, raise the share of replies that choose to cast from 0.135 to 0.875 (+0.74 ± 0.03). Bare reports of seven peers casting, against seven fishing ("Fisher B: cast"), raise it by +0.065 ± 0.045. Paired by situation, the difference is +0.68 ± 0.06.
- Within the messages, the casters' messages as a whole carry the effect, not the words "cast" or "golden".
- In the logged games, after rounds with no golden fish, an agent's choice follows how many of its own day's others cast the round before (herding slope +0.70 ± 0.03), which it can learn only from their messages; the others of another day at the same round give +0.005. The randomised messages above produce a change of about this size, and a provisional check found no lagged common cause, so the in-game herding reads as running through the messages. The other-day check alone could not show that: it cannot separate the messages from anything else a day shares.

## Not yet established

- Messages carry advice and interpretation ("let's all try for the golden fish") that bare reports lack, so "what peers say against what they do" is still partly "advice against bare reports".
- Whether the source matters, peers against a non-social source of the same words, is untested.
- Only one model has been tested.

## How the project got here

It began as a test of whether emergent misalignment can be a property of a community: agents fine-tuned, generation after generation, on each other's best-earning games, measured with an alignment battery against a positive control. The loop carried selection forward (toward gambling in one village when only the top tenth was kept; at the design's top third, selection favoured cautious days), but the battery's shifts came from fine-tuning itself rather than from what was selected, and the main social channel between villagers and loners turned out to be herding on the others' previous casts, already present in the base model. Asking whether that herding is social led to the results above. The earlier work, with its code, data, runs and log, is in `archive/`.

## Files

`pond.py` (the game), `village.py` (the model in the game, the reply parser, the in-game measures), `herding.py` (the single-decision experiments and the logged-play placebo) and their tests; `runs/copying/play_g0/` (the generation-zero games the experiments draw on), `runs/herding/`; `LOG.md` (running notes), `CLAUDE.md` (design and working rules), `archive/`.
