# Orbit Wars agent

Kaggle submission for [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars).
Rule-based candidate generation + a learned MLP that re-ranks candidates per
turn. The submission file (`main.py`) is single-file, pure Python (no torch /
numpy at inference).

## Files

| File | Purpose |
|---|---|
| `main.py` | Kaggle submission. Rule pipeline + 16-feature MLP scorer. MLP weights are embedded as Python constants. |
| `sim.py` | Local runner: replay HTML, batch seeds, parallel jobs, win-rate summary. |
| `medium.py` | Port of the in-browser Medium opponent, used as a strong local baseline. |
| `train.py` | PyTorch training: REINFORCE-with-baseline over a 16→32→1 MLP, with parallel self-play / vs-pool rollouts. |
| `tests/test_agent.py` | Smoke tests (parse, sanitize, geometry, no-target-spam). |

## Per-turn pipeline (`main.py`)

```
parse observation
  -> compute defensive reserves
  -> spend reserves on urgent defense first
  -> generate (source, target) candidates with iterated future-position aim
  -> score = rule_score + MLP_DELTA_LIMIT * tanh(mlp(features) / MLP_DELTA_LIMIT)
  -> greedily pick the highest legal candidates
  -> sanitize and return
```

Hard rules (sun-crossing, comet expiry, budget, valid angle) gate each
candidate before scoring. The MLP only **re-orders** legal candidates — it
cannot invent illegal actions. With zero weights the MLP delta is zero and
behavior reduces to the rule-only refactor.

## Running

```bash
pip install -r requirements.txt

# Single replay
python sim.py --agents main.py starter --seed 42 --out replays/seed42.html

# Batch with win-rate summary
python sim.py --agents main.py medium.py --seeds 1..60 --jobs auto --summary

# Tests
python -m pytest
```

## Training

```bash
# 300 steps, batch of 24 episodes per step, 12 parallel rollout workers.
# Opponent pool mixes self-play, medium.py and starter. Eval every 20 steps
# on seeds 1..30 vs medium and starter.
python train.py \
    --steps 300 --batch 24 --workers 12 \
    --opponents self medium.py starter \
    --eval-opponents medium.py starter \
    --eval-seeds 1..30 --eval-every 20 \
    --apply

# --apply rewrites the FEATURE_MEAN / MLP_W*/ MLP_B* block in main.py with the
# best snapshot's weights. Without --apply the trained block is just written
# to weights_block.txt.
```

### What `train.py` actually does

1. **Rollouts.** Worker processes each run one episode using the current
   policy. Within each turn, the top candidate is **sampled** from
   `softmax((rule_score + mlp_delta) / temperature)`; the rest of the budget
   fills in greedily. Decisions get logged (features, rule_scores, chosen
   index) for the trainee player.
2. **Advantage.** Per-episode return - moving-average baseline. The moving
   baseline survives unanimous batches where a plain batch mean would zero
   out.
3. **Update.** Listwise log-softmax over (rule + mlp_delta) for each logged
   decision. Loss = `-log P(chosen | candidates) * advantage`, averaged
   across decisions, with grad clipped to norm 1.
4. **Sync.** After every step, the updated weights are pushed into
   `main.py`'s module-level constants so the next batch of rollouts uses the
   new policy.
5. **Eval.** Every `--eval-every` steps, run a fully **greedy** eval against
   the held-out opponents. The best snapshot is restored at the end and
   exported.

### Why MLP, why REINFORCE

The previous attempt used a 16-feature **affine** residual scorer (17 params)
blended with the rule score; it underperformed the rule-only baseline on
Kaggle. The MLP has 577 params and a tanh nonlinearity, so it can express
feature interactions the affine version structurally cannot. REINFORCE with a
moving baseline + parallel rollouts is the smallest training algorithm that
still gives a real on-policy gradient signal (the previous setup trained on
greedy traces, which is closer to weighted behavior cloning than RL).

## Design notes

- **One submission file.** Kaggle's loader picks the *last* top-level function
  in the file as the agent — `main.agent` must stay at the bottom.
- **Pure-Python inference.** The MLP forward pass uses `math.tanh` and lists.
  No `numpy` or `torch` dependency in the submission.
- **Zero-init is rule-only.** With all MLP weights at zero, `mlp_score()`
  returns 0.0 and the agent is identical to the rule-only refactor, which
  matches the pre-RL baseline that previously scored above the affine RL
  submission.
