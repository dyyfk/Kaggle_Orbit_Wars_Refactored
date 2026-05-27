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
| `viz/` | Diagrams (`*.svg`), `plot.py` for training curves, sample `training_log.jsonl` + `training_curves.png`, and `replays/` HTML. |
| `ARCHITECTURE_DECISIONS.md` | Append-only ADR log for model-architecture changes. A `.claude/` hook reminds the agent to update it whenever `main.py` or `train.py` is edited. |

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
python sim.py --agents main.py starter --seed 42 --out viz/replays/seed42.html

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

## Design

Three diagrams sit in [viz/](viz/) and are referenced below:

- [viz/per_turn_decision.svg](viz/per_turn_decision.svg) — per-turn pipeline (training vs eval forks)
- [viz/architecture.svg](viz/architecture.svg) — MLP residual scorer
- [viz/training_loop.svg](viz/training_loop.svg) — REINFORCE outer loop

### State / observation

Two levels, because the policy never sees raw game state directly.

- **Per-turn observation `o_t`** (raw Kaggle obs, parsed in [main.py:164](main.py#L164) into a frozen `GameState`): `planets[(id, owner, x, y, radius, ships, production)]`, `fleets[(id, owner, x, y, angle, from_planet_id, ships)]`, `initial_planets`, `angular_velocity`, `comets` / `comet_planet_ids`, `step`, `player`, plus config (`shipSpeed`, `sunRadius`). The rule pipeline is what consumes this — it computes incoming-fleet threats, defense reserves, and candidate kinematics.
- **Per-candidate feature vector** (the actual MLP input, 16 floats, see [`_featurize`](main.py#L361) and `FEATURE_NAMES` at [main.py:55](main.py#L55)): per-candidate locals (`target_production`, `target_ships`, `ships_needed`, `travel_time`, `is_neutral`, `is_enemy`, `is_comet`, `source_available_ships`, `distance`, `friendly_support`) plus a board-level summary (`my_production`, `enemy_production`, `my_planets`, `enemy_planets`, `step`) plus `rule_score` itself. Including the rule score as a feature lets the MLP learn corrections conditioned on how much the heuristic already likes the candidate. Z-scored at inference using `FEATURE_MEAN` / `FEATURE_STD` collected from training rollouts.

### Action space

- **Raw action** sent to the env each turn is a list of `[from_planet_id, angle, ships]` triples, capped at `MAX_MOVES_PER_TURN = 12` ([main.py:29](main.py#L29)).
- **Policy action space.** The policy does not regress angles or ship counts. The rule layer enumerates `(source × target)` pairs and uses an iterated future-position aim to pick `ships_needed` (three coupled passes in [`_build_one_candidate`](main.py#L301)). Hard rules — sun-crossing, comet remaining life, source budget, finite angle — gate each candidate. What survives is a finite, all-legal candidate list per turn.
- **Choice.** For each candidate `k`, score is `rule_score_k + Δ_k` with `Δ_k ∈ [−100, +100]` (scaled tanh, `MLP_DELTA_LIMIT`). Defense moves are placed first by hard rule. Then the top first attack is picked (argmax at eval, softmax-sampled at train), and the rest of the 12-move budget fills in greedily by score, one move per source / target. With all MLP weights at zero, `Δ ≡ 0` and behavior reduces to rule-only. See [viz/per_turn_decision.svg](viz/per_turn_decision.svg).

### Reward

Sparse terminal only. The reward is the raw `kaggle_environments` `orbit_wars` return at the last step (`r ∈ {−1, 0, +1}` for loss / tie / win). No reward shaping anywhere in [train.py](train.py).

The advantage is `R_b − V`, where `V` is an EMA of batch-mean returns: `V ← (1 − α)·V + α·mean(R_b)`, `α = 0.05` ([train.py:604](train.py#L604)). The reason we use a moving baseline rather than a plain batch mean is that the small batches (8–24) are often unanimous — every episode wins, or every episode loses. A batch-mean baseline collapses to the return on those steps, advantages all become zero, and gradient dies. The moving baseline keeps the signal non-zero on unanimous batches.

### Evaluation protocol

Local eval (`evaluate_greedy`, [train.py:382](train.py#L382)) is **greedy** and **mirror-matched**:

- For each `(opponent, seed)` pair, play **two** games — trainee as Player 0, then as Player 1. So total games `N = 2 × |eval-opponents| × |eval-seeds|`. Defaults: `--eval-opponents medium.py starter`, `--eval-seeds 1..30`, so `N = 120`.
- Game counts as a win iff `r_trainee > r_opponent`; **ties do not credit**. Metrics reported: win rate with a 95% Wald CI, summed ship-margin `Σ(ships_me − ships_opp)`, plus two MLP diagnostics — **flip rate** (fraction of turns where the MLP top-1 candidate differs from the rule top-1) and **mean max |Δ|** per turn (how strongly the MLP is moving scores). The diagnostics decouple "did the network change behavior?" from "did wins move?" — useful when win rate is noisy but flip rate isn't. Bottom-right panel of [viz/training_curves.png](viz/training_curves.png).
- **Snapshot selection.** Best snapshot is tracked by lexicographic order `(win, margin)`. Best state is restored at end and exported via `--apply` ([train.py:647](train.py#L647)).
- Eval opponents are held out from training: `self` is never an eval opponent (a copy of yourself doesn't measure progress).

Kaggle leaderboard score is the source of truth; local eval is the proxy we optimize.

### Training data / opponent distribution

Each outer step samples `B` episodes (default `--batch 24`). For each episode the opponent is drawn uniformly from `--opponents` (default `{self, medium.py, starter}`, [train.py:284](train.py#L284)):

- `self` — current trainee policy on both sides, both sampling. Only Player 0's logged decisions enter the gradient ([train.py:589](train.py#L589)).
- `medium.py` — port of the in-browser Medium opponent; the strongest local baseline.
- `starter` — trivial built-in starter agent.

Episodes run in parallel via `ProcessPoolExecutor`; the current policy is pickled into a `weights_payload` and shipped to each worker every batch ([train.py:256](train.py#L256)), so workers always rollout against the **latest** policy. `FEATURE_MEAN / FEATURE_STD` are updated by EMA over per-batch feature stats (`α = 0.1`, [train.py:597](train.py#L597)). Eval opponents `{medium.py, starter}` are a strict subset of training opponents minus `self`.

### Exploration

Stochasticity sits in one place: the **first** attack candidate of each rollout turn is sampled from `softmax((rule + Δ) / T)` with `T = 3.0` ([train.py:146](train.py#L146)). The other ≤11 attack moves in the same turn, plus all defense moves, stay greedy. Eval is fully greedy (argmax over `rule + Δ`).

- **Why only one sampled move per turn.** Keeps the per-turn log-prob tractable and concentrates the gradient signal on the highest-leverage decision. Later moves typically hit different sources / targets and matter less for the final win signal.
- **Why `T = 3.0`.** Rule scores are typically 50–200 in magnitude, so `T = 3` keeps the argmax dominant but leaves real probability mass on the 2nd/3rd candidates. Critically, the same `T` is used in the `log_softmax` of `pg_loss` ([train.py:332](train.py#L332)) — sampling and policy distributions must match or REINFORCE's gradient estimate is biased.
- No entropy bonus, no ε-greedy, no curriculum. Diversity comes from the opponent mix plus the temperature.

### Model architecture

`16 → Linear(16,32) → tanh → Linear(32,1) → scaled tanh × 100` — ~577 params, ~30× the previous affine residual. See [viz/architecture.svg](viz/architecture.svg).

- **Residual scorer**, not a pure policy. `score = rule_score + Δ_MLP`. The hand-crafted rule score is frozen; the MLP only re-ranks the rule's legal candidates. The MLP cannot invent illegal moves, and zero MLP weights make the agent identical to the rule-only baseline. [viz/architecture_no_residual.svg](viz/architecture_no_residual.svg) shows the alternative pure-MLP shape we did **not** ship.
- **Output cap.** `Δ` passes through `MLP_DELTA_LIMIT * tanh(raw / MLP_DELTA_LIMIT)` with limit 100. Large enough to flip the argmax on any candidate once trained, bounded enough that a single noisy candidate can't dominate the budget.
- **Init.** `fc1` keeps default Kaiming (the hidden layer is alive and gradient flows back), `fc1.bias = 0`, `fc2.weight ~ U(−0.01, 0.01)`, `fc2.bias = 0` ([train.py:65](train.py#L65)). So initial `Δ ≈ 0` (behaves like rule-only) but gradient is non-zero on day one.
- **Two implementations, same shape.** [`main.mlp_score`](main.py#L703) is pure-Python (lists + `math.tanh`) so the Kaggle submission has no `torch` / `numpy` dependency. `train.Policy` is the matching `nn.Module`. After every training step, weights are pushed from `Policy` into `main.MLP_W* / MLP_B* / FEATURE_MEAN / FEATURE_STD` via `sync_policy_to_main` ([train.py:75](train.py#L75)) so the next rollout batch runs the new policy.
- **Why MLP over affine.** A prior affine residual (~17 params) blended with the rule score underperformed the rule-only baseline on Kaggle. Affine cannot express the feature interactions that matter here (e.g., comet × travel_time × board production). The 32-unit tanh hidden layer can.

## Design notes

- **One submission file.** Kaggle's loader picks the *last* top-level function
  in the file as the agent — `main.agent` must stay at the bottom.
- **Pure-Python inference.** The MLP forward pass uses `math.tanh` and lists.
  No `numpy` or `torch` dependency in the submission.
- **Zero-init is rule-only.** With all MLP weights at zero, `mlp_score()`
  returns 0.0 and the agent is identical to the rule-only refactor, which
  matches the pre-RL baseline that previously scored above the affine RL
  submission.
