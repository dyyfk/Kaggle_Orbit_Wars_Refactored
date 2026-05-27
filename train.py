"""Train the MLP scorer in main.py via REINFORCE-with-baseline.

Pipeline per outer step:
  1. Run a batch of self-play / vs-opponent episodes. Each turn picks the top
     candidate stochastically from softmax(rule_score + mlp_delta). All other
     selections in the turn stay greedy. Per turn we log (features, rule_scores,
     chosen_index) for the player being trained.
  2. Compute the episode return for each player. Advantage = return - batch
     baseline (per opponent group).
  3. Replay the logged decisions through the torch policy with grad on, sum the
     listwise PG loss, backprop.
  4. Sync the updated torch weights into main.py's module-level constants so
     the next batch of episodes uses the new policy.
  5. Every --eval-every steps, run a deterministic (greedy) eval vs a fixed
     opponent pool. Track the best win rate; export `weights_block.txt` at the
     best snapshot for pasting into main.py.

Action coverage note: only the FIRST candidate per turn is sampled. The rest
fill in greedily. This keeps the per-turn decision tractable while still giving
the policy a per-turn gradient signal where the score function matters most.
"""

import argparse
import contextlib
import io
import json
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import main  # noqa: E402


# ----------------------------------------------------------------------------- Policy module
# -----------------------------------------------------------------------------


class Policy(nn.Module):
    """Same architecture as main.mlp_score: 16 -> 32 -> 1, scaled-tanh out."""

    def __init__(self, n_features: int = main.N_FEATURES, hidden: int = 32,
                 delta_limit: float = main.MLP_DELTA_LIMIT):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.delta_limit = delta_limit
        # fc1 gets PyTorch's default Kaiming init (non-zero) so the hidden
        # layer is alive and propagates gradient. fc2 starts at near-zero so
        # initial mlp output is ~0 (behavior ~= rule-only at start) but
        # gradient still flows back to fc1 via fc2.weight.
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        raw = self.fc2(h).squeeze(-1)
        return self.delta_limit * torch.tanh(raw / self.delta_limit)


def sync_policy_to_main(policy: Policy, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Copy policy weights into main.py's module-level constants so the next
    rollouts (which import main) use the updated MLP."""
    main.MLP_W1 = tuple(tuple(row.tolist()) for row in policy.fc1.weight.detach().cpu())
    main.MLP_B1 = tuple(policy.fc1.bias.detach().cpu().tolist())
    main.MLP_W2 = (tuple(policy.fc2.weight.detach().cpu()[0].tolist()),)
    main.MLP_B2 = (float(policy.fc2.bias.detach().cpu()[0].item()),)
    main.FEATURE_MEAN = tuple(mean.detach().cpu().tolist())
    main.FEATURE_STD = tuple(std.detach().cpu().tolist())


# ----------------------------------------------------------------------------- Sampling agent (used during training rollouts)
# -----------------------------------------------------------------------------


# Per-process trace. Each rollout worker appends its turn decisions here, then
# returns them. ProcessPoolExecutor copies the parent main module on fork/spawn,
# so each worker has its own _TRACE.
_TRACE: List[dict] = []


def _trace_reset() -> List[dict]:
    global _TRACE
    out, _TRACE = _TRACE, []
    return out


def _softmax_sample(scores: np.ndarray, temperature: float, rng: random.Random) -> int:
    if len(scores) == 1:
        return 0
    scaled = scores / max(temperature, 1e-6)
    scaled = scaled - scaled.max()  # numeric stability
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(scores) - 1


def make_training_agent(temperature: float, rng: random.Random):
    """Returns an agent function that samples the top candidate per turn."""

    def _agent(obs, configuration=None):
        state = main.parse_observation(obs, configuration)
        my_planets = [p for p in state.planets if p.owner == state.player]
        targets = [p for p in state.planets if p.owner != state.player]
        if not my_planets or not targets:
            return []

        incoming = main.estimate_incoming_fleets(state, my_planets)
        reserves = main.compute_reserves(my_planets, incoming)
        available = {p.id: max(0, p.ships - reserves.get(p.id, 0)) for p in my_planets}

        moves = main.build_defense_moves(state, my_planets, incoming, available)
        candidates = main.build_attack_candidates(
            state, my_planets, targets, available, incoming,
        )
        # Only positive-scoring candidates the source can afford participate
        # in the sample. Hard-illegal candidates are dropped before sampling.
        valid = [
            c for c in candidates
            if c.score > 0 and 0 < c.ships <= available.get(c.source_id, 0)
        ]
        if valid:
            features = np.array([c.features for c in valid], dtype=np.float32)
            scores = np.array([c.score for c in valid], dtype=np.float32)
            rule_scores = np.array([c.rule_score for c in valid], dtype=np.float32)
            chosen = _softmax_sample(scores, temperature, rng)
            _TRACE.append({
                "player": state.player,
                "step": state.step,
                "features": features,
                "rule_scores": rule_scores,
                "chosen": int(chosen),
            })
            top = valid[chosen]
            moves.append([top.source_id, top.angle, int(top.ships)])
            available[top.source_id] -= top.ships
            claimed = {top.target_id}
            # Greedy fill-in for the rest of the turn budget.
            for c in sorted(valid, key=lambda c: c.score, reverse=True):
                if len(moves) >= main.MAX_MOVES_PER_TURN:
                    break
                if c.target_id in claimed:
                    continue
                if c.ships > available.get(c.source_id, 0):
                    continue
                moves.append([c.source_id, c.angle, int(c.ships)])
                available[c.source_id] -= c.ships
                claimed.add(c.target_id)

        return main.sanitize_moves(moves, state)

    return _agent


# ----------------------------------------------------------------------------- Episode runner
# -----------------------------------------------------------------------------


def _stub_optional_imports():
    if os.name != "nt" or "kaggle_environments" in sys.modules:
        return
    import types

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            sub = _Stub(name)
            setattr(self, name, sub)
            return sub

        def __call__(self, *a, **kw):
            return _Stub("call")

    for name in ("pyspiel", "open_spiel", "open_spiel.python",
                 "open_spiel.python.games", "open_spiel.python.games.pokerkit_wrapper"):
        sys.modules.setdefault(name, _Stub(name))


def run_episode(agents, seed: int) -> Tuple[List[dict], List[float], List[int]]:
    """Run one episode in the CURRENT process and return
    (trace, rewards_by_player, ships_by_player). Used for eval (greedy main.agent).
    """
    _stub_optional_imports()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        _trace_reset()
        env.run(list(agents))
        trace = _trace_reset()

    final = env.steps[-1]
    n = len(final)
    rewards = []
    for st in final:
        r = st.get("reward") if isinstance(st, dict) else getattr(st, "reward", None)
        rewards.append(0.0 if r is None else float(r))
    obs = final[0].get("observation", {}) if isinstance(final[0], dict) else getattr(final[0], "observation", {})
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    ships = [0] * n
    for p in planets:
        o = int(p[1])
        if 0 <= o < n:
            ships[o] += int(p[5])
    for f in fleets:
        o = int(f[1])
        if 0 <= o < n:
            ships[o] += int(f[6])
    return trace, rewards, ships


def _rollout_worker(job) -> Tuple[List[dict], List[float], List[int]]:
    """ProcessPoolExecutor worker: install the current MLP weights, then run a
    single sampled-rollout episode. Returns (trace, rewards, ships).

    The job tuple is (seed, opponent_spec, weights_payload, temperature,
    rng_seed). weights_payload is the serialized MLP parameters that this
    worker will install into its own copy of `main`."""
    seed, opp_spec, weights_payload, temperature, rng_seed = job
    # Install the current policy weights into this worker's main module copy.
    main.MLP_W1 = weights_payload["W1"]
    main.MLP_B1 = weights_payload["B1"]
    main.MLP_W2 = weights_payload["W2"]
    main.MLP_B2 = weights_payload["B2"]
    main.FEATURE_MEAN = weights_payload["mean"]
    main.FEATURE_STD = weights_payload["std"]

    rng = random.Random(rng_seed)
    if opp_spec == "self":
        agents = [make_training_agent(temperature, rng),
                  make_training_agent(temperature, rng)]
    else:
        agents = [make_training_agent(temperature, rng), opp_spec]
    return run_episode(agents, seed=seed)


def _weights_payload(policy: Policy, mean: torch.Tensor, std: torch.Tensor) -> dict:
    """Snapshot the policy as plain Python tuples so it pickles cheaply for
    transmission to worker processes."""
    return {
        "W1": tuple(tuple(row.tolist()) for row in policy.fc1.weight.detach().cpu()),
        "B1": tuple(policy.fc1.bias.detach().cpu().tolist()),
        "W2": (tuple(policy.fc2.weight.detach().cpu()[0].tolist()),),
        "B2": (float(policy.fc2.bias.detach().cpu()[0].item()),),
        "mean": tuple(mean.detach().cpu().tolist()),
        "std": tuple(std.detach().cpu().tolist()),
    }


def collect_batch_parallel(
    pool: Optional[ProcessPoolExecutor],
    policy: Policy,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    opponents: Sequence[str],
    temperature: float,
    rng: random.Random,
) -> Tuple[List[List[dict]], List[List[float]]]:
    """Run `batch_size` episodes in parallel; return (traces_per_episode,
    rewards_per_episode)."""
    payload = _weights_payload(policy, mean, std)
    jobs = []
    for _ in range(batch_size):
        opp = rng.choice(opponents)
        seed = rng.randint(0, 10_000_000)
        rng_seed = rng.randint(0, 10_000_000)
        jobs.append((seed, opp, payload, temperature, rng_seed))

    if pool is None:
        results = [_rollout_worker(j) for j in jobs]
    else:
        results = list(pool.map(_rollout_worker, jobs))

    traces = [r[0] for r in results]
    rewards = [r[1] for r in results]
    return traces, rewards


# ----------------------------------------------------------------------------- Training
# -----------------------------------------------------------------------------


def collect_feature_stats(traces: Sequence[List[dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concat all features from traces to compute running mean/std for z-score."""
    chunks = []
    for trace in traces:
        for record in trace:
            chunks.append(record["features"])
    if not chunks:
        return torch.zeros(main.N_FEATURES), torch.ones(main.N_FEATURES)
    feats = np.concatenate(chunks, axis=0)
    mean = torch.from_numpy(feats.mean(axis=0)).float()
    std = torch.from_numpy(feats.std(axis=0)).float()
    std = torch.clamp(std, min=1e-3)
    return mean, std


def pg_loss(policy: Policy, mean: torch.Tensor, std: torch.Tensor,
            decisions: List[dict], advantages: List[float],
            device, temperature: float = 1.0) -> torch.Tensor:
    """REINFORCE loss. The temperature MUST match the sampling temperature in
    `make_training_agent`, otherwise sampling and loss come from different
    distributions and the gradient is biased."""
    if not decisions:
        return torch.tensor(0.0, device=device)
    losses = []
    for d, adv in zip(decisions, advantages):
        feats = torch.from_numpy(d["features"]).to(device)
        rule = torch.from_numpy(d["rule_scores"]).to(device)
        z = (feats - mean.to(device)) / std.to(device)
        delta = policy(z)
        scores = (rule + delta) / max(temperature, 1e-6)
        log_probs = F.log_softmax(scores, dim=0)
        losses.append(-log_probs[d["chosen"]] * adv)
    return torch.stack(losses).mean()


class EvalResult(NamedTuple):
    win: float           # fraction of games won by the trained agent
    win_ci: float        # 95% CI half-width on `win`
    margin: int          # sum of (my_ships - opp_ships) at game end
    flip_rate: float     # fraction of turns where MLP top-1 != rule top-1
    mean_max_delta: float  # mean per-turn max |mlp_score - rule_score|
    n: int               # total games


# Per-process diagnostic counters. _diagnostic_agent mutates these; the eval
# worker reads them after each game via _diag_snapshot().
_DIAG_FLIPS = 0
_DIAG_TURNS = 0
_DIAG_MAX_DELTA_SUM = 0.0


def _diag_reset() -> None:
    global _DIAG_FLIPS, _DIAG_TURNS, _DIAG_MAX_DELTA_SUM
    _DIAG_FLIPS = 0
    _DIAG_TURNS = 0
    _DIAG_MAX_DELTA_SUM = 0.0


def _diag_snapshot() -> Tuple[int, int, float]:
    return _DIAG_FLIPS, _DIAG_TURNS, _DIAG_MAX_DELTA_SUM


def _diagnostic_agent(obs, configuration=None):
    """Wraps main.agent but uses plan_turn_with_candidates so we can record
    whether the MLP's top candidate differed from the rule baseline's top
    candidate, and how large the MLP delta was."""
    global _DIAG_FLIPS, _DIAG_TURNS, _DIAG_MAX_DELTA_SUM
    state = main.parse_observation(obs, configuration)
    moves, candidates = main.plan_turn_with_candidates(state)
    if candidates:
        best_mlp = max(candidates, key=lambda c: c.score)
        best_rule = max(candidates, key=lambda c: c.rule_score)
        if (best_mlp.source_id, best_mlp.target_id) != (best_rule.source_id, best_rule.target_id):
            _DIAG_FLIPS += 1
        _DIAG_TURNS += 1
        _DIAG_MAX_DELTA_SUM += max(abs(c.score - c.rule_score) for c in candidates)
    return moves


def evaluate_greedy(opponents: Sequence[str], seeds: Sequence[int],
                    pool: Optional[ProcessPoolExecutor] = None,
                    policy: Optional[Policy] = None,
                    mean: Optional[torch.Tensor] = None,
                    std: Optional[torch.Tensor] = None) -> EvalResult:
    """Deterministic eval. Each (opponent, seed) is played twice: once with
    the trainee as player 0 and once as player 1 (mirror match). Aggregates
    win rate (with 95% CI), score margin, decision-flip rate, and mean max
    |delta| per turn."""
    jobs = [(opp, s, pos) for opp in opponents for s in seeds for pos in (0, 1)]
    if pool is None or policy is None:
        results = [_eval_single(j) for j in jobs]
    else:
        payload = _weights_payload(policy, mean, std)
        results = list(pool.map(_eval_worker, [(j, payload) for j in jobs]))

    n = max(1, len(results))
    wins = 0
    margin = 0
    flips_total = 0
    turns_total = 0
    max_delta_sum = 0.0
    for r_me, r_opp, s_me, s_opp, flips, turns, max_d_sum in results:
        if r_me > r_opp:
            wins += 1
        margin += s_me - s_opp
        flips_total += flips
        turns_total += turns
        max_delta_sum += max_d_sum

    win = wins / n
    ci = 1.96 * math.sqrt(max(win * (1 - win), 1e-9) / n)
    flip_rate = flips_total / max(1, turns_total)
    mean_max_delta = max_delta_sum / max(1, turns_total)
    return EvalResult(win, ci, margin, flip_rate, mean_max_delta, n)


def _eval_single(job) -> Tuple[float, float, int, int, int, int, float]:
    opp, seed, position = job
    _diag_reset()
    agents = [_diagnostic_agent, opp] if position == 0 else [opp, _diagnostic_agent]
    _, rewards, ships = run_episode(agents, seed=seed)
    flips, turns, max_d_sum = _diag_snapshot()
    return (rewards[position], rewards[1 - position],
            ships[position], ships[1 - position],
            flips, turns, max_d_sum)


def _eval_worker(args) -> Tuple[float, float, int, int, int, int, float]:
    (opp, seed, position), payload = args
    main.MLP_W1 = payload["W1"]
    main.MLP_B1 = payload["B1"]
    main.MLP_W2 = payload["W2"]
    main.MLP_B2 = payload["B2"]
    main.FEATURE_MEAN = payload["mean"]
    main.FEATURE_STD = payload["std"]
    _diag_reset()
    agents = [_diagnostic_agent, opp] if position == 0 else [opp, _diagnostic_agent]
    _, rewards, ships = run_episode(agents, seed=seed)
    flips, turns, max_d_sum = _diag_snapshot()
    return (rewards[position], rewards[1 - position],
            ships[position], ships[1 - position],
            flips, turns, max_d_sum)


def save_weights_block(out_path: Path, policy: Policy,
                       mean: torch.Tensor, std: torch.Tensor) -> None:
    """Write the trained constants as a Python source block ready to paste
    into main.py (or used by --apply to replace the existing block)."""

    def fmt(values):
        return ", ".join(f"{v:.6f}" for v in values)

    w1 = policy.fc1.weight.detach().cpu().tolist()
    b1 = policy.fc1.bias.detach().cpu().tolist()
    w2 = policy.fc2.weight.detach().cpu().tolist()
    b2 = policy.fc2.bias.detach().cpu().tolist()

    lines = []
    lines.append(f"FEATURE_MEAN = ({fmt(mean.tolist())},)".replace(",,)", ",)"))
    lines.append(f"FEATURE_STD = ({fmt(std.tolist())},)".replace(",,)", ",)"))
    lines.append("MLP_W1 = (")
    for row in w1:
        lines.append(f"    ({fmt(row)},),".replace(",,),", ",),"))
    lines.append(")")
    lines.append(f"MLP_B1 = ({fmt(b1)},)".replace(",,)", ",)"))
    lines.append("MLP_W2 = (")
    lines.append(f"    ({fmt(w2[0])},),".replace(",,),", ",),"))
    lines.append(")")
    lines.append(f"MLP_B2 = ({fmt(b2)},)".replace(",,)", ",)"))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_block_to_main(block_path: Path, main_path: Path = ROOT / "main.py") -> None:
    """Replace the FEATURE_MEAN..MLP_B2 lines in main.py with the trained block."""
    src = main_path.read_text(encoding="utf-8")
    start = src.index("FEATURE_MEAN")
    end = src.index("MLP_B2 = (")
    end = src.index("\n", end + 1) + 1
    new = src[:start] + block_path.read_text(encoding="utf-8") + src[end:]
    main_path.write_text(new, encoding="utf-8")


# ----------------------------------------------------------------------------- Main loop
# -----------------------------------------------------------------------------


def parse_seed_range(spec: str) -> List[int]:
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


class _NoPool:
    """Context-manager stand-in when --workers=0 so the body works without
    branching on whether parallelism is enabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200,
                   help="Outer training steps (one batch per step).")
    p.add_argument("--batch", type=int, default=16, help="Episodes per training step.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel rollout processes. 0=serial; set to ~ncores for speed.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temp", type=float, default=3.0,
                   help="Softmax temperature for action sampling.")
    p.add_argument("--opponents", nargs="+", default=["self", "medium.py", "starter"],
                   help='"self" enables self-play (current policy on both sides).')
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-seeds", default="1..50",
                   help="Each seed is played twice (mirror match), so N = 2 * |seeds| * |opps|.")
    p.add_argument("--eval-opponents", nargs="+", default=["medium.py", "starter"])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="weights_block.txt")
    p.add_argument("--apply", action="store_true",
                   help="Rewrite main.py with the best snapshot at the end.")
    args = p.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    policy = Policy().to(device)
    mean = torch.zeros(main.N_FEATURES)
    std = torch.ones(main.N_FEATURES)
    sync_policy_to_main(policy, mean, std)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)
    eval_seeds = parse_seed_range(args.eval_seeds)

    pool_ctx = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 0 else _NoPool()
    with pool_ctx as pool:
        pool_obj = pool if args.workers > 0 else None

        print("Baseline (rule-only) eval...", flush=True)
        base = evaluate_greedy(args.eval_opponents, eval_seeds,
                               pool_obj, policy, mean, std)
        print(f"  baseline: win={base.win:.3f}+/-{base.win_ci:.3f} margin={base.margin:+d} "
              f"flips={base.flip_rate:.1%} |delta|max={base.mean_max_delta:.2f} "
              f"(N={base.n})", flush=True)

        best_win = base.win
        best_margin = base.margin
        best_state = (deepcopy(policy.state_dict()), mean.clone(), std.clone())

        running_baseline = 0.0
        for step in range(1, args.steps + 1):
            t0 = time.time()
            ep_traces, ep_rewards = collect_batch_parallel(
                pool_obj, policy, mean, std,
                batch_size=args.batch, opponents=args.opponents,
                temperature=args.temp, rng=rng,
            )
            # Filter each trace to player 0's decisions (the trainee). In self-play,
            # both players use the same policy but we use player 0's advantage.
            traces = [[d for d in t if d["player"] == 0] for t in ep_traces]
            rewards_by_player = [r[0] for r in ep_rewards]

            # Update feature stats (running, dominated by recent rollouts)
            new_mean, new_std = collect_feature_stats(traces)
            if step == 1:
                mean, std = new_mean, new_std
            else:
                mean = 0.9 * mean + 0.1 * new_mean
                std = 0.9 * std + 0.1 * new_std

            # Advantage = return - moving baseline. A pure batch-mean baseline
            # goes to zero when a small batch is unanimous, killing the
            # gradient; the moving baseline gives a non-zero signal even then.
            returns = torch.tensor(rewards_by_player, dtype=torch.float32)
            baseline_alpha = 0.05
            if step == 1:
                running_baseline = float(returns.mean().item())
            running_baseline = ((1 - baseline_alpha) * running_baseline
                                + baseline_alpha * float(returns.mean().item()))
            adv = returns - running_baseline

            decisions = []
            advantages = []
            for trace, a in zip(traces, adv.tolist()):
                for d in trace:
                    decisions.append(d)
                    advantages.append(a)

            if decisions:
                optim.zero_grad()
                loss = pg_loss(policy, mean, std, decisions, advantages,
                               device, temperature=args.temp)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()
                loss_val = float(loss.item())
            else:
                loss_val = 0.0

            sync_policy_to_main(policy, mean, std)

            wins = sum(1 for r in rewards_by_player if r > 0)
            elapsed = time.time() - t0
            print(f"[{step:4d}/{args.steps}] batch_win={wins}/{len(rewards_by_player)} "
                  f"return_mean={returns.mean().item():+.3f} loss={loss_val:+.4f} "
                  f"({elapsed:.1f}s)", flush=True)

            if step % args.eval_every == 0:
                ev = evaluate_greedy(args.eval_opponents, eval_seeds,
                                     pool_obj, policy, mean, std)
                tag = "  "
                if ev.win > best_win or (ev.win == best_win and ev.margin > best_margin):
                    best_win = ev.win
                    best_margin = ev.margin
                    best_state = (deepcopy(policy.state_dict()),
                                  mean.clone(), std.clone())
                    tag = "**"
                print(f"  {tag} eval: win={ev.win:.3f}+/-{ev.win_ci:.3f} margin={ev.margin:+d} "
                      f"flips={ev.flip_rate:.1%} |delta|max={ev.mean_max_delta:.2f} "
                      f"(best win={best_win:.3f})", flush=True)

    # Restore best snapshot and export
    policy.load_state_dict(best_state[0])
    best_mean, best_std = best_state[1], best_state[2]
    sync_policy_to_main(policy, best_mean, best_std)
    out_path = ROOT / args.out
    save_weights_block(out_path, policy, best_mean, best_std)
    print(f"\nBest eval win={best_win:.3f}, margin={best_margin:+d}")
    print(f"Saved weights block to {out_path}")
    if args.apply:
        apply_block_to_main(out_path)
        print("Rewrote main.py with the best snapshot.")


if __name__ == "__main__":
    main_cli()
