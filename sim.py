"""Local Orbit Wars runner.

Usage:
    python sim.py --agents main.py starter --seed 42 --out replays/seed42.html
    python sim.py --agents main.py medium.py --seeds 1 2 3 4 5 --jobs auto
"""

import argparse
import contextlib
import io
import json
import os
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def make_env(seed: int, debug: bool = True):
    _stub_optional_imports()
    from kaggle_environments import make
    return make("orbit_wars", configuration={"seed": seed}, debug=debug)


def run_episode(agents: Sequence[Any], seed: int = 42, debug: bool = True) -> Dict[str, Any]:
    env = make_env(seed, debug=debug)
    env.run(list(agents))
    return {"seed": seed, "summary": _summarize(env.steps[-1])}


def save_replay(agents: Sequence[Any], seed: int, out_path: str,
                width: int = 900, height: int = 700) -> Dict[str, Any]:
    env = make_env(seed)
    env.run(list(agents))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.render(mode="html", width=width, height=height), encoding="utf-8")
    return {"seed": seed, "replay": str(out), "summary": _summarize(env.steps[-1])}


def run_many(agents: Sequence[str], seeds: Iterable[int], jobs: str = "1") -> List[Dict[str, Any]]:
    seeds = list(seeds)
    if not seeds:
        return []
    specs = [_resolve(s) for s in agents]
    workers = _resolve_jobs(jobs, len(seeds))
    work = [(specs, s) for s in seeds]
    if workers == 1:
        return [_run_quiet(j) for j in work]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run_quiet, work))


def _run_quiet(job):
    agents, seed = job
    with contextlib.redirect_stdout(io.StringIO()):
        return run_episode(agents, seed=seed)


def _summarize(final_state) -> List[Dict[str, Any]]:
    rows = []
    if not final_state:
        return rows
    obs = _get(final_state[0], "observation", {})
    planets = _get(obs, "planets", []) or []
    fleets = _get(obs, "fleets", []) or []
    ships = [0] * len(final_state)
    for p in planets:
        owner = int(p[1])
        if 0 <= owner < len(ships):
            ships[owner] += int(p[5])
    for f in fleets:
        owner = int(f[1])
        if 0 <= owner < len(ships):
            ships[owner] += int(f[6])
    for i, st in enumerate(final_state):
        rows.append({
            "player": i,
            "reward": _get(st, "reward", None),
            "status": _get(st, "status", None),
            "ships": ships[i],
        })
    return rows


def _resolve(spec: str) -> str:
    if spec in {"random", "starter"}:
        return spec
    p = Path(spec)
    return str(p.resolve()) if p.exists() else spec


def _resolve_jobs(jobs: str, n: int) -> int:
    if jobs == "auto":
        return max(1, min(n, os.cpu_count() or 1))
    return max(1, int(jobs))


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_seeds(spec: str) -> List[int]:
    if ".." in spec:
        lo, hi = spec.split("..")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s]


def _stub_optional_imports() -> None:
    """Some Windows installs crash registering OpenSpiel envs we don't use."""
    if os.name != "nt" or "kaggle_environments" in sys.modules:
        return

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Orbit Wars agents locally.")
    p.add_argument("--agents", nargs="+", default=["main.py", "starter"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=_parse_seeds, default=None,
                   help='Comma list or "lo..hi" range, e.g. "1,2,5" or "1..100".')
    p.add_argument("--jobs", default="1", help="positive int or 'auto'")
    p.add_argument("--out", default="replays/seed42.html")
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--height", type=int, default=700)
    p.add_argument("--summary", action="store_true",
                   help="With --seeds, print aggregate win rate / ship totals instead of full JSON.")
    return p.parse_args()


def summarize_batch(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    players = len(results[0]["summary"])
    wins = [0] * players
    ships = [0] * players
    for r in results:
        rewards = [row["reward"] for row in r["summary"]]
        if all(x is not None for x in rewards):
            top = max(rewards)
            if rewards.count(top) == 1:
                wins[rewards.index(top)] += 1
        for i, row in enumerate(r["summary"]):
            ships[i] += row["ships"]
    return {
        "n": n,
        "win_rate": [round(w / n, 4) for w in wins],
        "wins": wins,
        "ships_total": ships,
    }


def main() -> None:
    args = _parse_args()
    if args.seeds:
        results = run_many(args.agents, args.seeds, args.jobs)
        if args.summary:
            print(json.dumps(summarize_batch(results), indent=2))
        else:
            print(json.dumps(results, indent=2))
        return
    agents = [_resolve(s) for s in args.agents]
    print(json.dumps(save_replay(agents, args.seed, args.out, args.width, args.height),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
