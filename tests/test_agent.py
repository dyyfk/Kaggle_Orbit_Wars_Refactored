"""Smoke tests for main.py.

These tests do NOT validate strategy — only that the agent loads, parses a
realistic observation, and emits actions Kaggle will accept. Strategy quality
is checked by simulator win-rate runs, not unit tests.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


def _planet(pid, owner, x, y, r=2.0, ships=10, prod=2):
    return [pid, owner, float(x), float(y), float(r), int(ships), int(prod)]


def _fleet(fid, owner, x, y, angle, from_pid, ships):
    return [fid, owner, float(x), float(y), float(angle), int(from_pid), int(ships)]


def _obs(planets, fleets=None, player=0, step=0):
    return {
        "planets": planets,
        "fleets": fleets or [],
        "player": player,
        "angular_velocity": 0.01,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
        "initial_planets": planets,
    }


def test_agent_with_no_targets_returns_empty():
    obs = _obs([_planet(0, 0, 20, 20), _planet(1, 0, 80, 80)])
    assert main.agent(obs) == []


def test_agent_with_no_owned_planets_returns_empty():
    obs = _obs([_planet(0, 1, 20, 20)], player=0)
    assert main.agent(obs) == []


def test_agent_emits_valid_action_shape():
    obs = _obs([
        _planet(0, 0, 20, 20, ships=50),
        _planet(1, -1, 60, 60, ships=5),
    ])
    moves = main.agent(obs)
    assert isinstance(moves, list)
    for m in moves:
        assert len(m) == 3
        from_id, angle, ships = m
        assert isinstance(from_id, int) and from_id == 0
        assert math.isfinite(angle)
        assert isinstance(ships, int) and ships > 0


def test_sanitize_drops_overdraw():
    state = main.parse_observation(_obs([_planet(0, 0, 20, 20, ships=10)]))
    # request more ships than the planet has
    cleaned = main.sanitize_moves([[0, 0.0, 999]], state)
    assert cleaned == []


def test_sanitize_drops_unowned_source():
    state = main.parse_observation(_obs([_planet(0, 1, 20, 20, ships=10)]))
    cleaned = main.sanitize_moves([[0, 0.0, 5]], state)
    assert cleaned == []


def test_path_crosses_sun_detects_blocked_line():
    # A segment passing through the center hits the sun.
    assert main.path_crosses_sun((0.0, 50.0), (100.0, 50.0))


def test_path_crosses_sun_clears_outer_path():
    # A segment along the bottom edge stays clear.
    assert not main.path_crosses_sun((0.0, 5.0), (100.0, 5.0))


def test_fleet_speed_monotonic_in_ships():
    speeds = [main.fleet_speed(n) for n in (1, 10, 100, 1000)]
    assert speeds == sorted(speeds)
    assert speeds[-1] <= main.DEFAULT_MAX_SPEED + 1e-9


def test_predict_planet_position_at_zero_is_current():
    state = main.parse_observation(_obs([_planet(0, 0, 30, 50)]))
    pos = main.predict_planet_position(state, state.planets[0], 0.0)
    assert pos is not None
    # Planet at (30, 50) has orbital radius 20 from center (50,50). With
    # angular velocity 0.01 and step 0, predicted position == current (30, 50).
    assert math.isclose(pos[0], 30.0, abs_tol=1e-6)
    assert math.isclose(pos[1], 50.0, abs_tol=1e-6)


def test_no_action_repeats_same_target():
    # Two friendly planets, one neutral target. We should pick the better
    # source, not blast both at the same target.
    obs = _obs([
        _planet(0, 0, 10, 10, ships=200),
        _planet(1, 0, 80, 80, ships=200),
        _planet(2, -1, 50, 10, ships=5),
    ])
    moves = main.agent(obs)
    targets = [int(m[0]) for m in moves]
    assert len(set(targets)) == len(targets), f"duplicate sources: {moves}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
