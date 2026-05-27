"""Orbit Wars Kaggle submission.

Rule-based agent. Single file, no external dependencies. The `agent` function
at the bottom is what Kaggle calls each turn.

Per-turn pipeline:
    parse observation
    -> compute defensive reserves
    -> spend reserved budget on urgent defense first
    -> generate (source, target) attack candidates with iterated aim
    -> score candidates; greedily pick the highest until budget runs out
    -> sanitize and return moves
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


# Board geometry
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ORBIT_LIMIT = 50.0
DEFAULT_MAX_SPEED = 6.0
NEUTRAL = -1

# Strategy knobs
MAX_MOVES_PER_TURN = 12
DEFENSE_WINDOW = 30.0
ENEMY_MIN_PACKET = 20
ENEMY_BUDGET_FRACTION = 0.35
NEUTRAL_OPENING_STEP = 30
NEUTRAL_LONG_FLIGHT_TURNS = 12.0
NEUTRAL_LONG_FLIGHT_MIN_PACKET = 12

# Score weights for the rule-based component.
W_NEUTRAL_BONUS = 12.0
W_ENEMY_BONUS = 20.0
W_NEUTRAL_COST = 1.10
W_ENEMY_COST = 1.25
W_TRAVEL = 1.15
W_EXPANSION = 18.0
W_EXPANSION_DECAY = 0.25
W_STATIC = 4.0
W_COMET = 10.0
COMET_USEFUL_MIN_TURNS = 8

# How strongly the learned MLP delta can move a candidate's final score, in
# rule-score units. Rule scores are typically of magnitude 50-200; this limit
# is large enough that the MLP can actually flip the argmax once trained, but
# bounded so a single noisy candidate can't dominate the budget. With all-zero
# weights the MLP outputs zero and behavior matches rule-only.
MLP_DELTA_LIMIT = 100.0
FEATURE_NAMES = (
    "target_production",
    "target_ships",
    "ships_needed",
    "travel_time",
    "is_neutral",
    "is_enemy",
    "is_comet",
    "source_available_ships",
    "distance",
    "friendly_support",
    "my_production",
    "enemy_production",
    "my_planets",
    "enemy_planets",
    "step",
    "rule_score",
)
N_FEATURES = len(FEATURE_NAMES)


class Planet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


class Fleet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass(frozen=True)
class GameState:
    player: int
    planets: List[Planet]
    fleets: List[Fleet]
    initial_planets: Dict[int, Planet]
    angular_velocity: float
    comet_planet_ids: set
    comets: List
    step: int
    max_speed: float
    sun_radius: float


@dataclass(frozen=True)
class IncomingFleet:
    planet_id: int
    owner: int
    ships: int
    turns: float


@dataclass(frozen=True)
class Candidate:
    source_id: int
    target_id: int
    angle: float
    ships: int
    score: float
    rule_score: float = 0.0
    features: Tuple[float, ...] = ()


def plan_turn(state: GameState) -> List[List[float]]:
    moves, _ = plan_turn_with_candidates(state)
    return moves


def plan_turn_with_candidates(state: GameState) -> Tuple[List[List[float]], List["Candidate"]]:
    """Return (moves, candidates_considered). Candidates are exposed so the
    training script can collect features and selected indices per turn."""
    my_planets = [p for p in state.planets if p.owner == state.player]
    targets = [p for p in state.planets if p.owner != state.player]
    if not my_planets or not targets:
        return [], []

    incoming = estimate_incoming_fleets(state, my_planets)
    reserves = compute_reserves(my_planets, incoming)
    available = {p.id: max(0, p.ships - reserves.get(p.id, 0)) for p in my_planets}

    moves = build_defense_moves(state, my_planets, incoming, available)

    candidates = build_attack_candidates(state, my_planets, targets, available, incoming)
    claimed: set = set()
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if len(moves) >= MAX_MOVES_PER_TURN or c.score <= 0:
            break
        if c.target_id in claimed:
            continue
        if c.ships <= 0 or c.ships > available.get(c.source_id, 0):
            continue
        moves.append([c.source_id, c.angle, int(c.ships)])
        available[c.source_id] -= c.ships
        claimed.add(c.target_id)

    return sanitize_moves(moves, state), candidates


def parse_observation(obs, configuration=None) -> GameState:
    raw_planets = _get(obs, "planets", []) or []
    raw_fleets = _get(obs, "fleets", []) or []
    raw_initial = _get(obs, "initial_planets", raw_planets) or raw_planets

    planets = [Planet(*p) for p in raw_planets if len(p) >= 7]
    fleets = [Fleet(*f) for f in raw_fleets if len(f) >= 7]
    initial = {int(p[0]): Planet(*p) for p in raw_initial if len(p) >= 7}

    return GameState(
        player=int(_get(obs, "player", 0) or 0),
        planets=planets,
        fleets=fleets,
        initial_planets=initial,
        angular_velocity=float(_get(obs, "angular_velocity", 0.0) or 0.0),
        comet_planet_ids=set(_get(obs, "comet_planet_ids", []) or []),
        comets=list(_get(obs, "comets", []) or []),
        step=int(_get(obs, "step", 0) or 0),
        max_speed=float(_cfg(configuration, "shipSpeed", DEFAULT_MAX_SPEED)),
        sun_radius=float(_cfg(configuration, "sunRadius", SUN_RADIUS)),
    )


def compute_reserves(
    my_planets: Sequence[Planet],
    incoming: Dict[int, List[IncomingFleet]],
) -> Dict[int, int]:
    reserves: Dict[int, int] = {}
    total_production = sum(p.production for p in my_planets)
    for planet in my_planets:
        base_keep = max(2, planet.production * 2)
        threats = [f for f in incoming.get(planet.id, []) if f.turns <= DEFENSE_WINDOW]
        if total_production <= 18 and planet.production >= 4 and threats:
            base_keep += planet.production
        if not threats:
            reserves[planet.id] = base_keep
            continue

        earliest = min(f.turns for f in threats)
        incoming_ships = sum(f.ships for f in threats)
        produced_before_hit = planet.production * max(0, int(math.floor(earliest)))
        needed_now = incoming_ships + defense_margin(planet) - produced_before_hit
        reserves[planet.id] = max(base_keep, int(math.ceil(needed_now)))
    return reserves


def build_defense_moves(
    state: GameState,
    my_planets: Sequence[Planet],
    incoming: Dict[int, List[IncomingFleet]],
    available: Dict[int, int],
) -> List[List[float]]:
    planets_by_id = {p.id: p for p in my_planets}
    threatened = []
    for planet in my_planets:
        threats = [f for f in incoming.get(planet.id, []) if f.turns <= DEFENSE_WINDOW]
        if not threats:
            continue
        earliest = min(f.turns for f in threats)
        incoming_ships = sum(f.ships for f in threats)
        future_garrison = planet.ships + planet.production * max(0, int(math.floor(earliest)))
        need = incoming_ships + defense_margin(planet) - future_garrison
        if need > 0:
            threatened.append((planet.production, earliest, planet.id, int(math.ceil(need))))

    moves: List[List[float]] = []
    for _, earliest, target_id, need in sorted(threatened, reverse=True):
        target = planets_by_id[target_id]
        donors = [p for p in my_planets if p.id != target_id and available.get(p.id, 0) > 0]
        donors.sort(key=lambda p: distance((p.x, p.y), (target.x, target.y)))

        for donor in donors:
            if need <= 0 or len(moves) >= MAX_MOVES_PER_TURN:
                break
            send = min(need, available.get(donor.id, 0))
            if send <= 0:
                continue

            travel_time = estimate_travel_time(donor, (target.x, target.y), send, state.max_speed)
            if travel_time > earliest + 1.0:
                continue
            target_pos = predict_planet_position(state, target, travel_time)
            if target_pos is None or path_crosses_sun((donor.x, donor.y), target_pos, state.sun_radius):
                continue

            angle = math.atan2(target_pos[1] - donor.y, target_pos[0] - donor.x)
            moves.append([donor.id, angle, int(send)])
            available[donor.id] -= send
            need -= send

    return moves


def build_attack_candidates(
    state: GameState,
    my_planets: Sequence[Planet],
    targets: Sequence[Planet],
    available: Dict[int, int],
    incoming: Optional[Dict[int, List[IncomingFleet]]] = None,
) -> List[Candidate]:
    friendly = estimate_friendly_target_support(state, targets)
    incoming = incoming or {}
    board = _board_summary(state, my_planets, targets)
    out: List[Candidate] = []

    for src in my_planets:
        budget = available.get(src.id, 0)
        if budget <= 0:
            continue
        for tgt in targets:
            if tgt.id == src.id:
                continue
            if tgt.owner != NEUTRAL and budget < ENEMY_MIN_PACKET:
                continue
            cand = _build_one_candidate(
                state, src, tgt, budget, friendly.get(tgt.id, 0), available, board,
            )
            if cand is not None:
                out.append(cand)
    return out


def _board_summary(
    state: GameState,
    my_planets: Sequence[Planet],
    targets: Sequence[Planet],
) -> Dict[str, float]:
    enemy_planets = [p for p in targets if p.owner != NEUTRAL]
    return {
        "my_production": float(sum(p.production for p in my_planets)),
        "enemy_production": float(sum(p.production for p in enemy_planets)),
        "my_planets": float(len(my_planets)),
        "enemy_planets": float(len(enemy_planets)),
        "step": float(state.step),
    }


def _build_one_candidate(
    state: GameState,
    src: Planet,
    tgt: Planet,
    budget: int,
    friendly_support: int,
    available: Dict[int, int],
    board: Dict[str, float],
) -> Optional[Candidate]:
    # ships_needed, travel_time, and target_pos form a fixed point: each
    # depends on the others. Three coupled passes are enough in practice.
    target_pos = (tgt.x, tgt.y)
    ships_needed = estimate_capture_cost(state, tgt, 0.0)
    if friendly_support >= ships_needed:
        return None
    ships_needed = enemy_packet_floor(tgt, max(1, ships_needed - friendly_support), budget)
    travel_time = estimate_travel_time(src, target_pos, ships_needed, state.max_speed)

    # Pass 2: aim at the predicted future position.
    target_pos = predict_planet_position(state, tgt, travel_time)
    if target_pos is None:
        return None
    ships_needed = estimate_capture_cost(state, tgt, travel_time)
    if friendly_support >= ships_needed:
        return None
    ships_needed = enemy_packet_floor(tgt, max(1, ships_needed - friendly_support), budget)
    travel_time = estimate_travel_time(src, target_pos, ships_needed, state.max_speed)
    target_pos = predict_planet_position(state, tgt, travel_time)
    if target_pos is None:
        return None

    # Pass 3: apply the long-flight floor for neutrals and re-aim once more.
    ships_needed = estimate_capture_cost(state, tgt, travel_time)
    if friendly_support >= ships_needed:
        return None
    ships_needed = max(1, ships_needed - friendly_support)
    ships_needed = enemy_packet_floor(tgt, ships_needed, budget)
    ships_needed = neutral_long_flight_floor(state, tgt, ships_needed, travel_time)
    if ships_needed <= 0 or ships_needed > budget:
        return None
    travel_time = estimate_travel_time(src, target_pos, ships_needed, state.max_speed)
    target_pos = predict_planet_position(state, tgt, travel_time)
    if target_pos is None:
        return None
    travel_time = estimate_travel_time(src, target_pos, ships_needed, state.max_speed)

    if path_crosses_sun((src.x, src.y), target_pos, state.sun_radius):
        return None
    angle = math.atan2(target_pos[1] - src.y, target_pos[0] - src.x)
    if not math.isfinite(angle):
        return None

    rule = score_candidate(state, src, tgt, ships_needed, travel_time)
    features = _featurize(state, src, tgt, ships_needed, travel_time,
                          friendly_support, available, board, rule)
    score = rule + mlp_score(features)
    return Candidate(src.id, tgt.id, angle, int(ships_needed), score,
                     rule_score=rule, features=features)


def _featurize(
    state: GameState,
    src: Planet,
    tgt: Planet,
    ships_needed: int,
    travel_time: float,
    friendly_support: int,
    available: Dict[int, int],
    board: Dict[str, float],
    rule_score: float,
) -> Tuple[float, ...]:
    return (
        float(tgt.production),
        float(tgt.ships),
        float(ships_needed),
        float(travel_time),
        1.0 if tgt.owner == NEUTRAL else 0.0,
        1.0 if tgt.owner != NEUTRAL and tgt.owner != state.player else 0.0,
        1.0 if tgt.id in state.comet_planet_ids else 0.0,
        float(available.get(src.id, 0)),
        distance((src.x, src.y), (tgt.x, tgt.y)),
        float(friendly_support),
        board["my_production"],
        board["enemy_production"],
        board["my_planets"],
        board["enemy_planets"],
        board["step"],
        float(rule_score),
    )


def estimate_capture_cost(state: GameState, target: Planet, travel_time: float) -> int:
    growth = target.production * max(0, int(math.ceil(travel_time))) if target.owner != NEUTRAL else 0
    margin = 1
    if target.owner != NEUTRAL:
        margin += max(2, target.production * 2)
    if target.id in state.comet_planet_ids:
        margin += 1
    return int(max(1, math.ceil(target.ships + growth + margin)))


def enemy_packet_floor(target: Planet, ships_needed: int, source_budget: int) -> int:
    """For enemy targets, ensure a minimum-sized strike packet."""
    if target.owner == NEUTRAL:
        return ships_needed
    budget_packet = int(math.ceil(source_budget * ENEMY_BUDGET_FRACTION))
    return max(ships_needed, ENEMY_MIN_PACKET, budget_packet)


def neutral_long_flight_floor(
    state: GameState,
    target: Planet,
    ships_needed: int,
    travel_time: float,
) -> int:
    """Bump tiny packets on distant neutrals after the opening to avoid getting
    sniped by opponents during a long flight."""
    if (
        target.owner == NEUTRAL
        and state.step >= NEUTRAL_OPENING_STEP
        and travel_time > NEUTRAL_LONG_FLIGHT_TURNS
        and ships_needed < NEUTRAL_LONG_FLIGHT_MIN_PACKET
    ):
        return NEUTRAL_LONG_FLIGHT_MIN_PACKET
    return ships_needed


def score_candidate(
    state: GameState,
    source: Planet,
    target: Planet,
    ships_needed: int,
    travel_time: float,
) -> float:
    remaining_life = comet_remaining_life(state, target)
    if remaining_life is not None:
        useful_turns = remaining_life - travel_time
        if useful_turns < COMET_USEFUL_MIN_TURNS:
            return -999.0
        horizon = min(50.0, useful_turns)
    else:
        horizon = 80.0

    closest_my = min(
        distance((target.x, target.y), (p.x, p.y))
        for p in state.planets
        if p.owner == state.player
    )
    expansion_bonus = max(0.0, W_EXPANSION - closest_my * W_EXPANSION_DECAY)
    static_bonus = W_STATIC if not is_orbiting(state, target) else 0.0
    comet_bonus = W_COMET if target.id in state.comet_planet_ids else 0.0

    if target.owner == NEUTRAL:
        owner_bonus = W_NEUTRAL_BONUS
        cost_weight = W_NEUTRAL_COST
    else:
        owner_bonus = W_ENEMY_BONUS
        cost_weight = W_ENEMY_COST

    return (
        target.production * horizon
        + owner_bonus
        + expansion_bonus
        + static_bonus
        + comet_bonus
        - ships_needed * cost_weight
        - travel_time * W_TRAVEL
    )


def estimate_incoming_fleets(
    state: GameState,
    my_planets: Sequence[Planet],
) -> Dict[int, List[IncomingFleet]]:
    incoming: Dict[int, List[IncomingFleet]] = {}
    if not my_planets:
        return incoming
    for fleet in state.fleets:
        if fleet.owner == state.player:
            continue
        hit = likely_hit_planet_over_time(fleet, my_planets, state)
        if hit is None:
            continue
        planet, turns = hit
        incoming.setdefault(planet.id, []).append(
            IncomingFleet(planet.id, fleet.owner, fleet.ships, turns)
        )
    return incoming


def estimate_friendly_target_support(
    state: GameState,
    targets: Sequence[Planet],
) -> Dict[int, int]:
    support: Dict[int, int] = {}
    enemy_targets = [p for p in targets if p.owner != state.player]
    if not enemy_targets:
        return support
    for fleet in state.fleets:
        if fleet.owner != state.player:
            continue
        hit = likely_hit_planet_over_time(fleet, enemy_targets, state)
        if hit is None:
            continue
        planet, _ = hit
        support[planet.id] = support.get(planet.id, 0) + fleet.ships
    return support


def likely_hit_planet_over_time(
    fleet: Fleet,
    planets: Sequence[Planet],
    state: GameState,
    max_turns: float = 80.0,
) -> Optional[Tuple[Planet, float]]:
    speed = fleet_speed(fleet.ships, state.max_speed)
    ux = math.cos(fleet.angle)
    uy = math.sin(fleet.angle)
    previous = (fleet.x, fleet.y)

    for turn in range(1, int(math.ceil(max_turns)) + 1):
        current = (fleet.x + ux * speed * turn, fleet.y + uy * speed * turn)
        if (
            current[0] < 0.0
            or current[0] > BOARD_SIZE
            or current[1] < 0.0
            or current[1] > BOARD_SIZE
            or path_crosses_sun(previous, current, state.sun_radius, margin=0.0)
        ):
            break
        for planet in planets:
            planet_pos = predict_planet_position(state, planet, turn)
            if planet_pos is None:
                continue
            if point_to_segment_distance(planet_pos, previous, current) <= planet.radius + 0.5:
                return planet, float(turn)
        previous = current
    return None


def predict_planet_position(
    state: GameState,
    planet: Planet,
    turns_from_now: float,
) -> Optional[Tuple[float, float]]:
    comet_pos = predict_comet_position(state, planet, turns_from_now)
    if comet_pos is not None:
        return comet_pos
    if planet.id in state.comet_planet_ids:
        return None

    initial = state.initial_planets.get(planet.id)
    if initial is None:
        return (planet.x, planet.y)

    orbital_radius = distance((initial.x, initial.y), (CENTER, CENTER))
    if orbital_radius + initial.radius >= ORBIT_LIMIT:
        return (planet.x, planet.y)

    initial_angle = math.atan2(initial.y - CENTER, initial.x - CENTER)
    future_step = state.step + max(0.0, turns_from_now)
    angle = initial_angle + state.angular_velocity * future_step
    return (
        CENTER + orbital_radius * math.cos(angle),
        CENTER + orbital_radius * math.sin(angle),
    )


def predict_comet_position(
    state: GameState,
    planet: Planet,
    turns_from_now: float,
) -> Optional[Tuple[float, float]]:
    for group in state.comets:
        planet_ids = list(_get(group, "planet_ids", []) or [])
        if planet.id not in planet_ids:
            continue
        paths = _get(group, "paths", []) or []
        local_index = planet_ids.index(planet.id)
        if local_index >= len(paths):
            return None
        path_index = int(_get(group, "path_index", 0) or 0)
        future_index = path_index + max(0, int(math.ceil(turns_from_now)))
        if future_index >= len(paths[local_index]):
            return None
        point = paths[local_index][future_index]
        return (float(point[0]), float(point[1]))
    return None


def comet_remaining_life(state: GameState, planet: Planet) -> Optional[int]:
    if planet.id not in state.comet_planet_ids:
        return None
    for group in state.comets:
        planet_ids = list(_get(group, "planet_ids", []) or [])
        if planet.id not in planet_ids:
            continue
        paths = _get(group, "paths", []) or []
        local_index = planet_ids.index(planet.id)
        if local_index >= len(paths):
            return 0
        path_index = int(_get(group, "path_index", 0) or 0)
        return max(0, len(paths[local_index]) - path_index - 1)
    return 0


def is_orbiting(state: GameState, planet: Planet) -> bool:
    initial = state.initial_planets.get(planet.id, planet)
    return distance((initial.x, initial.y), (CENTER, CENTER)) + initial.radius < ORBIT_LIMIT


def estimate_travel_time(
    source: Planet,
    target_pos: Tuple[float, float],
    ships: int,
    max_speed: float,
) -> float:
    return distance((source.x, source.y), target_pos) / fleet_speed(ships, max_speed)


def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SPEED) -> float:
    ships = max(1, int(ships))
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(speed, max_speed)


def path_crosses_sun(
    start: Tuple[float, float],
    end: Tuple[float, float],
    sun_radius: float = SUN_RADIUS,
    margin: float = 0.20,
) -> bool:
    return point_to_segment_distance((CENTER, CENTER), start, end) <= sun_radius + margin


def point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    sx, sy = start
    ex, ey = end
    px, py = point
    dx, dy = ex - sx, ey - sy
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return distance(point, start)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
    return distance(point, (sx + t * dx, sy + t * dy))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def defense_margin(planet: Planet) -> int:
    return max(2, planet.production * 2)


def sanitize_moves(moves: Iterable[Sequence[float]], state: GameState) -> List[List[float]]:
    owned = {p.id: p for p in state.planets if p.owner == state.player}
    spent: Dict[int, int] = {}
    clean: List[List[float]] = []
    for move in moves:
        if len(move) != 3:
            continue
        from_id, angle, ships = move
        from_id = int(from_id)
        ships = int(ships)
        if from_id not in owned or ships <= 0 or not math.isfinite(float(angle)):
            continue
        if spent.get(from_id, 0) + ships > owned[from_id].ships:
            continue
        spent[from_id] = spent.get(from_id, 0) + ships
        clean.append([from_id, float(angle), ships])
    return clean


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg(configuration, key, default):
    if configuration is None:
        return default
    return _get(configuration, key, default)


# ---------------------------------------------------------------------------
# Learned MLP scorer (pure-Python inference, ~577 params).
#
# Architecture: 16 features -> 32 hidden (tanh) -> 1 (scaled tanh).
# Output is bounded in [-MLP_DELTA_LIMIT, +MLP_DELTA_LIMIT] and added to the
# rule score. With FEATURE_STD all ones and all weights zero, mlp_score()
# returns 0.0 and the agent behaves exactly like the rule-only refactor.
#
# Train via train.py. After training, paste the printed BLOCK below.
# ---------------------------------------------------------------------------


def mlp_score(features: Sequence[float]) -> float:
    z = [(f - m) / s for f, m, s in zip(features, FEATURE_MEAN, FEATURE_STD)]
    hidden = [
        math.tanh(b + sum(w * x for w, x in zip(row, z)))
        for row, b in zip(MLP_W1, MLP_B1)
    ]
    raw = MLP_B2[0] + sum(w * x for w, x in zip(MLP_W2[0], hidden))
    return MLP_DELTA_LIMIT * math.tanh(raw / MLP_DELTA_LIMIT)


# Z-score normalization. Defaults of (0, 1) leave features unscaled. train.py
# overwrites these with the per-feature mean/std from the training rollouts.
FEATURE_MEAN: Tuple[float, ...] = (0.0,) * 16
FEATURE_STD: Tuple[float, ...] = (1.0,) * 16

# Hidden layer: 32 rows of 16 weights, plus 32 biases.
MLP_W1: Tuple[Tuple[float, ...], ...] = tuple((0.0,) * 16 for _ in range(32))
MLP_B1: Tuple[float, ...] = (0.0,) * 32

# Output layer: 1 row of 32 weights, plus 1 bias.
MLP_W2: Tuple[Tuple[float, ...], ...] = ((0.0,) * 32,)
MLP_B2: Tuple[float, ...] = (0.0,)


# Kaggle's file loader picks the LAST top-level function as the agent. Keep
# `agent` at the bottom so it stays the entry point.
def agent(obs, configuration=None):
    state = parse_observation(obs, configuration)
    return plan_turn(state)
