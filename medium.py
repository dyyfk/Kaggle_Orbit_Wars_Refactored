import math
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple


BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ORBIT_LIMIT = 50.0
MAX_SPEED = 6.0
EPISODE_STEPS = 500


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


def _agent_impl(obs, configuration=None):
    """Port of the browser playable Orbit Wars Medium bot."""
    planets = _parse_planets(_get(obs, "planets", []) or [])
    fleets = _parse_fleets(_get(obs, "fleets", []) or [])
    player = int(_get(obs, "player", 0) or 0)
    angular_velocity = float(
        _get(obs, "angularVelocity", _get(obs, "angular_velocity", 0.0)) or 0.0
    )
    step = int(_get(obs, "step", 0) or 0)
    remaining_turns = max(1, EPISODE_STEPS - step)
    my_planets = [planet for planet in planets if planet.owner == player]
    if not my_planets:
        return []

    pressure = estimate_owned_planet_pressure(fleets, planets, angular_velocity, player)
    committed = {planet.id: 0 for planet in planets}
    for fleet in fleets:
        if fleet.owner != player:
            continue
        speed = fleet_speed(fleet.ships)
        previous = (fleet.x, fleet.y)
        for turn in range(1, 90):
            current = (
                fleet.x + math.cos(fleet.angle) * speed * turn,
                fleet.y + math.sin(fleet.angle) * speed * turn,
            )
            if (
                current[0] < 0
                or current[0] > BOARD_SIZE
                or current[1] < 0
                or current[1] > BOARD_SIZE
                or point_to_segment_distance((CENTER, CENTER), previous, current) < SUN_RADIUS
            ):
                break

            hit_planet_id = None
            for planet in planets:
                planet_pos = predict_position(planet, angular_velocity, turn)
                if point_to_segment_distance(planet_pos, previous, current) < planet.radius + 0.5:
                    hit_planet_id = planet.id
                    break

            if hit_planet_id is not None:
                committed[hit_planet_id] = committed.get(hit_planet_id, 0) + fleet.ships
                break
            previous = current

    moves = []
    newly_committed: Dict[int, int] = {}
    sources = sorted(my_planets, key=lambda planet: planet.ships, reverse=True)
    for source in sources:
        if source.ships < 20:
            continue

        incoming_pressure = max(0, pressure.get(source.id, 0))
        best = None
        best_score = -1.0
        for target in planets:
            if target.id == source.id or target.owner == player:
                continue

            probe_ships = max(20, math.floor(source.ships / 2))
            intercept_pos, travel_time = intercept_position(
                (source.x, source.y),
                target,
                angular_velocity,
                probe_ships,
            )
            turns = max(1, math.ceil(travel_time))
            if turns > 80:
                continue

            angle = safe_launch_angle((source.x, source.y), intercept_pos)
            if angle is None:
                continue

            distance = euclidean_distance((source.x, source.y), intercept_pos)
            cost = (
                target.ships + 1
                if target.owner == -1
                else target.ships + target.production * turns + 1
            )
            already_sent = committed.get(target.id, 0) + newly_committed.get(target.id, 0)
            if target.owner == -1 and already_sent >= cost:
                continue

            score = (
                target.production
                * max(0, remaining_turns - turns)
                / (cost + 5)
                / (1 + 0.02 * turns)
            )
            if target.owner != -1:
                score *= 3
            score *= 30 / (30 + distance)
            if target.owner != -1 and target.owner != player and 0 < already_sent < cost:
                score *= 1.5
            if target.owner != -1 and target.owner != player:
                if distance < 25:
                    score *= 1.7
                elif distance < 40:
                    score *= 1.3

            if score > best_score:
                best_score = score
                best = (target, angle, cost)

        if best is None:
            continue

        target, angle, cost = best
        already_sent = committed.get(target.id, 0) + newly_committed.get(target.id, 0)
        wanted = max(20, cost - already_sent + 5)
        ships_to_send = max(wanted, math.floor(source.ships / 2))
        reserve = math.floor(incoming_pressure * 1.1) + 5
        capacity = max(0, source.ships - reserve)
        ships_to_send = min(ships_to_send, capacity)
        if ships_to_send < 20:
            continue

        ships_to_send = math.floor(ships_to_send)
        moves.append([source.id, angle, ships_to_send])
        newly_committed[target.id] = newly_committed.get(target.id, 0) + ships_to_send

    return moves


def estimate_owned_planet_pressure(
    fleets: Sequence[Fleet],
    planets: Sequence[Planet],
    angular_velocity: float,
    player: int,
) -> Dict[int, int]:
    pressure = {planet.id: 0 for planet in planets if planet.owner == player}
    for fleet in fleets:
        speed = fleet_speed(fleet.ships)
        previous = (fleet.x, fleet.y)
        for turn in range(1, 50):
            current = (
                fleet.x + math.cos(fleet.angle) * speed * turn,
                fleet.y + math.sin(fleet.angle) * speed * turn,
            )
            if (
                current[0] < 0
                or current[0] > BOARD_SIZE
                or current[1] < 0
                or current[1] > BOARD_SIZE
                or point_to_segment_distance((CENTER, CENTER), previous, current) < SUN_RADIUS
            ):
                break

            hit_planet = None
            for planet in planets:
                planet_pos = predict_position(planet, angular_velocity, turn)
                if point_to_segment_distance(planet_pos, previous, current) < planet.radius + 0.5:
                    hit_planet = planet
                    break

            if hit_planet is not None:
                if hit_planet.owner == player and fleet.owner != player:
                    pressure[hit_planet.id] = pressure.get(hit_planet.id, 0) + fleet.ships
                elif hit_planet.owner == player and fleet.owner == player:
                    pressure[hit_planet.id] = pressure.get(hit_planet.id, 0) - fleet.ships
                break
            previous = current
    return pressure


def intercept_position(
    source_pos: Tuple[float, float],
    target: Planet,
    angular_velocity: float,
    ships: int,
) -> Tuple[Tuple[float, float], float]:
    speed = fleet_speed(ships)
    target_pos = (target.x, target.y)
    if not is_orbiting(target):
        return target_pos, max(1, euclidean_distance(source_pos, target_pos) / speed)

    for _ in range(20):
        distance = euclidean_distance(source_pos, target_pos)
        travel_time = max(1, distance / speed)
        next_pos = predict_position(target, angular_velocity, travel_time)
        if euclidean_distance(next_pos, target_pos) < 0.1:
            target_pos = next_pos
            break
        target_pos = next_pos

    return target_pos, max(1, euclidean_distance(source_pos, target_pos) / speed)


def safe_launch_angle(
    source_pos: Tuple[float, float],
    target_pos: Tuple[float, float],
) -> Optional[float]:
    if point_to_segment_distance((CENTER, CENTER), source_pos, target_pos) >= SUN_RADIUS + 1.5:
        return math.atan2(target_pos[1] - source_pos[1], target_pos[0] - source_pos[0])
    return None


def fleet_speed(ships: int) -> float:
    if ships <= 1:
        return 1.0
    return min(
        MAX_SPEED,
        1 + (MAX_SPEED - 1) * math.pow(math.log(ships) / math.log(1000), 1.5),
    )


def is_orbiting(planet: Planet) -> bool:
    return math.hypot(planet.x - CENTER, planet.y - CENTER) + planet.radius < ORBIT_LIMIT


def rotate_position(pos: Tuple[float, float], angle: float) -> Tuple[float, float]:
    x = pos[0] - CENTER
    y = pos[1] - CENTER
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (
        CENTER + x * cos_angle - y * sin_angle,
        CENTER + x * sin_angle + y * cos_angle,
    )


def predict_position(
    planet: Planet,
    angular_velocity: float,
    turns: float,
) -> Tuple[float, float]:
    if is_orbiting(planet):
        return rotate_position((planet.x, planet.y), angular_velocity * turns)
    return (planet.x, planet.y)


def point_to_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    length_sq = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2
    if length_sq == 0:
        return euclidean_distance(point, start)

    t = (
        (point[0] - start[0]) * (end[0] - start[0])
        + (point[1] - start[1]) * (end[1] - start[1])
    ) / length_sq
    t = max(0, min(1, t))
    projection = (
        start[0] + t * (end[0] - start[0]),
        start[1] + t * (end[1] - start[1]),
    )
    return euclidean_distance(point, projection)


def euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_planets(raw_planets: Iterable[Sequence[Any]]) -> List[Planet]:
    return [
        Planet(
            id=int(planet[0]),
            owner=int(planet[1]),
            x=float(planet[2]),
            y=float(planet[3]),
            radius=float(planet[4]),
            ships=int(planet[5]),
            production=int(planet[6]),
        )
        for planet in raw_planets
        if len(planet) >= 7
    ]


def _parse_fleets(raw_fleets: Iterable[Sequence[Any]]) -> List[Fleet]:
    return [
        Fleet(
            id=int(fleet[0]),
            owner=int(fleet[1]),
            x=float(fleet[2]),
            y=float(fleet[3]),
            angle=float(fleet[4]),
            from_planet_id=int(fleet[5]),
            ships=int(fleet[6]),
        )
        for fleet in raw_fleets
        if len(fleet) >= 7
    ]


def _get(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def agent(obs, configuration=None):
    return _agent_impl(obs, configuration)
