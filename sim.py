"""Draw simulator: Elo win probabilities, bracket Monte Carlo, and random draws
under Grand Slam seeding rules. Used offline by build_data.py and at runtime by tools.py.
"""

import numpy as np

N_SIMS = 20_000

# --- Win probabilities ---

# Elo is fit mostly on best-of-3 matches. For best-of-5 we back out the per-set win
# probability s from p3 = s^2 (3 - 2s), then compute p5 = s^3 (10 - 15s + 6s^2).
_S_GRID = np.linspace(0, 1, 10_001)
_P3_GRID = _S_GRID**2 * (3 - 2 * _S_GRID)


def bo5_from_bo3(p3):
    s = np.interp(p3, _P3_GRID, _S_GRID)
    return s**3 * (10 - 15 * s + 6 * s**2)


# Off on purpose: on 5,715 men's Slam matches (2015-2026) the best-of-5 adjustment had a
# worse Brier score (0.1833) than plain Elo (0.1817). See the model_checks table.
BO5_ADJUST = False


def win_prob(elo_a, elo_b, best_of=3):
    p = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    return bo5_from_bo3(p) if (best_of == 5 and BO5_ADJUST) else p


# --- Bracket simulation ---


def round_names(size: int) -> list[str]:
    """Names of 'reached round k' for k = 0..log2(size). Index 0 = in the draw, last = won it."""
    n_rounds = int(np.log2(size))
    names = []
    for k in range(n_rounds):
        remaining = size >> k
        names.append({8: "QF", 4: "SF", 2: "F"}.get(remaining, f"R{remaining}"))
    return names + ["W"]


def simulate(elo: np.ndarray, slots: np.ndarray, best_of: int, n: int = N_SIMS, seed: int = 0) -> np.ndarray:
    """Play the bracket n times.

    elo:   (P,) rating per player index
    slots: (size,) one fixed draw, or (n, size) one draw per simulation, as player indices
    Returns reach: (P, rounds+1) probability each player reaches each round.
    """
    rng = np.random.default_rng(seed)
    if slots.ndim == 1:
        slots = np.broadcast_to(slots, (n, slots.shape[0]))
    P = len(elo)
    n_rounds = int(np.log2(slots.shape[1]))
    reach = np.zeros((P, n_rounds + 1))
    cur = slots
    reach[:, 0] = np.bincount(cur.ravel(), minlength=P)
    for k in range(1, n_rounds + 1):
        a, b = cur[:, 0::2], cur[:, 1::2]
        p = win_prob(elo[a], elo[b], best_of)
        cur = np.where(rng.random(p.shape) < p, a, b)
        reach[:, k] = np.bincount(cur.ravel(), minlength=P)
    return reach / n


# --- Random draws under seeding rules ---


def standard_order(size: int) -> list[int]:
    """Seed 'line' at each slot of a standard bracket: [1, 128, 65, 64, ...] for 128."""
    order = [1]
    while len(order) < size:
        m = len(order) * 2
        order = [x for s in order for x in (s, m + 1 - s)]
    return order


def seed_groups(n_seeds: int) -> list[tuple[int, int]]:
    """Line ranges that are drawn together: 1 | 2 | 3-4 | 5-8 | 9-16 | 17-32 ..."""
    groups, lo = [(1, 1), (2, 2)], 3
    while lo <= n_seeds:
        groups.append((lo, 2 * (lo - 1)))
        lo = 2 * (lo - 1) + 1
    return [g for g in groups if g[0] <= n_seeds]


def random_draws(seeded: np.ndarray, unseeded: np.ndarray, size: int, n: int = N_SIMS, seed: int = 1) -> np.ndarray:
    """n random draws that obey Slam seeding: seeds 1 and 2 in opposite halves,
    3-4 drawn into the remaining quarters, 5-8 into eighths, and so on. Unseeded
    players fill the rest at random.

    seeded:   player indices ordered best seed first
    unseeded: player indices
    """
    rng = np.random.default_rng(seed)
    lines = np.array(standard_order(size))
    slot_of_line = np.empty(size + 1, dtype=int)
    slot_of_line[lines] = np.arange(size)

    out = np.empty((n, size), dtype=int)
    filled = np.zeros((n, size), dtype=bool)
    rows = np.arange(n)[:, None]
    S = len(seeded)
    for lo, hi in seed_groups(S):
        group_slots = slot_of_line[np.arange(lo, min(hi, size) + 1)]
        players = seeded[lo - 1 : min(hi, S)]
        pick = np.argsort(rng.random((n, len(group_slots))), axis=1)[:, : len(players)]
        chosen = group_slots[pick]
        out[rows, chosen] = players
        filled[rows, chosen] = True

    perm = np.argsort(rng.random((n, len(unseeded))), axis=1)
    out[~filled] = unseeded[perm].ravel()
    return out


# --- Draw luck ---


def draw_luck(elo, real_slots, seeds, best_of, n=N_SIMS):
    """Compare each player's chances in the real draw vs. the average random draw.

    elo:        (size,) ratings, indexed by player index 0..size-1
    real_slots: (size,) player index in each slot of the real draw
    seeds:      (size,) seed per player index, 0 if unseeded
    Returns (real_reach, random_reach), each (size, rounds+1).
    """
    seeds = np.asarray(seeds)
    idx = np.arange(len(elo))
    seeded = idx[seeds > 0][np.argsort(seeds[seeds > 0])]
    unseeded = idx[seeds == 0]
    real = simulate(elo, real_slots, best_of, n, seed=0)
    rand = simulate(elo, random_draws(seeded, unseeded, len(real_slots), n), best_of, n, seed=0)
    return real, rand
