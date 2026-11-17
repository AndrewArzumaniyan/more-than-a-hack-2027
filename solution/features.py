"""Tracking geometry and temporal context used by the shipped C6 model.

Runtime dependencies: NumPy, SciPy, and Python dataclasses only.
Extracted unchanged from the competition feature implementation.
"""


from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

H, A, D, X = 0, 1, 2, 3
FPS = 5


def mean_window(values, width, origin=0):
    finite = np.isfinite(values)
    numerator = uniform_filter1d(np.where(finite, values, 0.), width, origin=origin, mode="nearest")
    count = uniform_filter1d(finite.astype(np.float32), width, origin=origin, mode="nearest")
    return np.divide(numerator, count, out=np.full_like(numerator, np.nan), where=count > 1e-6)


def since(mask):
    index = np.arange(len(mask))
    return index - np.maximum.accumulate(np.where(mask, index, -1000))


def age_features(mask):
    return [np.minimum(since(mask), 100) / FPS, np.minimum(since(mask[::-1])[::-1], 100) / FPS]


@dataclass
class View:
    global_values: np.ndarray
    team_values: np.ndarray
    contact: np.ndarray
    speed: np.ndarray
    near: np.ndarray
    exit_margin: np.ndarray
    seen: np.ndarray
    player_speed: np.ndarray
    labels: np.ndarray

    def design(self, flip=False):
        order = (1, 0) if flip else (0, 1)
        left, right = self.team_values[:, order].transpose(1, 0, 2)
        difference = left - right
        return np.concatenate((self.global_values, left, right, difference), axis=1)


def features(half):
    players, ball = half.players, half.ball
    player_delta = np.diff(players, axis=0, prepend=players[:1]) * FPS
    ball_delta = np.diff(ball, axis=0, prepend=ball[:1]) * FPS
    player_speed = np.linalg.norm(player_delta, axis=2)
    ball_speed = np.linalg.norm(ball_delta, axis=1)
    player_valid = np.isfinite(players[:, :, 0])
    coverage = player_valid.sum(axis=1)
    seen = np.isfinite(ball).all(axis=1)
    depth = np.minimum(ball, half.pitch - ball).min(axis=1)
    motion = np.isfinite(player_speed)
    collective = np.divide(np.nansum(player_speed, axis=1), motion.sum(axis=1),
                           out=np.full(len(ball), np.nan), where=motion.sum(axis=1) > 0)
    fast = (player_speed > 2.5).sum(axis=1) / np.maximum(motion.sum(axis=1), 1)
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    nearest = np.where(np.isfinite(distance), distance, np.inf).min(axis=1)
    speed_before = np.r_[np.nan, ball_speed[:-1]]
    direction_before = np.r_[[[np.nan, np.nan]], ball_delta[:-1]]
    impulse = np.linalg.norm(ball_delta - direction_before, axis=1)
    jump = np.diff(ball_delta, axis=0, append=ball_delta[-1:])
    impulse = np.fmax(impulse, np.linalg.norm(jump, axis=1))
    collective_past = mean_window(collective, 21, origin=10)
    collective_next = mean_window(collective, 21, origin=-10)
    global_values = np.column_stack([
        seen, coverage, nearest, ball_speed, speed_before, impulse, depth,
        np.abs(ball[:, 0] - half.pitch[0] / 2), np.abs(ball[:, 1] - half.pitch[1] / 2),
        *age_features(seen), *age_features(depth < -.4), *age_features(seen & (ball_speed > 2)),
        collective, fast, collective_past, collective_next, collective_next - collective_past,
        *(mean_window(collective, width) for width in (5, 11, 25, 51)),
        *(mean_window(fast.astype(np.float32), width) for width in (5, 15, 35)),
        *(mean_window(seen.astype(np.float32), width) for width in (5, 15, 35, 65)),
        *(mean_window(ball_speed, width) for width in (5, 15, 35)),
    ])
    team_values, close, touches = [], [], []
    orientation = np.sign(np.nanmean(players[:, half.teams == A, 0]) -
                          np.nanmean(players[:, half.teams == H, 0]))
    for team in (H, A):
        pick = half.teams == team
        xy = players[:, pick]
        groups = player_valid[:, pick].sum(axis=1)
        team_dist = distance[:, pick]
        observed_dist = np.where(np.isfinite(team_dist), team_dist, np.inf)
        top = np.sort(observed_dist, axis=1)[:, :3]
        closest_slot = observed_dist.argmin(axis=1)
        row = np.arange(len(ball))
        relative = np.linalg.norm(player_delta[:, pick][row, closest_slot] - ball_delta, axis=1)
        var = np.nansum(np.square(xy - np.nansum(xy, axis=1, keepdims=True) /
                                    np.maximum(groups[:, None, None], 1)), axis=1)
        spread = np.sqrt(var / np.maximum(groups[:, None], 1))
        center = np.nansum(xy, axis=1) / np.maximum(groups[:, None], 1)
        attack = orientation * (1 - 2 * team)
        moving = np.diff(center[:, 0], prepend=center[:1, 0]) * attack * FPS
        mean_speed = np.nansum(player_speed[:, pick], axis=1) / np.maximum(motion[:, pick].sum(axis=1), 1)
        nearest = top[:, 0]
        contact = (nearest < 1.7) & (impulse > 3.0)
        controlled = (nearest < 1.7) & (relative < 4.0)
        closest = (nearest < 1.9) & seen
        close.append(nearest)
        touches.append(contact)
        team_values.append(np.column_stack([
            top, np.minimum(relative, 30.), groups, mean_speed, spread, moving,
            (center[:, 0] - half.pitch[0] / 2) * attack,
            (ball[:, 0] - center[:, 0]) * attack, nearest < 1.2, contact, controlled,
            (team_dist < 2.5).sum(axis=1), (team_dist < 5.).sum(axis=1), (team_dist < 9.).sum(axis=1),
            *age_features(closest), *age_features(contact), *age_features(controlled),
            *(mean_window(moving, width) for width in (5, 15, 35)),
            *(mean_window(closest.astype(np.float32), width) for width in (5, 15, 35, 65)),
            *(mean_window(contact.astype(np.float32), width) for width in (5, 15, 35)),
            *(mean_window(controlled.astype(np.float32), width) for width in (5, 15, 35)),
        ]))
    team_values = np.stack(team_values, axis=1)
    global_values[~np.isfinite(global_values)] = -1.
    team_values[~np.isfinite(team_values)] = -1.
    return View(global_values.astype(np.float32), team_values.astype(np.float32),
                np.stack(touches, axis=1), ball_speed, np.stack(close, axis=1), depth,
                seen, collective, half.labels)


def candidate_design(view):
    """Own/opponent geometry and fixed-window statistics for each team row."""
    n = len(view.labels)
    output = []
    nteam = view.team_values.shape[2]
    for candidate in (H, A):
        own = view.team_values[:, candidate]
        rival = view.team_values[:, 1 - candidate]
        difference = own - rival
        # Summaries cover history and future within a half, like offline Flow 1.
        temporal_base = np.column_stack((
            view.global_values[:, [2, 3, 6, 9, 10, 15, 16, 17, 18]],
            own[:, :3], rival[:, :3], own[:, 5], rival[:, 5],
            difference[:, [7, 8, 9, 10, 11, 12, 13]],
        ))
        context = []
        for width in (5, 15, 35):
            prev = uniform_filter1d(temporal_base, width, origin=width // 2, axis=0, mode="nearest")
            after = uniform_filter1d(temporal_base, width, origin=-(width // 2), axis=0, mode="nearest")
            context.extend((prev, after, after - prev))
        row = np.column_stack((view.global_values, own, rival, difference, *context))
        assert row.shape[0] == n and nteam > 0
        output.append(row.astype(np.float32))
    return np.stack(output, axis=1)
