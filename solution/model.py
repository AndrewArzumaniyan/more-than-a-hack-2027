"""Portable inference-only native C6 with frozen 358-channel team context.

Inputs use 5 Hz tracking; output probabilities are ordered H, A, D. Context
normalization moments are loaded from the checkpoint, never fitted at inference.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

FPS = 5
H, A, D = range(3)


class GraphDirection(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width * 4 + 9, width), nn.GELU(), nn.Linear(width, width))
        self.aggregate = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.gru = nn.GRUCell(width * 2, width)

    def forward(self, e, x, mask, reverse=False):
        b, t, n, w = e.shape
        h = e.new_zeros(b, n, w)
        centre = t // 2
        steps = range(t - 1, centre - 1, -1) if reverse else range(centre + 1)
        eye = ~torch.eye(n, device=e.device, dtype=torch.bool)[None]
        for k in steps:
            valid = mask[:, k]
            h = h * valid[..., None]
            eh = torch.cat((e[:, k], h), -1)
            a = eh[:, :, None].expand(-1, -1, n, -1)
            z = eh[:, None, :].expand(-1, n, -1, -1)
            dx = x[:, k, :, None] - x[:, k, None, :]
            edges = valid[:, :, None] & valid[:, None, :] & eye
            messages = self.message(torch.cat((a, z, dx), -1))
            context = self.aggregate((messages * edges[..., None]).sum(2))
            h = self.gru(torch.cat((e[:, k], context), -1).reshape(-1, 2 * w), h.reshape(-1, w)).reshape(b, n, w)
            h = h * valid[..., None]
        return h

class GraphRNN(nn.Module):

    def __init__(self, feature_count=9, width=32, state_count=5):
        super().__init__()
        if feature_count != 9:
            raise ValueError('GraphDirection expects the frozen nine-channel schema')
        self.embed = nn.Sequential(nn.Linear(feature_count, width), nn.LayerNorm(width), nn.GELU())
        self.types = nn.Embedding(3, width)
        self.forward_graph = GraphDirection(width)
        self.backward_graph = GraphDirection(width)
        self.head = nn.Sequential(nn.Linear(width * 6, width * 2), nn.GELU(), nn.Linear(width * 2, state_count))

    def forward(self, x, mask, types):
        e = (self.embed(x) + self.types(types)[:, None]) * mask[..., None]
        z = torch.cat((self.forward_graph(e, x, mask), self.backward_graph(e, x, mask, True)), -1)
        centre_mask = mask[:, x.shape[1] // 2]
        pooled = []
        for kind in range(3):
            take = (types == kind) & centre_mask
            pooled.append((z * take[..., None]).sum(1) / take.sum(1, keepdim=True).clamp_min(1))
        return self.head(torch.cat(pooled, -1))

DISTANCE_SCALE = 120.0

SPEED_SCALE = 20.0

DENSITY_RADIUS = 10.0

EPS = 1e-06

NODE_NAMES = ('speed', 'speed_valid', 'own_longitudinal', 'own_goal_distance', 'opponent_goal_distance', 'own_centroid_dx', 'own_centroid_dy', 'own_centroid_distance', 'nearest_teammate', 'teammate_present', 'nearest_opponent', 'opponent_present', 'teammate_density_r10', 'opponent_density_r10', 'own_longitudinal_rank', 'rank_valid', 'ball_radial_closing', 'ball_closing_valid', 'ball_metric_distance', 'ball_visible')

EDGE_NAMES = ('pair_distance', 'unit_dx', 'unit_dy', 'radial_closing', 'same_team', 'player_ball', 'direction_valid', 'motion_pair_valid')

def instantaneous(x, mask, types):
    """Return node[B,T,N,20], edge[B,T,N,N,8]; no trainable state or history.

    Direction i-j matches original base9diff. Closing is minus distance derivative
    from EXISTING instantaneous velocity; positive means approaching. Coincident
    coordinates have distance0, direction/closing0 with explicit validity0.
    """
    if x.ndim != 4 or x.shape[-1] != 9 or x.shape[1] < 1 or (mask.shape != x.shape[:-1]) or (types.shape != (x.shape[0], x.shape[2])):
        raise ValueError('Instantaneous input requires x[B,T>=1,N,9], matching mask/types')
    if x.dtype != torch.float32 or mask.dtype != torch.bool or types.dtype != torch.long or (x.device != mask.device) or (x.device != types.device):
        raise ValueError('Frozen feature dtypes/device required')
    if not torch.isfinite(x).all() or ((types < 0) | (types > 2)).any():
        raise ValueError('Nonfinite features/invalid type')
    if not torch.all((types == 2).sum(1) == 1):
        raise ValueError('Exactly one ball slot per graph required')
    clean = torch.where(mask[..., None], x, torch.zeros_like(x))
    scale = x.new_tensor([60.0, 40.0])
    p = clean[..., :2] * scale
    motion = mask & (clean[..., 7] > 0.5)
    v = torch.where(motion[..., None], clean[..., 2:4], torch.zeros_like(clean[..., 2:4])) * scale
    player = mask & (types[:, None] < 2)
    ball = mask & (types[:, None] == 2)
    same = (types[:, :, None] == types[:, None, :])[:, None]
    distinct = ~torch.eye(x.shape[2], device=x.device, dtype=torch.bool)
    valid_edges = mask[:, :, :, None] & mask[:, :, None, :] & distinct
    own_all = player[:, :, :, None] & player[:, :, None, :] & same
    own = own_all & distinct
    opponent = player[:, :, :, None] & player[:, :, None, :] & ~same
    d = p[:, :, :, None] - p[:, :, None, :]
    distance = torch.linalg.vector_norm(d, dim=-1)
    direction_valid = valid_edges & (distance > EPS)
    unit = d / distance.clamp_min(EPS)[..., None]
    relative_v = v[:, :, :, None] - v[:, :, None, :]
    motion_pair = direction_valid & motion[:, :, :, None] & motion[:, :, None, :]
    closing = -(relative_v * unit).sum(-1) / SPEED_SCALE
    edge = torch.stack((distance / DISTANCE_SCALE, unit[..., 0], unit[..., 1], torch.where(motion_pair, closing, 0.0), same.expand_as(valid_edges) & player[:, :, :, None] & player[:, :, None, :], ((types[:, :, None] == 2) ^ (types[:, None, :] == 2))[:, None].expand_as(valid_edges), direction_valid, motion_pair), -1).to(x.dtype)
    edge[..., 1:3] = torch.where(direction_valid[..., None], edge[..., 1:3], 0.0)
    edge = torch.where(valid_edges[..., None], edge, 0.0)
    centroid = (own_all[..., None] * p[:, :, None, :]).sum(-2) / own_all.sum(-1).clamp_min(1)[..., None]
    own_delta = p - centroid
    sign = torch.where(types == 0, 1.0, -1.0)[:, None]

    def nearest(which):
        present = which.any(-1)
        value = distance.masked_fill(~which, float('inf')).amin(-1)
        return (torch.where(present, value / DISTANCE_SCALE, 0.0), present)
    nearest_own, own_present = nearest(own)
    nearest_opp, opp_present = nearest(opponent)
    longitude = sign * p[..., 0]
    lesser = longitude[:, :, None, :] < longitude[:, :, :, None]
    tied = longitude[:, :, None, :] == longitude[:, :, :, None]
    rank = ((lesser.to(x.dtype) + 0.5 * tied) * own).sum(-1) / own.sum(-1).clamp_min(1)
    ball_p = (p * ball[..., None]).sum(-2)
    ball_v = (v * ball[..., None]).sum(-2)
    to_ball = p - ball_p[:, :, None]
    ball_distance = torch.linalg.vector_norm(to_ball, dim=-1)
    ball_visible = player & ball.any(-1)[..., None]
    ball_closing_valid = ball_visible & motion & (ball & motion).any(-1)[..., None] & (ball_distance > EPS)
    ball_closing = ((ball_v[:, :, None] - v) * to_ball).sum(-1) / ball_distance.clamp_min(EPS) / SPEED_SCALE
    own_goal = torch.stack((p[..., 0] + 60 * sign, p[..., 1]), -1)
    opp_goal = torch.stack((p[..., 0] - 60 * sign, p[..., 1]), -1)
    node = torch.stack((torch.linalg.vector_norm(v, dim=-1) / SPEED_SCALE, motion, (longitude + 60) / DISTANCE_SCALE, torch.linalg.vector_norm(own_goal, dim=-1) / DISTANCE_SCALE, torch.linalg.vector_norm(opp_goal, dim=-1) / DISTANCE_SCALE, sign * own_delta[..., 0] / DISTANCE_SCALE, sign * own_delta[..., 1] / DISTANCE_SCALE, torch.linalg.vector_norm(own_delta, dim=-1) / DISTANCE_SCALE, nearest_own, own_present, nearest_opp, opp_present, (own & (distance <= DENSITY_RADIUS)).sum(-1) / 10.0, (opponent & (distance <= DENSITY_RADIUS)).sum(-1) / 11.0, rank, own_present, torch.where(ball_closing_valid, ball_closing, 0.0), ball_closing_valid, torch.where(ball_visible, ball_distance / DISTANCE_SCALE, 0.0), ball_visible), -1).to(x.dtype)
    node = torch.where(player[..., None], node, 0.0)
    if not torch.isfinite(node).all() or not torch.isfinite(edge).all():
        raise ValueError('Nonfinite instantaneous features')
    return (node, edge)

CONTEXT_SIZE = 358

HEAD_SEED_OFFSET = 2026110602

def check_inputs(x, mask, types):
    if x.ndim != 4 or x.shape[-1] != 9 or x.shape[1] < 3 or (x.shape[1] % 2 != 1):
        raise ValueError('Expected x[B,odd T>=3,N,9]')
    if mask.shape != x.shape[:-1] or types.shape != (x.shape[0], x.shape[2]):
        raise ValueError('Mask/types shapes differ from observations')
    if x.dtype != torch.float32 or mask.dtype != torch.bool or types.dtype != torch.long:
        raise ValueError('Frozen dtypes: float32 observations, bool mask, int64 types')
    if x.device != mask.device or x.device != types.device:
        raise ValueError('All model inputs must be on the same device')
    if not torch.isfinite(x).all() or ((types < 0) | (types > 2)).any():
        raise ValueError('Nonfinite observations or invalid node type')

def legal_states_from_heads(heads):
    """Joint order: global NONE, team 0 active labels, team 1 active labels."""
    if heads.ndim != 3 or heads.shape[1] != 2 or heads.shape[2] not in (2, 3, 5) or (not torch.isfinite(heads).all()):
        raise ValueError('Expected finite candidate logits[B,2,2|3|5]')
    a, b = (heads[:, 0], heads[:, 1])
    return torch.cat((a[:, :1] + b[:, :1], a[:, 1:] + b[:, :1], a[:, :1] + b[:, 1:]), dim=1)

def expand(linear, additional):
    with torch.random.fork_rng(devices=[]):
        result = nn.Linear(linear.in_features + additional, linear.out_features)
    with torch.no_grad():
        result.weight.zero_()
        result.weight[:, :linear.in_features].copy_(linear.weight)
        result.bias.copy_(linear.bias)
    return result

class SpatialEncoder(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = width
        original = GraphRNN(feature_count=9, width=width, state_count=9)
        self.embed = original.embed
        self.types = original.types
        old = original.forward_graph.message[0]
        with torch.random.fork_rng(devices=[]):
            first = nn.Linear(2 * width + 9, width)
        columns = [*range(width), *range(2 * width, 3 * width), *range(4 * width, 4 * width + 9)]
        with torch.no_grad():
            first.weight.copy_(old.weight[:, columns])
            first.bias.copy_(old.bias)
        self.message = nn.Sequential(first, original.forward_graph.message[1], original.forward_graph.message[2])
        self.aggregate = original.forward_graph.aggregate
        self.project = nn.Sequential(nn.Linear(2 * width, width), nn.GELU())

    def adjacency(self, x, mask, types):
        n = x.shape[2]
        return mask[:, :, :, None] & mask[:, :, None, :] & ~torch.eye(n, dtype=torch.bool, device=x.device)

    def messages(self, embedded, x, types, edges):
        n = x.shape[2]
        receiver = embedded[:, :, :, None].expand(-1, -1, -1, n, -1)
        sender = embedded[:, :, None].expand(-1, -1, n, -1, -1)
        difference = x[:, :, :, None] - x[:, :, None, :]
        return self.message(torch.cat((receiver, sender, difference), -1))

    def forward(self, x, mask, types):
        values = x.masked_fill(~mask[..., None], 0)
        embedded = (self.embed(values) + self.types(types)[:, None]) * mask[..., None]
        edges = self.adjacency(values, mask, types)
        message = self.messages(embedded, values, types, edges)
        context = self.aggregate((message * edges[..., None]).sum(3))
        return self.project(torch.cat((embedded, context), -1)) * mask[..., None]

class TeamDirection(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.gru = nn.GRUCell(3 * width, width)

    def forward(self, vectors, valid, reverse=False):
        b, t, teams, _ = vectors.shape
        h = vectors.new_zeros(b, teams, self.width)
        centre = t // 2
        steps = range(t - 1, centre - 1, -1) if reverse else range(centre + 1)
        for k in steps:
            keep = valid[:, k, :, None]
            h = h * keep
            h = self.gru(vectors[:, k].reshape(b * teams, -1), h.reshape(b * teams, -1)).reshape(b, teams, -1) * keep
        return h

class TeamTemporalC(nn.Module):

    def __init__(self, width=16, *, spatial_type=SpatialEncoder, state_count=9):
        super().__init__()
        self.width = width
        if state_count not in (3, 5, 9):
            raise ValueError('C6 requires legal state count 3, 5 or 9')
        self.spatial = spatial_type(width)
        self.forward_team = TeamDirection(width)
        self.backward_team = TeamDirection(width)
        self.ball_context = nn.Linear(width, 2 * width)
        self.head_seed = (torch.initial_seed() + HEAD_SEED_OFFSET) % 2 ** 63
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.head_seed)
            self.candidate_head = nn.Sequential(nn.Linear(6 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, (state_count + 1) // 2))

    def frame_vectors(self, x, mask, types):
        nodes = self.spatial(x, mask, types)
        pooled = []
        visible = []
        for kind in range(3):
            take = mask & (types[:, None, :] == kind)
            pooled.append((nodes * take[..., None]).sum(2) / take.sum(2, keepdim=True).clamp_min(1))
            visible.append(take.any(2))
        return (torch.stack(pooled, 2), torch.stack(visible, 2))

    def pooled(self, x, mask, types, team_context=None):
        check_inputs(x, mask, types)
        frames, visible = self.frame_vectors(x, mask, types)
        a, b, ball = frames.unbind(2)
        vectors = torch.stack((torch.cat((a, b, ball), -1), torch.cat((b, a, ball), -1)), 2)
        if team_context is not None:
            vectors = torch.cat((vectors, team_context), dim=-1)
        valid = visible[:, :, :2]
        team = torch.cat((self.forward_team(vectors, valid), self.backward_team(vectors, valid, True)), -1)
        centre = x.shape[1] // 2
        ball = self.ball_context(frames[:, centre, 2]) * visible[:, centre, 2, None]
        return torch.cat((team, ball[:, None]), 1)

    def readout(self, pooled):
        if pooled.ndim != 3 or pooled.shape[1:] != (3, 2 * self.width):
            raise ValueError('Expected original pooled team0/team1/ball vectors')
        own, opponent, ball = pooled.unbind(1)
        ordered = torch.stack((torch.cat((own, opponent, ball), -1), torch.cat((opponent, own, ball), -1)), 1)
        return self.candidate_head(ordered)

    def team_logits(self, x, mask, types):
        return self.readout(self.pooled(x, mask, types))

    def forward(self, x, mask, types):
        return legal_states_from_heads(self.team_logits(x, mask, types))

class RichSpatialEncoder(SpatialEncoder):

    def __init__(self, width):
        super().__init__(width)
        self.embed[0] = expand(self.embed[0], len(NODE_NAMES))
        self.message[0] = expand(self.message[0], len(EDGE_NAMES))

    def forward(self, x, mask, types):
        values = x.masked_fill(~mask[..., None], 0)
        node, edge = instantaneous(values, mask, types)
        embedded = (self.embed(torch.cat((values, node), -1)) + self.types(types)[:, None]) * mask[..., None]
        edges = self.adjacency(values, mask, types)
        n = x.shape[2]
        receiver = embedded[:, :, :, None].expand(-1, -1, -1, n, -1)
        sender = embedded[:, :, None].expand(-1, -1, n, -1, -1)
        message = self.message(torch.cat((receiver, sender, values[:, :, :, None] - values[:, :, None, :], edge), -1))
        context = self.aggregate((message * edges[..., None]).sum(3))
        return self.project(torch.cat((embedded, context), -1)) * mask[..., None]

class StaticContextC(TeamTemporalC):
    """C6 with PHASE-29 zero-expanded team-GRU inputs and portable moments."""

    def __init__(self, width, state_count):
        super().__init__(width, spatial_type=RichSpatialEncoder, state_count=state_count)
        for direction in (self.forward_team, self.backward_team):
            old = direction.gru
            with torch.random.fork_rng(devices=[]):
                cell = nn.GRUCell(old.input_size + 2 * CONTEXT_SIZE, old.hidden_size)
            with torch.no_grad():
                cell.weight_ih.zero_()
                cell.weight_ih[:, :old.input_size].copy_(old.weight_ih)
                cell.weight_hh.copy_(old.weight_hh)
                cell.bias_ih.copy_(old.bias_ih)
                cell.bias_hh.copy_(old.bias_hh)
            direction.gru = cell
        self.register_buffer('context_count', torch.zeros(CONTEXT_SIZE, dtype=torch.long))
        self.register_buffer('context_mean', torch.zeros(CONTEXT_SIZE))
        self.register_buffer('context_scale', torch.ones(CONTEXT_SIZE))
        self.register_buffer('context_usable', torch.zeros(CONTEXT_SIZE, dtype=torch.bool))

    def normalized_context(self, raw):
        if not self.context_count.any():
            raise ValueError('Static context needs fitted training moments')
        valid = torch.isfinite(raw) & (self.context_count > 0)
        safe = torch.where(valid, raw, self.context_mean)
        values = ((safe - self.context_mean) / self.context_scale).clamp(-8.0, 8.0)
        return torch.cat((torch.where(valid & self.context_usable, values, 0.0), valid.to(raw.dtype)), dim=-1)

    def team_logits(self, x, mask, types, team_context):
        if team_context.shape != (*x.shape[:2], 2, CONTEXT_SIZE) or team_context.dtype != x.dtype or team_context.device != x.device:
            raise ValueError('Expected raw context[B,T,2,358] with the Graph batch dtype/device')
        return self.readout(self.pooled(x, mask, types, self.normalized_context(team_context)))

    def forward(self, x, mask, types, team_context):
        return legal_states_from_heads(self.team_logits(x, mask, types, team_context))

@dataclass
class Graph:
    x: np.ndarray
    mask: np.ndarray
    types: np.ndarray
    context: np.ndarray


def graph_inputs(half, context):
    context = np.asarray(context)
    if context.shape != (len(half.ball), 2, CONTEXT_SIZE):
        raise ValueError("Expected context[N,2,358]")
    ball = half.ball[:, None]
    positions = np.concatenate((half.players, ball), axis=1).astype(np.float32)
    direction = np.sign(np.nanmean(half.players[:, half.teams == A, 0]) - np.nanmean(half.players[:, half.teams == H, 0])) or 1.0
    positions = (positions / half.pitch * 2 - 1) * direction
    mask = np.isfinite(positions).all(axis=2)
    positions = np.where(mask[:, :, None], positions, 0.0)
    x = np.zeros((*mask.shape, 9), dtype=np.float32)
    x[:, :, :2] = positions
    moving = mask[1:] & mask[:-1]
    x[1:, :, 2:4] = np.where(moving[:, :, None], np.diff(positions, axis=0) * FPS, 0.0)
    x[1:, :, 7] = moving
    relative = mask & mask[:, -1:]
    relative[:, -1] = False
    to_ball = positions - positions[:, -1:]
    x[:, :, 4:6] = np.where(relative[:, :, None], to_ball, 0.0)
    x[:, :, 6] = np.linalg.norm(x[:, :, 4:6], axis=2)
    x[:, :, 8] = relative
    assert np.isfinite(x).all()
    return Graph(x.astype(np.float16), mask, np.r_[half.teams, 2].astype(np.int64), np.asarray(context).astype(np.float16))

def encoded_logits(model, row, points, flip=False):
    """Cache exact spatial outputs, then apply native team-GRU to each context."""
    frames, visible = ([], [])
    x = row.x.astype(np.float32)
    types = row.types.copy()
    context = row.context.astype(np.float32)
    if flip:
        x[:, :, :6] *= -1
        types = np.where(types < 2, 1 - types, types)
        context = context[:, ::-1].copy()
    with torch.no_grad():
        for a in range(0, len(x), 96):
            b = min(len(x), a + 96)
            values, present = model.frame_vectors(torch.from_numpy(x[a:b, None]), torch.from_numpy(row.mask[a:b, None]), torch.from_numpy(np.broadcast_to(types, (b - a, len(types))).copy()))
            frames.append(values[:, 0])
            visible.append(present[:, 0])
        frames, visible = (torch.cat(frames), torch.cat(visible))
        context = torch.from_numpy(context)
        n = len(x)
        scores = []
        for a in range(0, n, 256):
            b = min(n, a + 256)
            ix = np.arange(a, b)[:, None] + np.arange(-points // 2 + 1, points // 2 + 1)
            valid = (ix >= 0) & (ix < n)
            lookup = torch.from_numpy(ix.clip(0, n - 1))
            f = frames[lookup]
            seen = visible[lookup] & torch.from_numpy(valid[..., None])
            f = f * seen[..., None]
            own, other, ball = f.unbind(2)
            vectors = torch.stack((torch.cat((own, other, ball), -1), torch.cat((other, own, ball), -1)), 2)
            if isinstance(model, StaticContextC):
                local = context[lookup].clone()
                local[~torch.from_numpy(valid)] = float('nan')
                vectors = torch.cat((vectors, model.normalized_context(local)), dim=-1)
            team = torch.cat((model.forward_team(vectors, seen[:, :, :2]), model.backward_team(vectors, seen[:, :, :2], True)), dim=-1)
            ball_center = model.ball_context(f[:, points // 2, 2]) * seen[:, points // 2, 2, None]
            heads = model.readout(torch.cat((team, ball_center[:, None]), dim=1))
            left, right = heads.unbind(1)
            native = torch.cat((left[:, :1] + right[:, :1], left[:, 1:] + right[:, :1], left[:, :1] + right[:, 1:]), dim=1)
            joint = native[:, [1, 2, 0]].softmax(dim=1).numpy()
            scores.append(joint[:, [1, 0, 2]] if flip else joint)
    return np.concatenate(scores)

def load_model(path, width):
    """Load a trusted local state dictionary on CPU; no training or refit."""
    model = StaticContextC(width, state_count=3)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def predict(model, half, context, points):
    """Exact spatial caching and two-team flip average for fixed context."""
    if points < 3 or points % 2 != 1:
        raise ValueError("Expected an odd window of at least three points")
    if len(half.ball) == 0:
        return np.empty((0, 3), dtype=np.float32)
    row = graph_inputs(half, context)
    return (encoded_logits(model, row, points) + encoded_logits(model, row, points, True)) * .5

