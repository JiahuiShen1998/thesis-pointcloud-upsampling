import numpy as np


def _pairwise_distances(points):
    """Return the full pairwise Euclidean distance matrix for a small point set."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))


def _normalize_probabilities(scores):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(scores.shape[0], 1.0 / max(scores.shape[0], 1), dtype=np.float64)
    return scores / total


def _estimate_edge_weights(points, k_neighbors=8, edge_power=1.5):
    """
    Estimate a simple edge-aware sampling weight for each point.

    Points in locally sparse or anisotropic neighborhoods receive higher weights,
    which makes them more likely to act as anchors for synthetic samples.
    """
    num_points = points.shape[0]
    if num_points <= 1:
        return np.ones((num_points,), dtype=np.float64), np.zeros((num_points, 0), dtype=np.int64), np.zeros(
            (num_points, 0), dtype=np.float64
        )

    dist = _pairwise_distances(points)
    np.fill_diagonal(dist, np.inf)

    k_eff = min(max(int(k_neighbors), 1), num_points - 1)
    neighbor_order = np.argsort(dist, axis=1)[:, :k_eff]
    neighbor_dist = np.take_along_axis(dist, neighbor_order, axis=1)

    # Sparse neighborhoods are used as a proxy for edge regions.
    edge_score = neighbor_dist.mean(axis=1)
    edge_score = np.power(edge_score + 1e-6, edge_power)
    return _normalize_probabilities(edge_score), neighbor_order, neighbor_dist


def ear_upsample_points(points_xyz, points_feat=None, target_num=512, k_neighbors=8,
                        edge_power=1.5, interp_min=0.35, interp_max=0.65, jitter_std=0.01):
    """
    Edge-aware resampling for small point sets.

    The routine keeps the observed points, then synthesizes additional samples by
    interpolating between anchor points and their local neighbors. Anchors are
    drawn with higher probability in sparse neighborhoods so that sharp, thin,
    or under-sampled structures are represented more often.
    """
    if points_feat is None:
        points_feat = np.zeros((points_xyz.shape[0], 0), dtype=np.float32)

    if points_xyz.shape[0] != points_feat.shape[0]:
        raise ValueError('points_xyz and points_feat must have the same length')

    if points_xyz.shape[0] == 0:
        out_xyz = np.zeros((target_num, 3), dtype=np.float32)
        out_feat = np.zeros((target_num, points_feat.shape[1]), dtype=np.float32)
        return out_xyz, out_feat

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    points_feat = np.asarray(points_feat, dtype=np.float32)
    num_points = points_xyz.shape[0]

    if num_points >= target_num:
        choice = np.random.choice(num_points, target_num, replace=False)
        return points_xyz[choice], points_feat[choice]

    anchor_prob, neighbors, neighbor_dist = _estimate_edge_weights(
        points_xyz, k_neighbors=k_neighbors, edge_power=edge_power
    )

    extra_num = target_num - num_points
    extra_xyz = np.empty((extra_num, 3), dtype=np.float32)
    extra_feat = np.empty((extra_num, points_feat.shape[1]), dtype=np.float32)

    for i in range(extra_num):
        anchor_idx = np.random.choice(num_points, p=anchor_prob)
        candidate_neighbors = neighbors[anchor_idx]
        candidate_dist = neighbor_dist[anchor_idx]

        if candidate_neighbors.size == 0:
            neighbor_idx = anchor_idx
        else:
            neighbor_prob = _normalize_probabilities(candidate_dist)
            neighbor_idx = np.random.choice(candidate_neighbors, p=neighbor_prob)

        alpha = np.random.uniform(interp_min, interp_max)
        base_xyz = points_xyz[anchor_idx]
        neigh_xyz = points_xyz[neighbor_idx]
        new_xyz = (1.0 - alpha) * base_xyz + alpha * neigh_xyz

        if jitter_std > 0.0:
            direction = neigh_xyz - base_xyz
            noise_scale = np.linalg.norm(direction) * jitter_std
            if noise_scale > 0.0:
                new_xyz = new_xyz + np.random.normal(scale=noise_scale, size=3).astype(np.float32)

        extra_xyz[i] = new_xyz.astype(np.float32)

        if points_feat.shape[1] > 0:
            base_feat = points_feat[anchor_idx]
            neigh_feat = points_feat[neighbor_idx]
            extra_feat[i] = (1.0 - alpha) * base_feat + alpha * neigh_feat

    out_xyz = np.concatenate([points_xyz, extra_xyz], axis=0)
    out_feat = np.concatenate([points_feat, extra_feat], axis=0)
    return out_xyz, out_feat
