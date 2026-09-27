#!/usr/bin/env python3
"""Minimal OFF mesh loader/sampler without trimesh dependency."""

import numpy as np


def _parse_off(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
        if first == "OFF":
            header = handle.readline().strip()
        elif first.startswith("OFF"):
            header = first[3:].strip() or handle.readline().strip()
        else:
            raise ValueError("not an OFF file: %s" % path)
        parts = header.split()
        if len(parts) < 2:
            raise ValueError("invalid OFF header: %s" % path)
        n_verts = int(parts[0])
        n_faces = int(parts[1]) if len(parts) > 1 else 0
        verts = []
        for _ in range(n_verts):
            line = handle.readline()
            while line.strip() == "":
                line = handle.readline()
            verts.append([float(x) for x in line.split()[:3]])
        faces = []
        for _ in range(n_faces):
            line = handle.readline()
            while line.strip() == "":
                line = handle.readline()
            items = line.split()
            if not items:
                continue
            count = int(items[0])
            faces.append([int(items[i + 1]) for i in range(count)])
    return np.asarray(verts, dtype=np.float32), faces


def normalize_points(points):
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.sqrt((points ** 2).sum(axis=1)).max()
    if scale > 0:
        points = points / scale
    return points


def sample_off(path, num_points, seed):
    verts, faces = _parse_off(path)
    rng = np.random.default_rng(seed)
    if faces:
        # Triangle surface sampling by area-weighted face choice.
        tris = []
        areas = []
        for face in faces:
            if len(face) < 3:
                continue
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            tris.append((v0, v1, v2))
            areas.append(np.linalg.norm(np.cross(v1 - v0, v2 - v0)) * 0.5)
        if not tris:
            idx = rng.choice(verts.shape[0], size=num_points, replace=verts.shape[0] < num_points)
            return normalize_points(verts[idx])
        areas = np.asarray(areas, dtype=np.float64)
        areas = areas / areas.sum()
        picks = rng.choice(len(tris), size=num_points, replace=True, p=areas)
        u = rng.random(num_points)
        v = rng.random(num_points)
        mask = u + v > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        w = 1.0 - u - v
        pts = np.empty((num_points, 3), dtype=np.float32)
        for i, tri_idx in enumerate(picks):
            a, b, c = tris[tri_idx]
            pts[i] = u[i] * a + v[i] * b + w[i] * c
        return normalize_points(pts)
    idx = rng.choice(verts.shape[0], size=num_points, replace=verts.shape[0] < num_points)
    return normalize_points(verts[idx])
