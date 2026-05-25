#!/usr/bin/env python3
"""
Automatic PCB component instance segmentation and type classification.

What it does:
1) Load merged/full PCB point cloud
2) Detect dominant board plane (background)
3) Extract protruding component points
4) Instance-segment components by DBSCAN
5) Cluster instances into "component types" by geometric descriptors
6) Color same type with same color, while keeping background unpainted

Default input preference:
  - board_full.ply (if exists), otherwise merged.ply

Outputs:
  - components_labeled.ply        (same-type same-color, background unchanged)
  - components_report.json        (instance/type metadata)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def estimate_avg_nn_spacing(
    pcd: o3d.geometry.PointCloud, sample_size: int = 20000
) -> float:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 3:
        return 0.0
    n = pts.shape[0]
    m = min(sample_size, n)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=m, replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)

    dists: List[float] = []
    for idx in sample_idx:
        _, _, sq = tree.search_knn_vector_3d(pcd.points[int(idx)], 2)
        if len(sq) >= 2:
            d = math.sqrt(float(sq[1]))
            if d > 0 and np.isfinite(d):
                dists.append(d)
    if not dists:
        return 0.0
    return float(np.mean(dists))


def default_input_path() -> Path:
    board_full = Path("board_full.ply")
    if board_full.exists():
        return board_full
    return Path("merged.ply")


def make_palette(n: int) -> np.ndarray:
    # Distinct but stable colors.
    base = np.array(
        [
            [0.90, 0.10, 0.10],
            [0.10, 0.70, 0.20],
            [0.10, 0.40, 0.90],
            [0.90, 0.70, 0.10],
            [0.70, 0.20, 0.80],
            [0.10, 0.80, 0.80],
            [0.95, 0.40, 0.15],
            [0.55, 0.55, 0.10],
            [0.35, 0.25, 0.85],
            [0.20, 0.60, 0.60],
        ],
        dtype=np.float64,
    )
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return np.tile(base, (reps, 1))[:n]


@dataclass
class ComponentInstance:
    instance_id: int
    type_id: int
    num_points: int
    centroid: List[float]
    bbox_size: List[float]
    height_mean: float
    height_max: float


def detect_board_plane(
    pcd: o3d.geometry.PointCloud, spacing: float
) -> Tuple[np.ndarray, np.ndarray]:
    dist_th = max(spacing * 1.8, 1e-6)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_th, ransac_n=3, num_iterations=3000
    )
    return np.asarray(plane_model, dtype=np.float64), np.asarray(inliers, dtype=np.int64)


def signed_distance_to_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm <= 0:
        return np.zeros(points.shape[0], dtype=np.float64)
    return (points @ n + d) / n_norm


def plane_basis(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, _ = plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm <= 0:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / n_norm

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u_norm = np.linalg.norm(u)
    if u_norm <= 0:
        u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        u = u / u_norm
    v = np.cross(n, u)
    v = v / max(np.linalg.norm(v), 1e-12)
    return n, u, v


def project_points_to_plane_uv(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    _, u, v = plane_basis(plane)
    x = points @ u
    y = points @ v
    return np.stack([x, y], axis=1)


def choose_component_side(d: np.ndarray, min_h: float) -> int:
    pos = np.sum(d > min_h)
    neg = np.sum(d < -min_h)
    return 1 if pos >= neg else -1


def pca_eigs(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    c = points - np.mean(points, axis=0, keepdims=True)
    cov = np.dot(c.T, c) / max(points.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))[::-1]
    return eigvals


def pca_eigs_2d(points_2d: np.ndarray) -> np.ndarray:
    if points_2d.shape[0] < 3:
        return np.array([0.0, 0.0], dtype=np.float64)
    c = points_2d - np.mean(points_2d, axis=0, keepdims=True)
    cov = np.dot(c.T, c) / max(points_2d.shape[0] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))[::-1]
    return eigvals


def cluster_instances_into_types(features: np.ndarray, threshold: float) -> np.ndarray:
    if features.shape[0] == 1:
        return np.array([0], dtype=np.int64)
    x = StandardScaler().fit_transform(features)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage="ward"
    )
    labels = model.fit_predict(x)
    return labels.astype(np.int64)


def normalize_type_labels(labels: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels)
    remap = {int(v): i for i, v in enumerate(uniq.tolist())}
    out = np.array([remap[int(v)] for v in labels], dtype=np.int64)
    return out


def bbox_gap_distance(
    mn_a: np.ndarray, mx_a: np.ndarray, mn_b: np.ndarray, mx_b: np.ndarray
) -> float:
    # Axis-aligned box gap. 0 means overlap or touching.
    gap = np.maximum(np.maximum(mn_b - mx_a, mn_a - mx_b), 0.0)
    return float(np.linalg.norm(gap))


def enforce_connected_same_type(
    type_labels: np.ndarray,
    bbox_mins: Sequence[np.ndarray],
    bbox_maxs: Sequence[np.ndarray],
    connect_gap_threshold: float,
) -> np.ndarray:
    n = len(type_labels)
    if n <= 1:
        return type_labels

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            gap = bbox_gap_distance(bbox_mins[i], bbox_maxs[i], bbox_mins[j], bbox_maxs[j])
            if gap <= connect_gap_threshold:
                union(i, j)

    # Connected instances must share one type id.
    out = type_labels.copy()
    group_to_type: Dict[int, int] = {}
    for i in range(n):
        root = find(i)
        if root not in group_to_type:
            group_to_type[root] = int(out[i])
        out[i] = group_to_type[root]
    return normalize_type_labels(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify PCB components and color same-type instances."
    )
    parser.add_argument(
        "--input",
        default=str(default_input_path()),
        help="Input merged PCB point cloud (PLY).",
    )
    parser.add_argument(
        "--output", default="components_labeled.ply", help="Labeled colored output PLY."
    )
    parser.add_argument(
        "--components-only-output",
        default="components_only.ply",
        help="Optional output PLY that keeps only component points.",
    )
    parser.add_argument(
        "--report", default="components_report.json", help="Output JSON report."
    )
    parser.add_argument(
        "--type-threshold",
        type=float,
        default=2.0,
        help="Agglomerative distance threshold for grouping component types.",
    )
    parser.add_argument(
        "--min-component-points",
        type=int,
        default=120,
        help="Minimum points per component instance after clustering.",
    )
    parser.add_argument(
        "--paint-background",
        action="store_true",
        help="If set, paint background gray; otherwise keep original background color.",
    )
    parser.add_argument(
        "--connected-gap-scale",
        type=float,
        default=1.5,
        help="Connected-instance gap threshold = spacing * scale; connected instances share same type.",
    )
    parser.add_argument(
        "--components-only",
        action="store_true",
        help="If set, main output keeps only component points (board removed).",
    )
    args = parser.parse_args()

    try:
        inp = Path(args.input)
        ensure_file(inp)
        log(f"Loading point cloud: {inp}")
        pcd = o3d.io.read_point_cloud(str(inp))
        if len(pcd.points) == 0:
            raise RuntimeError("Input point cloud is empty.")

        points = np.asarray(pcd.points)
        has_color = pcd.has_colors() and len(pcd.colors) == len(pcd.points)
        if has_color:
            colors = np.asarray(pcd.colors).copy()
        else:
            colors = np.full((len(points), 3), 0.75, dtype=np.float64)

        spacing = estimate_avg_nn_spacing(pcd)
        if spacing <= 0:
            bbox = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
            spacing = max(float(np.linalg.norm(bbox)) * 1e-4, 1e-6)
        log(f"Estimated spacing: {spacing:.8f}")

        log("Detecting board plane...")
        plane, inliers = detect_board_plane(pcd, spacing)
        if inliers.size < 200:
            raise RuntimeError("Board plane detection failed (too few inliers).")

        d = signed_distance_to_plane(points, plane)
        uv = project_points_to_plane_uv(points, plane)
        min_height = max(spacing * 3.0, 1e-6)
        side = choose_component_side(d, min_height)

        # Background mask: keep near-board points unpainted.
        bg_band = max(spacing * 2.2, 1e-6)
        background_mask = np.abs(d) <= bg_band

        if side > 0:
            comp_mask = d > min_height
        else:
            comp_mask = d < -min_height

        comp_idx = np.where(comp_mask)[0]
        if comp_idx.size < 200:
            raise RuntimeError(
                "Too few component candidates. Try reducing min_height multiplier."
            )
        log(f"Component candidate points: {comp_idx.size:,}")

        # Instance segmentation on protruding points in board-plane 2D.
        # This is robust for 3D components (top/sides) that should stay one instance.
        comp_uv = uv[comp_idx]
        eps = max(spacing * 4.0, 1e-6)
        db = DBSCAN(eps=eps, min_samples=max(20, args.min_component_points // 6))
        labels = db.fit_predict(comp_uv)

        valid_instances: List[np.ndarray] = []
        instance_labels = np.unique(labels[labels >= 0])
        for lb in instance_labels:
            loc = np.where(labels == lb)[0]
            if loc.size >= args.min_component_points:
                valid_instances.append(comp_idx[loc])

        if not valid_instances:
            raise RuntimeError("No valid component instances found.")
        log(f"Detected component instances: {len(valid_instances)}")

        # Build instance descriptors for type clustering.
        feat_rows: List[List[float]] = []
        inst_meta_raw: List[Dict[str, float]] = []
        inst_bbox_mins: List[np.ndarray] = []
        inst_bbox_maxs: List[np.ndarray] = []
        for inst_global_idx in valid_instances:
            pts = points[inst_global_idx]
            pts_uv = uv[inst_global_idx]
            h = np.abs(d[inst_global_idx])
            mn = np.min(pts, axis=0)
            mx = np.max(pts, axis=0)
            inst_bbox_mins.append(mn)
            inst_bbox_maxs.append(mx)
            mn_uv = np.min(pts_uv, axis=0)
            mx_uv = np.max(pts_uv, axis=0)
            size_uv = np.sort(mx_uv - mn_uv)  # rotation-invariant in board plane

            eig2 = pca_eigs_2d(pts_uv)
            eig2_sum = np.sum(eig2) + 1e-12
            eig2_ratio = eig2 / eig2_sum

            h_p50 = float(np.percentile(h, 50))
            h_p90 = float(np.percentile(h, 90))
            h_std = float(np.std(h))
            footprint_area = float(size_uv[0] * size_uv[1])
            footprint_aspect = float(size_uv[1] / max(size_uv[0], 1e-12))

            feat = [
                float(size_uv[0]),
                float(size_uv[1]),
                footprint_area,
                footprint_aspect,
                h_p50,
                h_p90,
                h_std,
                float(np.log(len(inst_global_idx) + 1.0)),
                float(eig2_ratio[0]),
                float(eig2_ratio[1]),
            ]
            feat_rows.append(feat)
            inst_meta_raw.append(
                {
                    "num_points": int(len(inst_global_idx)),
                    "height_mean": float(np.mean(h)),
                    "height_max": float(np.max(h)),
                    "bbox_size": (mx - mn).astype(float).tolist(),
                    "centroid": np.mean(pts, axis=0).astype(float).tolist(),
                }
            )

        features = np.asarray(feat_rows, dtype=np.float64)
        type_labels = cluster_instances_into_types(features, args.type_threshold)
        type_labels = normalize_type_labels(type_labels)
        n_types_before = int(np.max(type_labels)) + 1

        connect_gap = max(spacing * args.connected_gap_scale, 1e-7)
        type_labels = enforce_connected_same_type(
            type_labels, inst_bbox_mins, inst_bbox_maxs, connect_gap_threshold=connect_gap
        )
        n_types = int(np.max(type_labels)) + 1
        log(f"Detected component types (before connectivity merge): {n_types_before}")
        log(f"Detected component types (after connectivity merge) : {n_types}")

        # Paint same type with same color.
        palette = make_palette(n_types)
        if args.paint_background and not has_color:
            colors[:, :] = 0.70
        elif args.paint_background and has_color:
            colors[background_mask] = np.array([0.70, 0.70, 0.70], dtype=np.float64)

        instances: List[ComponentInstance] = []
        for i, inst_global_idx in enumerate(valid_instances):
            t = int(type_labels[i])
            colors[inst_global_idx] = palette[t]
            meta = inst_meta_raw[i]
            instances.append(
                ComponentInstance(
                    instance_id=i,
                    type_id=t,
                    num_points=meta["num_points"],
                    centroid=meta["centroid"],
                    bbox_size=meta["bbox_size"],
                    height_mean=meta["height_mean"],
                    height_max=meta["height_max"],
                )
            )

        component_indices = np.unique(np.concatenate(valid_instances)).astype(np.int64)
        points_components = points[component_indices]
        colors_components = colors[component_indices]

        out = o3d.geometry.PointCloud()
        if args.components_only:
            out.points = o3d.utility.Vector3dVector(points_components)
            out.colors = o3d.utility.Vector3dVector(colors_components)
        else:
            out.points = o3d.utility.Vector3dVector(points)
            out.colors = o3d.utility.Vector3dVector(colors)

        ok = o3d.io.write_point_cloud(args.output, out, write_ascii=False)
        if not ok:
            raise RuntimeError("Failed to write labeled output point cloud.")

        components_only_output = ""
        if args.components_only_output:
            comp_only = o3d.geometry.PointCloud()
            comp_only.points = o3d.utility.Vector3dVector(points_components)
            comp_only.colors = o3d.utility.Vector3dVector(colors_components)
            ok_comp = o3d.io.write_point_cloud(
                args.components_only_output, comp_only, write_ascii=False
            )
            if not ok_comp:
                raise RuntimeError("Failed to write components-only point cloud.")
            components_only_output = args.components_only_output

        report = {
            "input": str(inp),
            "output": args.output,
            "components_only_output": components_only_output,
            "components_only_mode": bool(args.components_only),
            "spacing": float(spacing),
            "plane_model_abcd": plane.astype(float).tolist(),
            "num_points_total": int(len(points)),
            "num_component_candidate_points": int(comp_idx.size),
            "num_component_points_final": int(len(component_indices)),
            "num_instances": int(len(instances)),
            "num_types": int(n_types),
            "num_types_before_connectivity_merge": int(n_types_before),
            "type_threshold": float(args.type_threshold),
            "min_component_points": int(args.min_component_points),
            "connected_gap_scale": float(args.connected_gap_scale),
            "connected_gap_threshold": float(connect_gap),
            "instances": [asdict(x) for x in instances],
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        log(f"Saved labeled point cloud: {args.output}")
        if components_only_output:
            log(f"Saved components-only cloud: {components_only_output}")
        log(f"Saved component report : {args.report}")
        return 0

    except Exception as exc:
        log(f"[ERROR] {exc}")
        log("Detailed traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
