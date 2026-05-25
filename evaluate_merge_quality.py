#!/usr/bin/env python3
"""
Automatic quality evaluation for merged point cloud.

Default inputs:
  - align1.ply (target/reference)
  - align2.ply (source)
  - transform_align2_to_align1.txt (4x4 transform for align2 -> align1)
  - merged.ply (fused result)

It reports:
1) Registration consistency in overlap area (bidirectional NN distance)
2) Overlap ratio (how much of each scan overlaps after alignment)
3) Density uniformity of merged cloud (NN distance variation + voxel occupancy variation)
4) Outlier ratio of merged cloud
5) Overall score (0-100) and quality grade
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def load_transform(path: Path) -> np.ndarray:
    mat = np.loadtxt(str(path))
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Transform matrix must be 4x4, got {mat.shape}")
    return mat


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
    dists = []
    for idx in sample_idx:
        _, _, sq_d = tree.search_knn_vector_3d(pcd.points[int(idx)], 2)
        if len(sq_d) >= 2:
            nn = math.sqrt(float(sq_d[1]))
            if nn > 0 and np.isfinite(nn):
                dists.append(nn)
    if not dists:
        return 0.0
    return float(np.mean(dists))


def robust_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "rmse": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "rmse": float(np.sqrt(np.mean(v**2))),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def evaluate_registration_consistency(
    target: o3d.geometry.PointCloud,
    source_aligned: o3d.geometry.PointCloud,
    spacing: float,
) -> Dict[str, float]:
    # Use a moderate downsampling for stable and fast distance estimation.
    eval_voxel = max(spacing * 1.5, 1e-7)
    target_eval = target.voxel_down_sample(eval_voxel)
    source_eval = source_aligned.voxel_down_sample(eval_voxel)

    if len(target_eval.points) < 100 or len(source_eval.points) < 100:
        raise RuntimeError("Too few points for robust registration evaluation.")

    d_s2t = np.asarray(source_eval.compute_point_cloud_distance(target_eval))
    d_t2s = np.asarray(target_eval.compute_point_cloud_distance(source_eval))

    # Points within overlap threshold are treated as overlap correspondences.
    overlap_threshold = spacing * 3.0
    s_in = d_s2t < overlap_threshold
    t_in = d_t2s < overlap_threshold

    overlap_ratio_source = float(np.mean(s_in)) if d_s2t.size > 0 else 0.0
    overlap_ratio_target = float(np.mean(t_in)) if d_t2s.size > 0 else 0.0
    overlap_ratio_avg = 0.5 * (overlap_ratio_source + overlap_ratio_target)

    overlap_dist = np.concatenate([d_s2t[s_in], d_t2s[t_in]])
    all_dist = np.concatenate([d_s2t, d_t2s])
    stats_overlap = robust_stats(overlap_dist)
    stats_all = robust_stats(all_dist)

    return {
        "eval_voxel": eval_voxel,
        "overlap_threshold": overlap_threshold,
        "overlap_ratio_source": overlap_ratio_source,
        "overlap_ratio_target": overlap_ratio_target,
        "overlap_ratio_avg": overlap_ratio_avg,
        "overlap_rmse": stats_overlap["rmse"],
        "overlap_mean": stats_overlap["mean"],
        "overlap_p95": stats_overlap["p95"],
        "all_rmse": stats_all["rmse"],
    }


def evaluate_density_uniformity(
    merged: o3d.geometry.PointCloud,
    spacing: float,
    sample_size: int = 30000,
) -> Dict[str, float]:
    pts = np.asarray(merged.points)
    if pts.shape[0] < 100:
        raise RuntimeError("Merged cloud has too few points for density evaluation.")

    # 1) NN-distance coefficient of variation (lower is better).
    n = pts.shape[0]
    m = min(sample_size, n)
    rng = np.random.default_rng(123)
    sample_idx = rng.choice(n, size=m, replace=False)
    tree = o3d.geometry.KDTreeFlann(merged)
    nn = np.zeros(m, dtype=np.float64)
    for i, idx in enumerate(sample_idx):
        _, _, sq_d = tree.search_knn_vector_3d(merged.points[int(idx)], 2)
        if len(sq_d) >= 2:
            nn[i] = math.sqrt(float(sq_d[1]))
    nn = nn[nn > 0]
    nn_mean = float(np.mean(nn)) if nn.size > 0 else float("nan")
    nn_std = float(np.std(nn)) if nn.size > 0 else float("nan")
    nn_cv = float(nn_std / max(nn_mean, 1e-12)) if nn.size > 0 else float("nan")

    # 2) Occupied-voxel population variation (lower is better).
    voxel = max(spacing * 2.0, 1e-7)
    mn = np.min(pts, axis=0)
    idx = np.floor((pts - mn) / voxel).astype(np.int64)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    occ_mean = float(np.mean(counts))
    occ_std = float(np.std(counts))
    occ_cv = float(occ_std / max(occ_mean, 1e-12))

    return {
        "density_eval_voxel": voxel,
        "nn_mean": nn_mean,
        "nn_std": nn_std,
        "nn_cv": nn_cv,
        "voxel_occ_mean": occ_mean,
        "voxel_occ_std": occ_std,
        "voxel_occ_cv": occ_cv,
    }


def evaluate_outlier_ratio(
    merged: o3d.geometry.PointCloud, nb_neighbors: int = 24, std_ratio: float = 2.0
) -> Dict[str, float]:
    if len(merged.points) < 200:
        return {"outlier_ratio": float("nan")}
    _, inlier_idx = merged.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    inlier_cnt = len(inlier_idx)
    total = len(merged.points)
    outlier_ratio = 1.0 - (inlier_cnt / max(total, 1))
    return {"outlier_ratio": float(outlier_ratio)}


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def score_quality(
    reg: Dict[str, float], den: Dict[str, float], outlier: Dict[str, float], spacing: float
) -> Tuple[float, str, Dict[str, float]]:
    # Registration error score (sub-mm goal if unit is meter).
    rmse = reg.get("overlap_rmse", float("nan"))
    if np.isfinite(rmse) and spacing > 0:
        # Around 0.5*spacing is good, >=2*spacing is poor.
        s_reg_err = clamp01((2.0 * spacing - rmse) / (1.5 * spacing))
    else:
        s_reg_err = 0.0

    # Overlap score.
    overlap = reg.get("overlap_ratio_avg", 0.0)
    s_overlap = clamp01(overlap / 0.5)  # 50%+ average overlap is full score.

    # Density uniformity score.
    nn_cv = den.get("nn_cv", float("nan"))
    occ_cv = den.get("voxel_occ_cv", float("nan"))
    s_nn = clamp01((0.35 - nn_cv) / 0.30) if np.isfinite(nn_cv) else 0.0
    s_occ = clamp01((0.80 - occ_cv) / 0.60) if np.isfinite(occ_cv) else 0.0
    s_density = 0.6 * s_nn + 0.4 * s_occ

    # Outlier score.
    out_ratio = outlier.get("outlier_ratio", float("nan"))
    s_out = clamp01((0.08 - out_ratio) / 0.08) if np.isfinite(out_ratio) else 0.0

    total = 100.0 * (0.50 * s_reg_err + 0.20 * s_overlap + 0.20 * s_density + 0.10 * s_out)

    if total >= 85:
        grade = "A (excellent)"
    elif total >= 70:
        grade = "B (good)"
    elif total >= 55:
        grade = "C (acceptable)"
    else:
        grade = "D (needs improvement)"

    parts = {
        "registration_score_0_100": 100.0 * s_reg_err,
        "overlap_score_0_100": 100.0 * s_overlap,
        "density_score_0_100": 100.0 * s_density,
        "outlier_score_0_100": 100.0 * s_out,
    }
    return float(total), grade, parts


@dataclass
class EvalReport:
    spacing_estimate: float
    registration: Dict[str, float]
    density: Dict[str, float]
    outlier: Dict[str, float]
    total_score: float
    grade: str
    score_breakdown: Dict[str, float]
    unit_hint: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate merged point cloud quality.")
    parser.add_argument("--target", default="align1.ply", help="Reference cloud")
    parser.add_argument("--source", default="align2.ply", help="Source cloud")
    parser.add_argument(
        "--transform",
        default="transform_align2_to_align1.txt",
        help="4x4 transform for source->target",
    )
    parser.add_argument("--merged", default="board_full.ply", help="Merged cloud to evaluate")
    parser.add_argument(
        "--json",
        default="evaluation_report_board_full.json",
        help="Output JSON report path (empty string to disable)",
    )
    args = parser.parse_args()

    try:
        target_path = Path(args.target)
        source_path = Path(args.source)
        transform_path = Path(args.transform)
        merged_path = Path(args.merged)

        ensure_file(target_path)
        ensure_file(source_path)
        ensure_file(transform_path)
        ensure_file(merged_path)

        log("Loading point clouds and transform...")
        target = o3d.io.read_point_cloud(str(target_path))
        source = o3d.io.read_point_cloud(str(source_path))
        merged = o3d.io.read_point_cloud(str(merged_path))
        tf = load_transform(transform_path)

        if len(target.points) == 0 or len(source.points) == 0 or len(merged.points) == 0:
            raise RuntimeError("At least one input point cloud is empty.")

        source_aligned = o3d.geometry.PointCloud(source)
        source_aligned.transform(tf)

        spacing_t = estimate_avg_nn_spacing(target)
        spacing_s = estimate_avg_nn_spacing(source_aligned)
        spacing_m = estimate_avg_nn_spacing(merged)
        valid = [x for x in [spacing_t, spacing_s, spacing_m] if x > 0 and np.isfinite(x)]
        spacing = float(np.mean(valid)) if valid else 1e-4

        log(f"Estimated spacing: {spacing:.8f}")

        reg = evaluate_registration_consistency(target, source_aligned, spacing)
        den = evaluate_density_uniformity(merged, spacing)
        out = evaluate_outlier_ratio(merged)
        total_score, grade, breakdown = score_quality(reg, den, out, spacing)

        unit_hint = (
            "If your unit is meter, then 0.0005 means 0.5 mm."
            " Use overlap_rmse to verify sub-millimeter precision."
        )

        report = EvalReport(
            spacing_estimate=spacing,
            registration=reg,
            density=den,
            outlier=out,
            total_score=total_score,
            grade=grade,
            score_breakdown=breakdown,
            unit_hint=unit_hint,
        )

        log("\n===== Evaluation Result =====")
        log(f"Total score: {report.total_score:.2f} / 100")
        log(f"Grade: {report.grade}")
        log(f"Overlap RMSE: {report.registration['overlap_rmse']:.8f}")
        log(f"Overlap P95 : {report.registration['overlap_p95']:.8f}")
        log(f"Overlap ratio(avg): {report.registration['overlap_ratio_avg']:.4f}")
        log(f"Density NN-CV: {report.density['nn_cv']:.4f} (lower is better)")
        log(f"Voxel Occ-CV: {report.density['voxel_occ_cv']:.4f} (lower is better)")
        log(f"Outlier ratio: {report.outlier['outlier_ratio']:.4f} (lower is better)")
        log("\nSub-score breakdown:")
        for k, v in report.score_breakdown.items():
            log(f"  - {k}: {v:.2f}")
        log(f"\nHint: {report.unit_hint}")

        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            log(f"\nSaved JSON report: {args.json}")

        return 0

    except Exception as exc:
        log(f"[ERROR] {exc}")
        log("Detailed traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
