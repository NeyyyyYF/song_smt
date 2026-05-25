#!/usr/bin/env python3
"""
Find corresponding 3D component in components-only point cloud by 2D mask.

Default inputs:
  - Point cloud: components_only.ply
  - Query mask : 1_mask.png

Outputs:
  - matched_component.ply          (best matched component point cloud)
  - match_report.json              (top-k matching results)
  - match_debug_top1.png           (query vs best silhouette debug image)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    from plyfile import PlyData, PlyElement
except Exception:
    PlyData = None
    PlyElement = None


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def default_full_cloud_path() -> str:
    p = Path("board_full.ply")
    return str(p) if p.exists() else ""


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def load_binary_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"Failed to read mask image: {path}")
    if m.ndim == 3:
        # If RGBA exists, prioritize alpha; else grayscale.
        if m.shape[2] == 4:
            gray = m[:, :, 3]
        else:
            gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    else:
        gray = m
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = largest_component(bw)
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    return bw


def contour_from_mask(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise RuntimeError("No contour found in mask.")
    return max(cnts, key=cv2.contourArea)


def contour_descriptors(cnt: np.ndarray, img_shape: Tuple[int, int]) -> Dict[str, float]:
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = max(float(cv2.contourArea(hull)), 1e-12)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = float(max(w, h) / max(min(w, h), 1))
    extent = float(area / max(w * h, 1))
    solidity = float(area / hull_area)
    img_area = float(img_shape[0] * img_shape[1])
    fill_ratio = float(area / max(img_area, 1.0))
    circ = float((4.0 * math.pi * area) / max(peri * peri, 1e-12))
    return {
        "area": area,
        "perimeter": peri,
        "aspect": aspect,
        "extent": extent,
        "solidity": solidity,
        "fill_ratio": fill_ratio,
        "circularity": circ,
    }


def pca_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return board-like 2D basis from global PCA.
    """
    c = points - np.mean(points, axis=0, keepdims=True)
    cov = np.dot(c.T, c) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u = eigvecs[:, 0]
    v = eigvecs[:, 1]
    return u, v


def local_axes_from_component(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build local orthonormal frame (u, v, n) from component points.
    n is component thickness direction (smallest PCA axis).
    """
    c = np.mean(points, axis=0)
    q = points - c[None, :]
    cov = np.dot(q.T, q) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u = eigvecs[:, 0]
    v = eigvecs[:, 1]
    n = np.cross(u, v)
    n = n / max(np.linalg.norm(n), 1e-12)
    return c, u, v, n


def project_local(points: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    q = points - c[None, :]
    return np.stack([q @ u, q @ v, q @ n], axis=1)


def extract_component_with_board_context(
    full_points: np.ndarray,
    full_colors: Optional[np.ndarray],
    matched_points: np.ndarray,
    xy_margin_scale: float,
    z_up_margin_scale: float,
    z_down_margin_scale: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, float]]:
    """
    Extract region from full board cloud around matched component.
    Keeps board points around the component (no board-removal).
    """
    c, u, v, n = local_axes_from_component(matched_points)
    comp_local = project_local(matched_points, c, u, v, n)
    full_local = project_local(full_points, c, u, v, n)

    comp_min = np.min(comp_local, axis=0)
    comp_max = np.max(comp_local, axis=0)
    size = comp_max - comp_min

    xy_margin = max(float(max(size[0], size[1]) * xy_margin_scale), 1e-6)
    z_up = max(float(size[2] * z_up_margin_scale), 1e-6)
    z_down = max(float(size[2] * z_down_margin_scale), 1e-6)

    lo = np.array([comp_min[0] - xy_margin, comp_min[1] - xy_margin, comp_min[2] - z_down])
    hi = np.array([comp_max[0] + xy_margin, comp_max[1] + xy_margin, comp_max[2] + z_up])

    keep = (
        (full_local[:, 0] >= lo[0])
        & (full_local[:, 0] <= hi[0])
        & (full_local[:, 1] >= lo[1])
        & (full_local[:, 1] <= hi[1])
        & (full_local[:, 2] >= lo[2])
        & (full_local[:, 2] <= hi[2])
    )
    idx = np.where(keep)[0]
    pts = full_points[idx]
    col = full_colors[idx] if (full_colors is not None and len(full_colors) == len(full_points)) else None
    meta = {
        "xy_margin": float(xy_margin),
        "z_up_margin": float(z_up),
        "z_down_margin": float(z_down),
        "num_points_extracted": int(len(idx)),
    }
    return pts, col, idx, meta


def project_to_uv(points: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack([points @ u, points @ v], axis=1)


def estimate_spacing_2d(uv: np.ndarray, sample_size: int = 30000) -> float:
    n = uv.shape[0]
    if n < 3:
        return 0.0
    m = min(sample_size, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=m, replace=False)
    s = uv[idx]
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(s)
    dist, _ = nn.kneighbors(s)
    d = dist[:, 1]
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def component_silhouette(
    uv_pts: np.ndarray, canvas: int = 256, pad: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    mn = np.min(uv_pts, axis=0)
    mx = np.max(uv_pts, axis=0)
    span = np.maximum(mx - mn, 1e-12)
    scale = (canvas - 2 * pad) / float(max(span[0], span[1]))

    pix = (uv_pts - mn) * scale + pad
    pix = np.round(pix).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, canvas - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, canvas - 1)

    img = np.zeros((canvas, canvas), dtype=np.uint8)
    img[pix[:, 1], pix[:, 0]] = 255
    img = cv2.dilate(
        img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    img = cv2.morphologyEx(
        img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    img = largest_component(img)
    cnt = contour_from_mask(img)
    return img, cnt


def match_score(
    query_cnt: np.ndarray,
    query_desc: Dict[str, float],
    cand_cnt: np.ndarray,
    cand_desc: Dict[str, float],
) -> float:
    s_shape = float(
        cv2.matchShapes(query_cnt, cand_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    )
    s_aspect = abs(math.log(max(cand_desc["aspect"], 1e-9) / max(query_desc["aspect"], 1e-9)))
    s_extent = abs(cand_desc["extent"] - query_desc["extent"])
    s_solidity = abs(cand_desc["solidity"] - query_desc["solidity"])
    s_fill = abs(math.log(max(cand_desc["fill_ratio"], 1e-9) / max(query_desc["fill_ratio"], 1e-9)))
    s_circ = abs(cand_desc["circularity"] - query_desc["circularity"])

    # Weighted sum: prioritize contour shape, then compactness/geometric consistency.
    score = (
        0.65 * s_shape
        + 0.12 * s_aspect
        + 0.08 * s_extent
        + 0.06 * s_solidity
        + 0.06 * s_fill
        + 0.03 * s_circ
    )
    return float(score)


def save_component_ply(
    points: np.ndarray, colors: Optional[np.ndarray], idx: np.ndarray, out_path: Path
) -> None:
    sel_pts = points[idx]
    sel_col = colors[idx] if (colors is not None and len(colors) == len(points)) else None
    write_point_cloud_with_fallback(out_path, sel_pts, sel_col)


def read_point_cloud_with_fallback(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise RuntimeError("Empty components point cloud.")
        pts = np.asarray(pcd.points)
        col = np.asarray(pcd.colors).copy() if pcd.has_colors() else None
        return pts, col

    if PlyData is None:
        raise RuntimeError(
            "Neither open3d nor plyfile is available. Install one of them to read PLY."
        )
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    pts = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    col = None
    names = v.data.dtype.names or []
    if {"red", "green", "blue"}.issubset(set(names)):
        col = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
    return pts, col


def write_point_cloud_with_fallback(
    path: Path, points: np.ndarray, colors: Optional[np.ndarray]
) -> None:
    if o3d is not None:
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) == len(points):
            p.colors = o3d.utility.Vector3dVector(colors)
        ok = o3d.io.write_point_cloud(str(path), p, write_ascii=False)
        if not ok:
            raise RuntimeError(f"Failed to save point cloud: {path}")
        return

    if PlyElement is None:
        raise RuntimeError(
            "Neither open3d nor plyfile is available. Install one of them to write PLY."
        )

    if colors is not None and len(colors) == len(points):
        c = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
        arr = np.empty(
            len(points),
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        arr["x"] = points[:, 0].astype(np.float32)
        arr["y"] = points[:, 1].astype(np.float32)
        arr["z"] = points[:, 2].astype(np.float32)
        arr["red"] = c[:, 0]
        arr["green"] = c[:, 1]
        arr["blue"] = c[:, 2]
    else:
        arr = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        arr["x"] = points[:, 0].astype(np.float32)
        arr["y"] = points[:, 1].astype(np.float32)
        arr["z"] = points[:, 2].astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def save_debug_image(query_mask: np.ndarray, best_mask: np.ndarray, out_path: Path) -> None:
    q = cv2.cvtColor(query_mask, cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
    q = cv2.resize(q, (256, 256), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((300, 540, 3), dtype=np.uint8)
    canvas[30:286, 10:266] = q
    canvas[30:286, 274:530] = b
    cv2.putText(canvas, "Query mask", (74, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(canvas, "Best matched silhouette", (296, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.imwrite(str(out_path), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find matching component in components_only.ply using mask shape."
    )
    parser.add_argument("--components", default="components_only.ply", help="Components-only point cloud.")
    parser.add_argument("--mask", default="1_mask.png", help="Query binary mask image.")
    parser.add_argument("--output", default="matched_component.ply", help="Output best-matched component PLY.")
    parser.add_argument(
        "--full-cloud",
        default=default_full_cloud_path(),
        help="Optional full cloud (e.g. board_full.ply) for context extraction.",
    )
    parser.add_argument(
        "--full-output",
        default="matched_component_with_board.ply",
        help="Output PLY path for matched component + board context crop.",
    )
    parser.add_argument("--report", default="match_report.json", help="Output JSON report path.")
    parser.add_argument("--debug-image", default="match_debug_top1.png", help="Debug image path.")
    parser.add_argument("--topk", type=int, default=5, help="Top-K candidates saved to report.")
    parser.add_argument("--eps-scale", type=float, default=3.0, help="DBSCAN eps = spacing2d * scale.")
    parser.add_argument("--min-points", type=int, default=100, help="Minimum points per component cluster.")
    parser.add_argument(
        "--full-xy-margin-scale",
        type=float,
        default=0.1,
        help="XY margin around matched component in full-cloud extraction.",
    )
    parser.add_argument(
        "--full-z-up-margin-scale",
        type=float,
        default=0.35,
        help="Positive local-Z margin scale in full-cloud extraction.",
    )
    parser.add_argument(
        "--full-z-down-margin-scale",
        type=float,
        default=2.0,
        help="Negative local-Z margin scale in full-cloud extraction (captures board under component).",
    )
    args = parser.parse_args()

    try:
        comp_path = Path(args.components)
        mask_path = Path(args.mask)
        ensure_file(comp_path)
        ensure_file(mask_path)

        log("Loading query mask...")
        qmask = load_binary_mask(mask_path)
        qcnt = contour_from_mask(qmask)
        qdesc = contour_descriptors(qcnt, qmask.shape[:2])

        log("Loading point cloud...")
        points, colors = read_point_cloud_with_fallback(comp_path)

        u, v = pca_basis(points)
        uv = project_to_uv(points, u, v)
        spacing2d = estimate_spacing_2d(uv)
        if spacing2d <= 0:
            span = np.max(uv, axis=0) - np.min(uv, axis=0)
            spacing2d = max(float(np.linalg.norm(span)) * 1e-4, 1e-8)
        eps = max(spacing2d * args.eps_scale, 1e-8)
        min_samples = max(12, args.min_points // 8)
        log(f"2D spacing={spacing2d:.8f}, DBSCAN eps={eps:.8f}, min_samples={min_samples}")

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(uv)
        uniq = [int(x) for x in np.unique(labels) if x >= 0]
        if not uniq:
            raise RuntimeError("No component clusters found. Try larger --eps-scale.")

        candidates: List[Dict[str, object]] = []
        sil_cache: Dict[int, np.ndarray] = {}
        for lb in uniq:
            idx = np.where(labels == lb)[0]
            if len(idx) < args.min_points:
                continue
            sil, cnt = component_silhouette(uv[idx], canvas=256, pad=12)
            cdesc = contour_descriptors(cnt, sil.shape[:2])
            score = match_score(qcnt, qdesc, cnt, cdesc)
            centroid = np.mean(points[idx], axis=0)
            bbox_min = np.min(points[idx], axis=0)
            bbox_max = np.max(points[idx], axis=0)
            candidates.append(
                {
                    "label": lb,
                    "num_points": int(len(idx)),
                    "score": float(score),
                    "centroid": centroid.astype(float).tolist(),
                    "bbox_size": (bbox_max - bbox_min).astype(float).tolist(),
                    "indices": idx,
                    "desc": cdesc,
                }
            )
            sil_cache[lb] = sil

        if not candidates:
            raise RuntimeError("No valid clusters after min-points filtering.")

        candidates.sort(key=lambda x: float(x["score"]))
        best = candidates[0]
        best_idx = np.asarray(best["indices"], dtype=np.int64)

        save_component_ply(points, colors, best_idx, Path(args.output))
        save_debug_image(qmask, sil_cache[int(best["label"])], Path(args.debug_image))

        full_extract_meta: Optional[Dict[str, float]] = None
        full_extract_path = ""
        if args.full_cloud:
            full_path = Path(args.full_cloud)
            ensure_file(full_path)
            log(f"Loading full cloud for context extraction: {full_path}")
            full_points, full_colors = read_point_cloud_with_fallback(full_path)
            matched_points = points[best_idx]
            cropped_pts, cropped_col, _, full_extract_meta = extract_component_with_board_context(
                full_points=full_points,
                full_colors=full_colors,
                matched_points=matched_points,
                xy_margin_scale=max(args.full_xy_margin_scale, 0.0),
                z_up_margin_scale=max(args.full_z_up_margin_scale, 0.0),
                z_down_margin_scale=max(args.full_z_down_margin_scale, 0.0),
            )
            if len(cropped_pts) == 0:
                raise RuntimeError("Full-cloud extraction returned empty result.")
            write_point_cloud_with_fallback(Path(args.full_output), cropped_pts, cropped_col)
            full_extract_path = args.full_output
            log(f"Matched component with board context saved: {args.full_output}")

        topk = candidates[: max(1, args.topk)]
        report_topk = []
        for rank, item in enumerate(topk, start=1):
            report_topk.append(
                {
                    "rank": rank,
                    "cluster_label": int(item["label"]),
                    "score": float(item["score"]),
                    "num_points": int(item["num_points"]),
                    "centroid": item["centroid"],
                    "bbox_size": item["bbox_size"],
                    "descriptor": item["desc"],
                }
            )

        report = {
            "components_path": str(comp_path),
            "mask_path": str(mask_path),
            "output_path": args.output,
            "debug_image": args.debug_image,
            "spacing2d": float(spacing2d),
            "dbscan_eps": float(eps),
            "dbscan_min_samples": int(min_samples),
            "num_clusters_considered": int(len(candidates)),
            "best_match": report_topk[0],
            "topk": report_topk,
            "full_cloud_input": args.full_cloud,
            "full_cloud_output": full_extract_path,
            "full_cloud_extraction": full_extract_meta,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        log(f"Best matched component saved: {args.output}")
        log(f"Debug comparison image saved: {args.debug_image}")
        log(f"Match report saved: {args.report}")
        log(f"Best score: {report_topk[0]['score']:.6f}")
        return 0

    except Exception as exc:
        log(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
