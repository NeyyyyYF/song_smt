#!/usr/bin/env python3
"""
Extract PCB component mask using Segment Anything (SAM).

Usage idea:
1) Install segment_anything and torch.
2) Provide SAM checkpoint path.
3) Script auto-generates candidate masks and picks the best component mask.

Default input/output:
  - input : 1.jpg
  - mask  : 1_mask_sam.png
  - cutout: 1_cutout_sam.png
  - debug : 1_sam_debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def circular_hue_distance(h: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(h.astype(np.float32) - float(center))
    return np.minimum(d, 180.0 - d)


def estimate_board_green_hue(hsv: np.ndarray) -> Tuple[float, float]:
    h, w = hsv.shape[:2]
    by = max(h // 10, 6)
    bx = max(w // 10, 6)
    border = np.zeros((h, w), dtype=np.uint8)
    border[:by, :] = 1
    border[-by:, :] = 1
    border[:, :bx] = 1
    border[:, -bx:] = 1

    px = hsv[border > 0]
    hh = px[:, 0].astype(np.float32)
    ss = px[:, 1].astype(np.float32)
    vv = px[:, 2].astype(np.float32)
    valid = (ss > np.percentile(ss, 30)) & (vv > np.percentile(vv, 20))
    if np.any(valid):
        hh = hh[valid]
        ss = ss[valid]

    hist = np.zeros(180, dtype=np.float64)
    for hval, sw in zip(hh.astype(np.int32), ss + 1.0):
        hist[int(hval) % 180] += float(sw)
    center = float(np.argmax(hist))
    tol = float(np.clip(np.percentile(circular_hue_distance(hh, center), 90) + 4.0, 8.0, 30.0))
    return center, tol


def mask_descriptors(mask_u8: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {
            "area": 0.0,
            "extent": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "border_touch_ratio": 1.0,
            "center_overlap": 0.0,
        }
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = max(float(cv2.contourArea(hull)), 1e-12)
    x, y, w, h = cv2.boundingRect(cnt)
    extent = float(area / max(w * h, 1))
    solidity = float(area / hull_area)
    circularity = float((4.0 * np.pi * area) / max(peri * peri, 1e-12))

    ih, iw = image_shape
    edge_band = max(min(ih, iw) // 40, 3)
    border = np.zeros_like(mask_u8)
    border[:edge_band, :] = 255
    border[-edge_band:, :] = 255
    border[:, :edge_band] = 255
    border[:, -edge_band:] = 255
    border_touch_ratio = float(np.mean((mask_u8 > 0) & (border > 0)))

    cy0, cy1 = int(ih * 0.3), int(ih * 0.7)
    cx0, cx1 = int(iw * 0.3), int(iw * 0.7)
    center_win = np.zeros_like(mask_u8)
    center_win[cy0:cy1, cx0:cx1] = 255
    center_overlap = float(np.mean((mask_u8 > 0) & (center_win > 0)))

    return {
        "area": area,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "border_touch_ratio": border_touch_ratio,
        "center_overlap": center_overlap,
    }


def recover_fine_details(img_bgr: np.ndarray, seed_mask: np.ndarray, expand_px: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 35, 120)
    dil = cv2.dilate(
        seed_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=max(expand_px // 2, 1),
    )
    add = np.where((edges > 0) & (dil > 0), 255, 0).astype(np.uint8)
    out = np.where((seed_mask > 0) | (add > 0), 255, 0).astype(np.uint8)
    out = cv2.morphologyEx(
        out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    return out


def keep_components_near_seed(mask: np.ndarray, seed_mask: np.ndarray, max_dist_px: int, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    inv_seed = np.where(seed_mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv_seed, cv2.DIST_L2, 3)
    out = np.zeros_like(mask)
    for comp_id in range(1, num):
        comp = labels == comp_id
        if not np.any(comp):
            continue
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        dmin = float(np.min(dist[comp]))
        if dmin <= max_dist_px:
            out[comp] = 255
    return out


def recover_pins_and_pads(
    img_bgr: np.ndarray,
    core_mask: np.ndarray,
    green_hue: float,
    green_tol: float,
    search_px: int,
) -> np.ndarray:
    """
    Recover thin pins/pads around core mask using edge + color priors.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    hue_dist = circular_hue_distance(h, green_hue)

    # Non-green and metallic priors (solder/pin often low-S high-V).
    non_green = hue_dist > (green_tol * 0.80)
    metallic = (s < 70.0) & (v > 95.0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 30, 110) > 0

    dil = cv2.dilate(
        core_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=max(search_px // 2, 1),
    )
    ring = (dil > 0) & (core_mask == 0)

    cand = ring & edges & (non_green | metallic)
    cand_u8 = np.where(cand, 255, 0).astype(np.uint8)
    cand_u8 = cv2.dilate(
        cand_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    merged = np.where((core_mask > 0) | (cand_u8 > 0), 255, 0).astype(np.uint8)
    merged = cv2.morphologyEx(
        merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    merged = keep_components_near_seed(
        merged,
        seed_mask=core_mask,
        max_dist_px=max(search_px * 2, 8),
        min_area=6,
    )
    return merged


def select_best_sam_mask(
    masks: List[Dict],
    img_bgr: np.ndarray,
    green_hue: float,
    green_tol: float,
    min_component_area_ratio: float,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)

    best_score = -1e18
    best_mask = None

    edge = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0), 35, 120)

    for item in masks:
        seg = item["segmentation"]
        if seg.dtype != np.bool_:
            seg = seg.astype(bool)
        area = float(np.count_nonzero(seg))
        if area < img_area * min_component_area_ratio or area > img_area * 0.85:
            continue

        mask_u8 = np.where(seg, 255, 0).astype(np.uint8)
        desc = mask_descriptors(mask_u8, (h, w))
        if desc["area"] <= 0:
            continue

        h_in = hsv[:, :, 0][seg]
        s_in = hsv[:, :, 1][seg].astype(np.float32)
        v_in = hsv[:, :, 2][seg].astype(np.float32)
        green_like = (circular_hue_distance(h_in, green_hue) <= green_tol) & (s_in > 35) & (v_in > 25)
        green_ratio = float(np.mean(green_like)) if h_in.size > 0 else 1.0

        mean_sat = float(np.mean(s_in)) if s_in.size > 0 else 0.0
        edge_density = float(np.mean(edge[seg] > 0)) if h_in.size > 0 else 0.0

        # How much green appears around mask boundary: whole component often sits on green board,
        # while printed letters are inside package and have low green ring ratio.
        dil = cv2.dilate(
            mask_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        ring = (dil > 0) & (mask_u8 == 0)
        if np.any(ring):
            h_ring = hsv[:, :, 0][ring]
            s_ring = hsv[:, :, 1][ring].astype(np.float32)
            v_ring = hsv[:, :, 2][ring].astype(np.float32)
            ring_green = (circular_hue_distance(h_ring, green_hue) <= green_tol) & (s_ring > 30) & (v_ring > 20)
            ring_green_ratio = float(np.mean(ring_green))
        else:
            ring_green_ratio = 0.0

        area_ratio = area / img_area

        # Score: component is usually less-green than board, detailed edges,
        # compact enough, and not glued to image border.
        score = (
            1.7 * (1.0 - green_ratio)
            + 1.35 * ring_green_ratio
            + 0.65 * edge_density
            + 0.35 * (mean_sat / 255.0)
            + 0.25 * desc["solidity"]
            + 0.20 * desc["extent"]
            + 0.20 * desc["center_overlap"]
            + 0.40 * min(max((area_ratio - min_component_area_ratio) / 0.15, 0.0), 1.0)
            - 1.3 * desc["border_touch_ratio"]
        )

        if score > best_score:
            best_score = score
            best_mask = mask_u8

    if best_mask is None:
        raise RuntimeError("No suitable SAM mask selected. Try another checkpoint/model-type.")
    return best_mask


def draw_debug_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    color = np.zeros_like(img_bgr)
    color[:, :] = (0, 0, 255)
    alpha = 0.35
    sel = mask > 0
    overlay[sel] = cv2.addWeighted(img_bgr[sel], 1.0 - alpha, color[sel], alpha, 0)
    return overlay


def make_cutout_rgba(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract component mask with SAM.")
    parser.add_argument("--input", default="1.jpg", help="Input image path.")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint .pth file.")
    parser.add_argument(
        "--model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM backbone type for the checkpoint.",
    )
    parser.add_argument("--mask-output", default="1_mask_sam.png", help="Output mask path.")
    parser.add_argument("--cutout-output", default="1_cutout_sam.png", help="Output RGBA cutout path.")
    parser.add_argument("--debug-output", default="1_sam_debug.png", help="Output debug overlay path.")
    parser.add_argument("--detail-expand-px", type=int, default=8, help="Fine-detail recovery radius.")
    parser.add_argument(
        "--pin-pad-search-px",
        type=int,
        default=12,
        help="Search radius for recovering pins/pads around component core.",
    )
    parser.add_argument(
        "--min-component-area-ratio",
        type=float,
        default=0.02,
        help="Reject SAM masks smaller than this image area ratio to avoid selecting letters/text.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device. Use cpu if CUDA unavailable.",
    )
    args = parser.parse_args()

    try:
        in_path = Path(args.input)
        ckpt_path = Path(args.checkpoint)
        mask_out = Path(args.mask_output)
        cutout_out = Path(args.cutout_output)
        debug_out = Path(args.debug_output)

        ensure_file(in_path)
        ensure_file(ckpt_path)

        img_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError("Failed to read input image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        log(f"Loaded image: {in_path} shape={img_bgr.shape}")

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            log("CUDA unavailable, fallback to CPU.")
            device = "cpu"

        log(f"Loading SAM model: {args.model_type} on {device} ...")
        sam = sam_model_registry[args.model_type](checkpoint=str(ckpt_path))
        sam.to(device=device)

        mask_gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=60,
        )
        masks = mask_gen.generate(img_rgb)
        if not masks:
            raise RuntimeError("SAM returned no masks.")
        log(f"SAM candidate masks: {len(masks)}")

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        g_hue, g_tol = estimate_board_green_hue(hsv)
        log(f"Estimated board green hue={g_hue:.1f}, tol={g_tol:.1f}")

        min_area_ratio = float(np.clip(args.min_component_area_ratio, 0.001, 0.30))
        log(f"Min component area ratio: {min_area_ratio:.4f}")
        mask = select_best_sam_mask(
            masks,
            img_bgr,
            g_hue,
            g_tol,
            min_component_area_ratio=min_area_ratio,
        )
        mask = recover_fine_details(img_bgr, mask, expand_px=max(args.detail_expand_px, 1))
        mask = recover_pins_and_pads(
            img_bgr,
            core_mask=mask,
            green_hue=g_hue,
            green_tol=g_tol,
            search_px=max(args.pin_pad_search_px, 2),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )

        cutout = make_cutout_rgba(img_bgr, mask)
        debug = draw_debug_overlay(img_bgr, mask)

        ok1 = cv2.imwrite(str(mask_out), mask)
        ok2 = cv2.imwrite(str(cutout_out), cutout)
        ok3 = cv2.imwrite(str(debug_out), debug)
        if not (ok1 and ok2 and ok3):
            raise RuntimeError("Failed to write output files.")

        fg_ratio = float(np.mean(mask > 0))
        log(f"Saved mask   : {mask_out}")
        log(f"Saved cutout : {cutout_out}")
        log(f"Saved debug  : {debug_out}")
        log(f"Foreground ratio: {fg_ratio:.4f}")
        return 0
    except Exception as exc:
        log(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
