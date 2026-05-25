#!/usr/bin/env python3
"""
Extract PCB component mask by leveraging green board background color.

Idea:
1) Estimate board-green color from image border area.
2) Segment board region in HSV color space.
3) Foreground = non-board region.
4) Keep main component and nearby fine structures (pins/pads).

Default input:
  - 1.jpg

Outputs:
  - 1_mask_green.png         (component mask)
  - 1_cutout_green.png       (RGBA cutout)
  - 1_board_mask_green.png   (estimated board/background mask for debugging)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def circular_hue_distance(h: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(h.astype(np.float32) - float(center))
    return np.minimum(d, 180.0 - d)


def estimate_green_model(hsv: np.ndarray, border_ratio: float = 0.12) -> Tuple[float, float, float, float]:
    """
    Estimate dominant board-green model from border pixels:
    returns (h_center, h_tol, s_min, v_min)
    """
    h, w = hsv.shape[:2]
    bx = max(int(w * border_ratio), 8)
    by = max(int(h * border_ratio), 8)

    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[:by, :] = 1
    border_mask[-by:, :] = 1
    border_mask[:, :bx] = 1
    border_mask[:, -bx:] = 1

    border_hsv = hsv[border_mask > 0]
    hh = border_hsv[:, 0].astype(np.float32)
    ss = border_hsv[:, 1].astype(np.float32)
    vv = border_hsv[:, 2].astype(np.float32)

    # Focus on moderately saturated pixels first (board color is usually not gray).
    valid = ss > np.percentile(ss, 35)
    if np.any(valid):
        hh = hh[valid]
        ss = ss[valid]
        vv = vv[valid]

    # Weighted hue histogram (saturation-weighted).
    bins = 180
    hist = np.zeros(bins, dtype=np.float64)
    for x, wgt in zip(hh.astype(np.int32), ss + 1.0):
        hist[int(x) % bins] += float(wgt)
    h_center = float(np.argmax(hist))

    hd = circular_hue_distance(hh, h_center)
    h_tol = float(np.clip(np.percentile(hd, 90) + 4.0, 8.0, 30.0))
    s_min = float(np.clip(np.percentile(ss, 20) - 8.0, 20.0, 120.0))
    v_min = float(np.clip(np.percentile(vv, 10) - 12.0, 20.0, 120.0))
    return h_center, h_tol, s_min, v_min


def keep_main_and_nearby_components(mask: np.ndarray, near_dist_px: int, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_id = int(np.argmax(areas)) + 1
    main = np.where(labels == main_id, 255, 0).astype(np.uint8)
    inv_main = np.where(main > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv_main, cv2.DIST_L2, 3)

    out = np.zeros_like(mask)
    for comp_id in range(1, num):
        comp_area = int(stats[comp_id, cv2.CC_STAT_AREA])
        comp = labels == comp_id
        if not np.any(comp):
            continue
        if comp_id == main_id:
            out[comp] = 255
            continue
        min_d = float(np.min(dist[comp]))
        if comp_area >= min_area and min_d <= near_dist_px:
            out[comp] = 255
    return out


def recover_fine_details(img_bgr: np.ndarray, seed_fg: np.ndarray, expand_px: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 35, 120)

    dil = cv2.dilate(
        seed_fg,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=max(expand_px // 2, 1),
    )
    add = np.where((edges > 0) & (dil > 0), 255, 0).astype(np.uint8)
    add = cv2.dilate(
        add,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    out = np.where((seed_fg > 0) | (add > 0), 255, 0).astype(np.uint8)
    out = cv2.morphologyEx(
        out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    out = keep_main_and_nearby_components(out, near_dist_px=max(expand_px * 3, 8), min_area=8)
    return out


def extract_component_by_green(img_bgr: np.ndarray, detail_expand_px: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_center, h_tol, s_min, v_min = estimate_green_model(hsv)
    log(
        f"Estimated green model: hue={h_center:.1f}, tol={h_tol:.1f}, "
        f"s_min={s_min:.1f}, v_min={v_min:.1f}"
    )

    hue_d = circular_hue_distance(hsv[:, :, 0], h_center)
    board_mask = (
        (hue_d <= h_tol)
        & (hsv[:, :, 1].astype(np.float32) >= s_min)
        & (hsv[:, :, 2].astype(np.float32) >= v_min)
    )
    board_u8 = np.where(board_mask, 255, 0).astype(np.uint8)

    # Clean board mask
    k = max(min(img_bgr.shape[:2]) // 140, 3)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    board_u8 = cv2.morphologyEx(board_u8, cv2.MORPH_CLOSE, ker, iterations=2)
    board_u8 = cv2.morphologyEx(board_u8, cv2.MORPH_OPEN, ker, iterations=1)

    # Component foreground = non-board.
    fg = np.where(board_u8 == 0, 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(
        fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    h, w = fg.shape[:2]
    fg = keep_main_and_nearby_components(
        fg,
        near_dist_px=max(detail_expand_px * 3, 10),
        min_area=max((h * w) // 50000, 10),
    )
    fg = recover_fine_details(img_bgr, fg, expand_px=detail_expand_px)
    return fg, board_u8


def make_cutout_rgba(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, mask])


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract component mask using green board color.")
    parser.add_argument("--input", default="1.jpg", help="Input image path.")
    parser.add_argument("--mask-output", default="1_mask_green.png", help="Output component mask path.")
    parser.add_argument("--cutout-output", default="1_cutout_green.png", help="Output RGBA cutout path.")
    parser.add_argument(
        "--board-mask-output",
        default="1_board_mask_green.png",
        help="Output estimated board mask path.",
    )
    parser.add_argument(
        "--detail-expand-px",
        type=int,
        default=8,
        help="Detail expansion radius for recovering pins/pads.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    mask_out = Path(args.mask_output)
    cutout_out = Path(args.cutout_output)
    board_out = Path(args.board_mask_output)

    try:
        ensure_file(in_path)
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to read image.")

        log(f"Loaded image: {in_path} shape={img.shape}")
        comp_mask, board_mask = extract_component_by_green(
            img_bgr=img,
            detail_expand_px=max(args.detail_expand_px, 1),
        )
        cutout = make_cutout_rgba(img, comp_mask)

        ok1 = cv2.imwrite(str(mask_out), comp_mask)
        ok2 = cv2.imwrite(str(cutout_out), cutout)
        ok3 = cv2.imwrite(str(board_out), board_mask)
        if not (ok1 and ok2 and ok3):
            raise RuntimeError("Failed to write output files.")

        fg_ratio = float(np.mean(comp_mask > 0))
        board_ratio = float(np.mean(board_mask > 0))
        log(f"Saved component mask : {mask_out}")
        log(f"Saved cutout image   : {cutout_out}")
        log(f"Saved board mask     : {board_out}")
        log(f"Foreground ratio: {fg_ratio:.4f}")
        log(f"Board ratio     : {board_ratio:.4f}")
        return 0
    except Exception as exc:
        log(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
