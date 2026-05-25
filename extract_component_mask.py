#!/usr/bin/env python3
"""
Auto-extract PCB component from an image and output mask.

Default input:
  - 1.jpg

Outputs:
  - 1_mask.png   (binary mask, component=255, background=0)
  - 1_cutout.png (RGBA cutout with transparent background)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    # 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def keep_main_and_nearby_components(
    mask: np.ndarray, near_dist_px: int, min_area: int
) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    main_id = int(np.argmax(areas)) + 1
    main_mask = np.where(labels == main_id, 255, 0).astype(np.uint8)

    # Distance-to-main map: main pixels are 0, farther pixels are larger.
    inv_main = np.where(main_mask > 0, 0, 255).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv_main, cv2.DIST_L2, 3)

    out = np.zeros_like(mask)
    for comp_id in range(1, num):
        comp_area = int(stats[comp_id, cv2.CC_STAT_AREA])
        comp_pix = labels == comp_id
        if not np.any(comp_pix):
            continue
        min_dist = float(np.min(dist_map[comp_pix]))
        if comp_id == main_id or (comp_area >= min_area and min_dist <= near_dist_px):
            out[comp_pix] = 255
    return out


def recover_fine_pins_and_pads(
    img_bgr: np.ndarray,
    seed_mask: np.ndarray,
    pin_expand_px: int,
) -> np.ndarray:
    """
    Recover thin leads/pads using edge + color consistency around the seed mask.
    """
    h, w = seed_mask.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    fg_seed = seed_mask > 0
    if np.count_nonzero(fg_seed) < 20:
        return seed_mask

    # Build a local search ring around current foreground.
    r_outer = max(pin_expand_px * 3, 6)
    r_inner = max(pin_expand_px, 2)
    ker_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_outer + 1, 2 * r_outer + 1))
    ker_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_inner + 1, 2 * r_inner + 1))
    outer = cv2.dilate(seed_mask, ker_outer, iterations=1)
    inner = cv2.dilate(seed_mask, ker_inner, iterations=1)
    ring = (outer > 0) & (inner == 0)

    # Foreground color model in Lab.
    fg_lab = lab[fg_seed]
    mu = np.mean(fg_lab, axis=0)
    sigma = np.std(fg_lab, axis=0) + 1e-6
    z = np.abs((lab.astype(np.float32) - mu[None, None, :]) / sigma[None, None, :])
    color_ok = (np.mean(z, axis=2) < 2.6)  # robust but not too strict

    # Edge candidates for thin structures.
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 35, 120) > 0

    # Candidate pixels: close to seed, on edge, and color-consistent.
    cand = ring & edges & color_ok
    cand_u8 = np.where(cand, 255, 0).astype(np.uint8)
    cand_u8 = cv2.dilate(
        cand_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    out = np.where((seed_mask > 0) | (cand_u8 > 0), 255, 0).astype(np.uint8)
    out = cv2.morphologyEx(
        out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    # Keep nearby connected structures; avoid picking far noisy pads.
    near_dist_px = max(pin_expand_px * 4, 10)
    min_area = max((h * w) // 60000, 6)
    out = keep_main_and_nearby_components(out, near_dist_px=near_dist_px, min_area=min_area)
    return out


def extract_mask_grabcut(
    img_bgr: np.ndarray,
    margin_ratio: float = 0.08,
    keep_fine_detail: bool = True,
    pin_expand_px: int = 6,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    margin_x = max(int(w * margin_ratio), 5)
    margin_y = max(int(h * margin_ratio), 5)

    rect = (
        margin_x,
        margin_y,
        max(w - 2 * margin_x, 1),
        max(h - 2 * margin_y, 1),
    )

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr,
        mask,
        rect,
        bgd_model,
        fgd_model,
        iterCount=6,
        mode=cv2.GC_INIT_WITH_RECT,
    )

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Conservative cleanup: avoid eating thin solder pins.
    k = max(min(h, w) // 160, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)

    near_dist_px = max(pin_expand_px * 2, 6)
    min_area = max((h * w) // 30000, 12)
    fg = keep_main_and_nearby_components(fg, near_dist_px=near_dist_px, min_area=min_area)

    if keep_fine_detail:
        # Recover thin leads/pads via edge-color guided expansion.
        fg = recover_fine_pins_and_pads(
            img_bgr,
            seed_mask=fg,
            pin_expand_px=pin_expand_px,
        )
    return fg


def make_cutout_rgba(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr)
    alpha = mask
    return cv2.merge([b, g, r, alpha])


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract component and output mask.")
    parser.add_argument("--input", default="1.jpg", help="Input image path.")
    parser.add_argument("--mask-output", default="1_mask.png", help="Output mask path.")
    parser.add_argument(
        "--cutout-output",
        default="1_cutout.png",
        help="Output cutout RGBA image path.",
    )
    parser.add_argument(
        "--disable-fine-detail",
        action="store_true",
        help="Disable edge-guided fine-detail recovery for thin pins.",
    )
    parser.add_argument(
        "--pin-expand-px",
        type=int,
        default=6,
        help="Expansion radius in pixels for recovering thin solder pins.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    mask_path = Path(args.mask_output)
    cutout_path = Path(args.cutout_output)

    try:
        ensure_file(in_path)
        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to read image. Check image format/path.")

        log(f"Loaded image: {in_path}  shape={img.shape}")
        mask = extract_mask_grabcut(
            img,
            keep_fine_detail=not args.disable_fine_detail,
            pin_expand_px=max(args.pin_expand_px, 1),
        )
        cutout = make_cutout_rgba(img, mask)

        ok_mask = cv2.imwrite(str(mask_path), mask)
        ok_cut = cv2.imwrite(str(cutout_path), cutout)
        if not ok_mask or not ok_cut:
            raise RuntimeError("Failed to save output files.")

        fg_ratio = float(np.mean(mask > 0))
        log(f"Saved mask   : {mask_path}")
        log(f"Saved cutout : {cutout_path}")
        log(f"Foreground ratio: {fg_ratio:.4f}")
        return 0
    except Exception as exc:
        log(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
