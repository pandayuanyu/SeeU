import argparse
import csv
import os
from typing import Dict, Tuple

import cv2
import numpy as np

def sampson_dist(F: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """Sampson distance (approx. reprojection error) for fundamental matrix."""
    x1h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2h = np.hstack([x2, np.ones((x2.shape[0], 1))])
    Fx1 = (F @ x1h.T).T                   # N x 3
    Ftx2 = (F.T @ x2h.T).T                # N x 3
    num = np.sum(x2h * (F @ x1h.T).T, axis=1) ** 2  # (x2^T F x1)^2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / (denom + 1e-12)

def match_sift(img1: np.ndarray, img2: np.ndarray, nfeatures=4000, ratio=0.75):
    """SIFT + ratio test + cross-check."""
    if img1.ndim == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        g1 = img1
    if img2.ndim == 3:
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        g2 = img2

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None, None, [], 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    m12 = bf.knnMatch(d1, d2, k=2)
    m21 = bf.knnMatch(d2, d1, k=2)

    def ratio_f(ms, r=ratio):
        return [m[0] for m in ms if len(m) == 2 and m[0].distance < r * m[1].distance]

    m12 = ratio_f(m12)
    m21 = ratio_f(m21)
    idx21 = {(m.trainIdx, m.queryIdx) for m in m21}
    good = [m for m in m12 if (m.queryIdx, m.trainIdx) in idx21]

    if len(good) < 8:
        return k1, k2, [], len(good)

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    return k1, k2, (pts1, pts2), len(good)

def eval_pair_noK(img1: np.ndarray, img2: np.ndarray) -> Dict:
    """Evaluate one frame-pair with only two-view geometry (no intrinsics)."""
    k1, k2, pts, n_good = match_sift(img1, img2)
    if not pts:
        return {"ok": False, "reason": "too_few_matches", "n_matches": n_good}

    pts1, pts2 = pts

    # --- Fundamental matrix (RANSAC) ---
    F, maskF = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.999
    )
    if F is None or maskF is None:
        return {"ok": False, "reason": "F_fail", "n_matches": n_good}

    inF = maskF.ravel().astype(bool)
    n_inF = int(inF.sum())
    if n_inF < 8:
        return {"ok": False, "reason": "too_few_inliers_F", "n_matches": n_good, "n_inliers_F": n_inF}

    P1F, P2F = pts1[inF], pts2[inF]
    sd = sampson_dist(F, P1F, P2F)
    sd_px = np.sqrt(sd)

    # --- Homography (RANSAC) for planarity/pure rotation check ---
    H, maskH = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=2.0, confidence=0.999)
    n_inH = int(maskH.sum()) if maskH is not None else 0

    # --- Model preference: does F explain more than H? ---
    parallax_score = n_inF / max(n_inF + n_inH, 1)  # ∈(0,1); >0.5 => F preferred

    metrics = {
        "ok": True,
        "n_matches": int(n_good),
        "n_inliers_F": n_inF,
        "inlier_ratio_F": float(n_inF / max(n_good, 1)),
        "sampson_median_px": float(np.median(sd_px)),
        "sampson_mean_px": float(np.mean(sd_px)),
        "sampson_<1px_%": float(np.mean(sd_px < 1.0)),
        "sampson_<2px_%": float(np.mean(sd_px < 2.0)),
        "n_inliers_H": int(n_inH),
        "F_minus_H_inliers": int(n_inF - n_inH),
        "parallax_score": float(parallax_score),
        "F_beats_H": bool(n_inF > n_inH + 10),  
    }
    return metrics

def read_video_frames(path: str, max_frames=None, stride=5):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_a", type=str, required=True, help="input/reference video path")
    ap.add_argument("--video_b", type=str, required=True, help="generated/novel-view video path")
    ap.add_argument("--stride", type=int, default=5, help="frame sampling stride")
    ap.add_argument("--max_pairs", type=int, default=None, help="limit number of pairs (after stride)")
    ap.add_argument("--resize_w", type=int, default=None, help="optional width to resize both videos for speed")
    ap.add_argument("--out_csv", type=str, default="geom_eval_noK_perframe.csv")
    args = ap.parse_args()

    A = read_video_frames(args.video_a, max_frames=args.max_pairs, stride=args.stride)
    B = read_video_frames(args.video_b, max_frames=args.max_pairs, stride=args.stride)
    n = min(len(A), len(B))
    if n == 0:
        raise RuntimeError("No frames read. Check paths or stride/max_pairs.")
    A, B = A[:n], B[:n]

    if args.resize_w is not None:
        def resize_seq(seq):
            out = []
            for im in seq:
                h, w = im.shape[:2]
                new_w = args.resize_w
                new_h = int(h * (new_w / w))
                out.append(cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA))
            return out
        A = resize_seq(A)
        B = resize_seq(B)

    rows = []
    for i, (fa, fb) in enumerate(zip(A, B)):
        m = eval_pair_noK(fa, fb)
        m["pair_idx"] = i
        rows.append(m)

    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    val = [r for r in rows if r.get("ok")]
    if not val:
        print("No valid pairs.")
        return

    def med(k): return float(np.median([r[k] for r in val]))
    def mean(k): return float(np.mean([r[k] for r in val]))

    win_rate = float(np.mean([1.0 if r.get("F_beats_H") else 0.0 for r in val]))
    print("\n=== Video-level summary (no intrinsics) ===")
    print(f"#pairs_valid: {len(val)} / {len(rows)}")
    print(f"F inlier ratio (median): {med('inlier_ratio_F'):.3f}")
    print(f"Sampson median (px):    {med('sampson_median_px'):.3f}")
    print(f"<1px% median:           {med('sampson_<1px_%'):.3f}")
    print(f"<2px% median:           {med('sampson_<2px_%'):.3f}")
    print(f"Parallax score median:  {med('parallax_score'):.3f}")
    print(f"F beats H (win rate):   {win_rate:.3f}")
    print(f"Per-frame CSV saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
