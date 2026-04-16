# analyze.py
# Date Created: 2024-06-01
# Last Modified: 2024-06-01
# Description: This file is used to analyze the video frames extracted from main.py. It will use methods to detect the
# pendulum pivot and bobs, estimate the angles of the pendulum, and then analyze the onset of chaos in the system (when 
# either pendulum flips beyond 180 degrees).

import numpy as np
import cv2


def detect_pivot(frames):
    """Detect the fixed pivot using the first 10 frames."""
    if not frames:
        raise ValueError("No frames provided")

    pivot_candidates = []
    
    for frame in frames[:10]:
        pivot = _detect_pivot_in_single_frame(frame)
        if pivot is not None:
            pivot_candidates.append(pivot)
    
    if not pivot_candidates:
        raise ValueError("Could not detect pivot in any frame")
    
    # Median for robustness
    pivot_array = np.array(pivot_candidates)
    final_pivot = tuple(np.median(pivot_array, axis=0).astype(int))
    
    print(f"Detected pivot at: {final_pivot}")
    return final_pivot


def _detect_pivot_in_single_frame(frame):
    """Helper: Detect pivot in one grayscale frame."""
    # 1. Detect bobs
    circles = cv2.HoughCircles(
        frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=25, minRadius=8, maxRadius=25
    )
    
    if circles is None or len(circles[0]) < 2:
        return None
    
    circles = np.round(circles[0, :]).astype("int")
    bob_centers = circles[:, :2]

    # 2. Detect lines (rods)
    edges = cv2.Canny(frame, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=40,
        minLineLength=30, maxLineGap=8
    )
    
    if lines is None:
        return None

    # 3. Find endpoint farthest from any bob → pivot
    min_dist_to_bob = 15

    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints = np.array([[x1, y1], [x2, y2]])
        
        for pt in endpoints:
            dists = np.linalg.norm(bob_centers - pt, axis=1)
            if np.all(dists > min_dist_to_bob):
                return tuple(pt)
    
    return None


def detect_bobs(frames, pivot=None):
    """Detect the two bobs in every frame.
    Uses the fixed-length constraint (Bob 1 is always ~L1 from pivot)
    to correctly label them even when Bob 2 swings inside.
    """
    if pivot is None:
        raise ValueError("Pivot must be provided for correct bob ordering")

    pivot_arr = np.array(pivot)
    bob_positions = []

    # === Step 1: Estimate fixed length L1 from first 30 frames ===
    l1_candidates = []
    for frame in frames[:30]:                     # early frames are usually clean
        circles = cv2.HoughCircles(
            frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=25, minRadius=8, maxRadius=25
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            centers = circles[:, :2]
            if len(centers) >= 2:
                dists = np.linalg.norm(centers - pivot_arr, axis=1)
                closer_dist = np.min(dists)          # the closer one is almost always Bob 1 early on
                l1_candidates.append(closer_dist)

    if l1_candidates:
        L1 = np.median(l1_candidates)
        print(f"Estimated fixed length L1 (pivot → Bob 1): {L1:.1f} pixels")
    else:
        L1 = 150.0  # fallback (very unlikely)
        print("Warning: Could not estimate L1, using fallback 150 px")

    # === Step 2: Detect and correctly label bobs in EVERY frame ===
    for frame in frames:
        circles = cv2.HoughCircles(
            frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=25, minRadius=8, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            centers = circles[:, :2]
            
            if len(centers) >= 2:
                dists = np.linalg.norm(centers - pivot_arr, axis=1)
                
                # Find which detected point has distance closest to L1 → that's Bob 1
                diff_to_L1 = np.abs(dists - L1)
                idx_bob1 = np.argmin(diff_to_L1)
                
                # The other point is Bob 2
                idx_bob2 = 0 if idx_bob1 == 1 else 1   # works for exactly 2 detections
                
                bob1 = centers[idx_bob1]
                bob2 = centers[idx_bob2]
                
                ordered_bobs = np.array([bob1, bob2])
                bob_positions.append(ordered_bobs)
            else:
                bob_positions.append(None)
        else:
            bob_positions.append(None)
    
    return bob_positions


def estimate_angles(pivot_coords, bob_coords):
    """Compute θ1 and θ2 (radians) from vertical for each frame."""
    angles = []
    px, py = pivot_coords
    
    for bobs in bob_coords:
        if bobs is None or len(bobs) < 2:
            angles.append((np.nan, np.nan))
            continue
            
        bob1 = bobs[0]
        bob2 = bobs[1]
        
        # arctan2(x, y) → angle from downward vertical
        theta1 = np.arctan2(bob1[0] - px, bob1[1] - py)
        theta2 = np.arctan2(bob2[0] - px, bob2[1] - py)
        
        angles.append((theta1, theta2))
    
    return angles


def analyze_chaos(angles):
    """Return (chaos_detected, first_chaos_frame_index)"""
    for i, (theta1, theta2) in enumerate(angles):
        if np.isnan(theta1) or np.isnan(theta2):
            continue
        # Chaos when any arm goes beyond ~±180°
        if abs(theta1) > np.pi * 0.95 or abs(theta2) > np.pi * 0.95:
            return True, i
    
    return False, -1