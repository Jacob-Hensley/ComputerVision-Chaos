# analyze.py
# Date Created: 2024-06-01
# Last Modified: 2024-06-01
# Description: This file is used to analyze the video frames extracted from main.py. It will use methods to detect the
# pendulum pivot and bobs, estimate the angles of the pendulum, and then analyze the onset of chaos in the system (when 
# either pendulum flips beyond 180 degrees).
import numpy as np
import cv2


def detect_pivot(frames):
    """Detect the fixed pivot and estimate L1 (length of first pendulum)."""
    if not frames:
        raise ValueError("No frames provided")

    pivot_candidates = []
    l1_candidates = []

    for frame in frames[:30]:           # Use first 30 frames for stability
        result = _detect_pivot_and_l1_in_frame(frame)
        if result is not None:
            pivot, l1 = result
            pivot_candidates.append(pivot)
            l1_candidates.append(l1)

    if not pivot_candidates:
        raise ValueError("Could not detect pivot in any frame")

    # Final pivot = median
    pivot_array = np.array(pivot_candidates)
    final_pivot = tuple(np.median(pivot_array, axis=0).astype(int))

    # Final L1 = median of candidates
    L1 = np.median(l1_candidates) if l1_candidates else 150.0
    print(f"Detected pivot at: {final_pivot}")
    print(f"Estimated fixed length L1 (pivot → Bob 1): {L1:.1f} pixels")

    return final_pivot, L1


def _detect_pivot_and_l1_in_frame(frame):
    """Helper: returns (pivot, l1) or None"""
    # Detect bobs
    circles = cv2.HoughCircles(
        frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=25, minRadius=8, maxRadius=25
    )
    
    if circles is None or len(circles[0]) < 2:
        return None
    
    circles = np.round(circles[0, :]).astype("int")
    bob_centers = circles[:, :2]

    # Detect lines (rods)
    edges = cv2.Canny(frame, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=40,
        minLineLength=30, maxLineGap=8
    )
    
    if lines is None:
        return None

    # Find pivot: endpoint not close to any bob
    min_dist_to_bob = 15
    pivot = None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints = np.array([[x1, y1], [x2, y2]])
        
        for pt in endpoints:
            dists = np.linalg.norm(bob_centers - pt, axis=1)
            if np.all(dists > min_dist_to_bob):
                pivot = tuple(pt)
                break
        if pivot is not None:
            break

    if pivot is None:
        return None

    # Estimate L1: distance from pivot to the closer bob in this frame
    pivot_arr = np.array(pivot)
    dists = np.linalg.norm(bob_centers - pivot_arr, axis=1)
    l1 = np.min(dists)          # closer bob should be Bob 1

    return pivot, l1


def detect_bobs(frames, pivot=None, L1=None):
    """Detect and correctly order bobs using fixed L1 length + tolerance."""
    if pivot is None or L1 is None:
        raise ValueError("Both pivot and L1 must be provided")

    pivot_arr = np.array(pivot)
    tolerance = 0.10 * L1          # ±10% tolerance
    bob_positions = []

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
                
                # Find detection closest to expected L1 (±10%)
                diff_to_L1 = np.abs(dists - L1)
                idx_bob1 = np.argmin(diff_to_L1)
                
                # Safety check: if the best match is still way off, use closest point as fallback
                if diff_to_L1[idx_bob1] > tolerance:
                    idx_bob1 = np.argmin(dists)   # fallback to closest point
                
                # The other point is Bob 2
                idx_bob2 = 0 if idx_bob1 == 1 else 1
                
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
    """Compute θ1 and θ2 in radians from downward vertical."""
    angles = []
    px, py = pivot_coords
    
    for bobs in bob_coords:
        if bobs is None or len(bobs) < 2:
            angles.append((np.nan, np.nan))
            continue
            
        bob1 = bobs[0]
        bob2 = bobs[1]
        
        theta1 = np.arctan2(bob1[0] - px, bob1[1] - py)
        theta2 = np.arctan2(bob2[0] - px, bob2[1] - py)
        
        angles.append((theta1, theta2))
    
    return angles


def analyze_chaos(angles):
    """Return (chaos_detected, first_chaos_frame_index)"""
    for i, (theta1, theta2) in enumerate(angles):
        if np.isnan(theta1) or np.isnan(theta2):
            continue
        if abs(theta1) > np.pi * 0.95 or abs(theta2) > np.pi * 0.95:
            return True, i
    
    return False, -1