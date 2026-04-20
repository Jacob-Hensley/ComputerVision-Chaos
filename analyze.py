# analyze.py
# Date Created: 4-16-2026
# Last Modified: 4-17-2026
# Description: This file is used to analyze the video frames extracted from main.py. It will use methods to detect the
# pendulum pivot and bobs, estimate the angles of the pendulum, and then analyze the onset of chaos in the system (when 
# either pendulum flips beyond 180 degrees).

# Import necessary libraries
import numpy as np
import cv2

# Function for detecting the fixed position pivot of the system
def detect_pivot(frames):
    # If frames is empty, raise an error
    if not frames:
        raise ValueError("No frames provided")

    # Initialize list to hold multiple frames to get stable detection of pivot 
    # and L1 (pivot → Bob 1 length)
    pivot_candidates = []
    l1_candidates = []

    # Loop through first 30 frames to get reliable location data
    for frame in frames[:30]:
        # Call helper function to detect pivot and L1 in this frame
        result = detect_pivot_and_l1_in_frame(frame)
        # If we got a valid detection, store it for later median calculation
        if result is not None:
            pivot, l1 = result
            pivot_candidates.append(pivot)
            l1_candidates.append(l1)

    # If we couldn't find any valid pivot candidates, raise an error
    if not pivot_candidates:
        raise ValueError("Could not detect pivot in any frame")

    # Final pivot = median of candidates to reduce noise
    pivot_array = np.array(pivot_candidates)
    final_pivot = tuple(np.median(pivot_array, axis=0).astype(int))

    # Final L1 location = median of candidates, with a reasonable default if none found
    L1 = np.median(l1_candidates) if l1_candidates else 150.0
    print(f"Detected pivot at: {final_pivot}")
    print(f"Estimated fixed length L1 (pivot → Bob 1): {L1:.1f} pixels")

    # Return the pivot and L1 positions for use in bob detection and angle estimation
    return final_pivot, L1

# Helper function to detect pivot and L1 in a single frame
def detect_pivot_and_l1_in_frame(frame):
    # Detect bobs using Hough Circle detector
    circles = cv2.HoughCircles(
        frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=50, param2=25, minRadius=8, maxRadius=25
    )
    
    # If there are not at least 2 circles detected, we cannot reliably
    # find the pivot, so return None
    if circles is None or len(circles[0]) < 2:
        return None
    
    # Circles are equal to the detected bobs, we will use their centers 
    # to find the pivot
    circles = np.round(circles[0, :]).astype("int")
    bob_centers = circles[:, :2]

    # Detect the rods of the pendulum using Hough Line Transform to find
    # the pivot point
    edges = cv2.Canny(frame, 30, 100)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180, threshold=40,
        minLineLength=30, maxLineGap=8
    )
    
    # If no lines are detected, we cannot find the pivot, so return None
    if lines is None:
        return None

    # Initialize distance threshold for considering a line endpoint as a pivot candidate
    min_dist_to_bob = 15
    # Initialize variable to hold the best pivot candidate
    pivot = None

    # Loop through detected lines and check if their endpoints are reasonably 
    # far from the detected bobs (to avoid mistaking a bob for the pivot)
    for line in lines:
        # Each line is represented as its endpoints (x1, y1, x2, y2)
        x1, y1, x2, y2 = line[0]
        endpoints = np.array([[x1, y1], [x2, y2]])
        
        # Check if either endpoint is a valid pivot candidate by being sufficiently
        # far enough from all bob centers
        for pt in endpoints:
            dists = np.linalg.norm(bob_centers - pt, axis=1)
            if np.all(dists > min_dist_to_bob):
                pivot = tuple(pt)
                break
        if pivot is not None:
            break

    # If pivot location is not selected, return None
    if pivot is None:
        return None

    # Estimate L1: distance from pivot to the closer bob in this frame
    pivot_arr = np.array(pivot)
    dists = np.linalg.norm(bob_centers - pivot_arr, axis=1)
    # Closer bob should be Bob 1
    l1 = np.min(dists)          

    # Return th detected pivot and L1 bob length for this frame
    return pivot, l1

# Function for detecting the bob (pendulum masses)
def detect_bobs(frames, pivot=None, L1=None):
    # If pivot or L1 is not provided, we cannot reliably detect bobs, so raise an error
    if pivot is None or L1 is None:
        raise ValueError("Both pivot and L1 must be provided")

    # Convert pivot to numpy array for distance calculations
    pivot_arr = np.array(pivot)
    # Tolerance for L1 matching: allow some deviation due to perspective and detection noise
    tolerance = 0.15 * L1
    
    # Initialize list to hold bob position candidates for each frame
    bob_positions = []
    # Initialize variable to hold previous Bob 1 position for temporal smoothing
    prev_bob1 = None

    # Loop through each frame and detect bobs using Hough Circle detector
    for frame in frames:
        circles = cv2.HoughCircles(
            frame, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=50, param2=25, minRadius=8, maxRadius=25
        )
        
        # If we detected at least 2 circles, we can try to identify which ones are
        # Bob 1 and Bob 2 based on their distance from the pivot and temporal consistency
        if circles is not None and len(circles[0]) >= 2:
            centers = np.round(circles[0, :]).astype("int")[:, :2]
            dists = np.linalg.norm(centers - pivot_arr, axis=1)

            # Identify Bob 1 as the circle closest to the expected L1 distance from the pivot
            idx_bob1 = np.argmin(np.abs(dists - L1))
            
            # Fallback if the distance betwee the closest circle and L1 is too large
            if np.abs(dists[idx_bob1] - L1) > tolerance:
                idx_bob1 = np.argmin(dists)

            # Light temporal smoothing: prefer the candidate closer to previous Bob1
            if prev_bob1 is not None and len(centers) == 2:
                dist_to_prev = np.linalg.norm(centers - prev_bob1, axis=1)
                if dist_to_prev[1 - idx_bob1] < dist_to_prev[idx_bob1] * 0.7:
                    idx_bob1 = 1 - idx_bob1   # swap if the other is much closer to previous

            # Bob 2 is the other circle detected in this frame
            idx_bob2 = 0 if idx_bob1 == 1 else 1

            # Store the detected bob positions for this frame, ordered as [Bob 1, Bob 2]
            bob1 = centers[idx_bob1]
            bob2 = centers[idx_bob2]
            
            # Store the ordered bob positions for this frame
            ordered_bobs = np.array([bob1, bob2])
            bob_positions.append(ordered_bobs)
            # Update previous Bob 1 position for next frame's temporal smoothing
            prev_bob1 = bob1.copy()
        # If we couldn't detect 2 valid circles, append None for this frame to maintain alignment
        else:
            bob_positions.append(None)
            prev_bob1 = None
    
    # Return list of bob positions for the frames
    return bob_positions

# Function for estimating angles θ1 and θ2 from the detected pivot and bob positions
def estimate_angles(pivot_coords, bob_coords):
    # Initialize empty list to hold calculated angles for each frame
    angles = []
    # Initialize pivot coordinates for calculations
    px, py = pivot_coords
    
    # Loop through each frame's bob positions and calculate angles
    for bobs in bob_coords:
        if bobs is None or len(bobs) < 2:
            angles.append((np.nan, np.nan))
            continue

        # Bob 1 and Bob 2 positions for this frame    
        bob1 = bobs[0]
        bob2 = bobs[1]
        
        # Create vectors for angle calculations:
        # Vector 1: Pivot → Bob 1
        v1x = bob1[0] - px
        v1y = bob1[1] - py
        theta1 = np.arctan2(v1x, v1y)
        
        # Vector 2: Bob1 → Bob2
        v2x = bob2[0] - bob1[0]
        v2y = bob2[1] - bob1[1]
        theta2_abs_from_pivot = np.arctan2(v2x, v2y)
        theta2 = theta2_abs_from_pivot - theta1
        
        # Normalize angle 2 to be within -π to π for better interpretation
        theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
        
        # Append the calculated angles for this frame to the list
        angles.append((theta1, theta2))
    
    # Return the list of angles for all frames
    return angles

# Function for analyzing the angles to detect the onset of chaos in the system
def analyze_chaos(angles):
    # Initialize variable to hold the index of the first frame where chaos is detected
    first_chaos_frame = -1
    # Define a threshold for detecting a flip (e.g., angle exceeding ±175 degrees)
    threshold = np.pi * 0.97

    for i, (theta1, theta2) in enumerate(angles):
        if np.isnan(theta1) or np.isnan(theta2):
            continue
            
        # Primary check: clear flip beyond threshold
        if abs(theta1) > threshold or abs(theta2) > threshold:
            first_chaos_frame = i
            break
            
        # Secondary check for very fast flips (angle jumps over the top)
        # This catches cases where the angle goes from e.g. +170° to -170° in one frame
        if i > 0:
            prev_theta1, prev_theta2 = angles[i-1]
            if not np.isnan(prev_theta1) and not np.isnan(prev_theta2):
                # Detect crossing of the ±180° line
                if (abs(prev_theta1) > np.pi * 0.85 and abs(theta1) > np.pi * 0.85 and 
                    np.sign(prev_theta1) != np.sign(theta1)) or \
                   (abs(prev_theta2) > np.pi * 0.85 and abs(theta2) > np.pi * 0.85 and 
                    np.sign(prev_theta2) != np.sign(theta2)):
                    first_chaos_frame = i
                    break

    has_chaos = first_chaos_frame != -1
    return has_chaos, first_chaos_frame