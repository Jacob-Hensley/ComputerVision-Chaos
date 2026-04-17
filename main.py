# main.py
# Date Created: 2024-06-01
# Last Modified: 2024-06-01
# Description: Main entry point for the application. This file is used to load and preprocess
# the .mp4 video file, extract the frames, and then pass them to another file for analysis.

import cv2
import os
from analyze import *

# load the video file and reduce the size of the video frames to 224x224
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (500, 500))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smoothed_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        frames.append(smoothed_frame)
    cap.release()
    return frames

# play the video frames
def play_video(frames):
    for frame in frames:
        cv2.imshow("Video", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# function to draw the detected pivot and bobs on the video frames
def visualize_detections(frames, pivot, bob_positions, angles=None, delay=30):
    """
    Plays the video with overlaid detections:
      - Green circle + text for the fixed pivot
      - Red circles for the two bobs
      - Lines for the pendulum rods
      - Angle values (in degrees) displayed on screen
    """
    for i, frame in enumerate(frames):
        # Convert grayscale back to BGR so we can draw colors
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if pivot is not None:
            px, py = pivot
            # Draw pivot
            cv2.circle(display, (px, py), 6, (0, 255, 0), -1)  # Green filled
            cv2.circle(display, (px, py), 8, (0, 255, 0), 2)
            cv2.putText(display, "PIVOT", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if bob_positions[i] is not None:
            bobs = bob_positions[i]
            for j, (bx, by) in enumerate(bobs):
                color = (0, 0, 255) if j == 0 else (255, 0, 255)  # Red / Magenta
                cv2.circle(display, (bx, by), 10, color, -1)
                cv2.circle(display, (bx, by), 12, color, 2)

                # Draw rod from pivot to first bob, and first to second bob
                if j == 0 and pivot is not None:
                    cv2.line(display, pivot, (bx, by), (0, 255, 255), 3)  # Cyan rod 1
                elif j == 1 and len(bobs) > 1 and pivot is not None:
                    cv2.line(display, tuple(bobs[0]), (bx, by), (0, 255, 255), 3)  # Cyan rod 2

        # Display angles if provided
        if angles is not None and i < len(angles):
            theta1, theta2 = angles[i]
            if not np.isnan(theta1):
                deg1 = np.degrees(theta1)
                cv2.putText(display, f"θ1: {deg1:.1f}°", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not np.isnan(theta2):
                deg2 = np.degrees(theta2)
                cv2.putText(display, f"θ2: {deg2:.1f}°", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Double Pendulum Analysis", display)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):   # Press 'p' to pause
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "dpsim1.mp4"
    frames = preprocess_video(video_path)
    print(f"Loaded {len(frames)} frames")
    
    # Analysis pipeline
    pivot, L1 = detect_pivot(frames)
    bob_positions = detect_bobs(frames, pivot=pivot, L1=L1)
    angles = estimate_angles(pivot, bob_positions)
    
    has_chaos, chaos_frame = analyze_chaos(angles)
    print(f"Chaos detected: {has_chaos}")
    if has_chaos:
        print(f"First chaos at frame: {chaos_frame}")

    # === VISUALIZATION WITH MARKERS ===
    print("Starting playback with detections...")
    visualize_detections(frames, pivot, bob_positions, angles, delay=30)

    print("Analysis complete.")