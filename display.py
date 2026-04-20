# display.py
# Date Created: 2024-06-01
# Last Modified: 2024-06-01
# Description: This file contains functions for displaying and visualizing the detected pendulum features,
# including the pivot, bob positions, connecting rods, angles, and chaos detection status overlaid on the video frames.

import cv2
import numpy as np


# play the video frames
def play_video(frames):
    # Loop through all frames and display
    for frame in frames:
        cv2.imshow("Video", frame)
        # Press 'q' to quit playback
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Draw the detected pivot, bobs, angles, and chaotic-status on 
# the video frames and save it to a .mp4 file
def visualize_detections(frames, pivot, bob_positions, angles=None, 
                         first_chaos_frame=-1, delay=25, output_path='output.mp4'):
    
    # Initialize chaos flag
    is_chaotic = False
    
    # Initialize video writer
    out = None
    if len(frames) > 0:
        frame_height, frame_width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1000 / delay  # Calculate fps from delay in milliseconds
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through each frame
    for i, frame in enumerate(frames):
        # Convert grayscale to BGR for color annotations
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Draw pivot at pivot position
        if pivot is not None:
            px, py = pivot
            cv2.circle(display, (px, py), 8, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 10, (0, 255, 0), 2)

        # Draw pendulum bobs (or masses) and connecting rods
        if bob_positions[i] is not None:
            # Draw bobs
            bobs = bob_positions[i]
            for j, (bx, by) in enumerate(bobs):
                color = (0, 0, 255) if j == 0 else (255, 0, 255)
                cv2.circle(display, (int(bx), int(by)), 10, color, -1)
                cv2.circle(display, (int(bx), int(by)), 12, color, 2)

            # Draw rods
            if pivot is not None:
                cv2.line(display, pivot, tuple(bobs[0].astype(int)), (0, 255, 255), 4)
            if len(bobs) > 1:
                cv2.line(display, tuple(bobs[0].astype(int)), tuple(bobs[1].astype(int)), (0, 255, 255), 4)

        # Display angles in window
        if angles is not None and i < len(angles):
            theta1, theta2 = angles[i]

            # Format angles with large font and clear signs, handle NaN cases
            if np.isnan(theta1):
                angle1_text = "Angle 1 = N/A"
            else:
                deg1 = np.degrees(theta1)
                angle1_text = f"Angle 1 = {deg1:+7.1f}"

            if np.isnan(theta2):
                angle2_text = "Angle 2 = N/A"
            else:
                deg2 = np.degrees(theta2)
                angle2_text = f"Angle 2 = {deg2:+7.1f}"

            # Draw text for displaying angles
            cv2.putText(display, angle1_text, (15, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)
            cv2.putText(display, angle2_text, (15, 45),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)

        # Display chaos status
        if first_chaos_frame != -1 and i >= first_chaos_frame:
            is_chaotic = True

        if is_chaotic:
            cv2.putText(display, "CHAOS DETECTED", (15, 65),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        else:
            cv2.putText(display, "Stable", (15, 65),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        # Show the frames with markers and angle/chaos data
        cv2.imshow("Double Pendulum Analysis", display)

        # Write frame to output video
        if out is not None:
            out.write(display)

        # Wait for user to quit window manually with 'q' or pause with 'p'
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    # Close windows and release video writer
    cv2.destroyAllWindows()
    if out is not None:
        out.release()