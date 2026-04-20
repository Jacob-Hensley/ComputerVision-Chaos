# preprocess.py
# Date Created: 4-17-2026
# Last Modified: 4-17-2026
# Description: This file contains preprocessing functions for video loading and frame preparation,
# including resizing, converting to grayscale, and applying blur filters for analysis.

# Import necessary libraries
import cv2

# Function for loading and preprocessing video
def preprocess_video(video_path):
    # Capture video
    cap = cv2.VideoCapture(video_path)
    # Initialize empty list of frames
    frames = []
    # Read and preprocess frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Reduce frame size to 500x500 pixels for faster processing
        frame = cv2.resize(frame, (500, 500))
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise and improve detection stability
        smoothed_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        # Append the preprocessed frame to the list
        frames.append(smoothed_frame)
    cap.release()
    return frames