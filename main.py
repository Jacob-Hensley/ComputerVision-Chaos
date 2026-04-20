# main.py
# Date Created: 4-15-2026
# Last Modified: 4-17-2026
# Description: Main entry point for the application. This file is used to load and preprocess
# the .mp4 video file, extract the frames, and then pass them to another file for analysis.

# Import necessary libraries
import cv2
import os

from numpy import test
# Import functions from files
from analyze import *
from display import *
from preprocess import *

# Main funtion
if __name__ == "__main__":
    # Load and preprocess video
    print("Welcome to the Double Pendulum Chaos Detection System! \n" \
    "This system analyzes videos of double pendulums to detect chaotic" \
    " behavior. \nPlease provide the path to a .mp4 video file of a " \
    "double pendulum in motion.\n")
    video_path = input("Enter the path to the .mp4 video file: ")

    try:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    output_path = input("Enter an output file to save the visualized video (e.g., output.mp4): ")

    frames = preprocess_video(video_path)
    print(f"Loaded {len(frames)} frames")
    
    # Perform pivot detection
    pivot, L1 = detect_pivot(frames)
    # Perform bob (or pendulum mass) detection 
    bob_positions = detect_bobs(frames, pivot=pivot, L1=L1)
    # Estimate angles
    angles = estimate_angles(pivot, bob_positions)

    # Analyze chaos
    has_chaos, first_chaos_frame = analyze_chaos(angles)

    # Visualize and playback detected features, angles, and chaos
    visualize_detections(frames, pivot, bob_positions, angles, 
                        first_chaos_frame=first_chaos_frame, delay=25, output_path=output_path)