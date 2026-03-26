# ComputerVision-Chaos
This project explores the capabilities of traditional computer vision in analyzing the chaotic movement of a physical system: the double pendulum. This project will utilize available functions within the OpenCV library to extract the angles (in real time) from a video simulation of the system.

# System Architecture
The system will include the following files:
    1. main.py          --      The main file for running the application
    2. load_mp4.py      --      Loads the .mp4 of the double-pendulum
    3. preprocess.py    --      Preprocesses the loaded video to be read by OpenCV
    4. analyze.py       --      Utilizes OpenCV to perform angle calculations

# main.py
This file will call the other dependency files as necessary to load, preprocess, and analyze the file.

# load.py
This file will pull the video from the project's root directory and return it

# preprocess.py
This file will take the loaded video and perform any preprocessing (e.g. smoothing and/or other methods) before analysis

# analyze.py
This file will utilize OpenCV to perform object tracking and angle calculations

# plot_data.py (unofficial)
If we have the flexibility to do so, we can plot the angle1 by angle2 plot of the pendulum over time.