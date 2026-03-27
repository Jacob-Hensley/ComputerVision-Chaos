## load.py
## Date Created: 03-27-2026
## Date Last Modified: 03-27-2026

## Description: This file will hold a class that will be responsible for
#  loading the .mp4 file and returning it in a format that can be processed
#  by the rest of the application.

import cv2


class LoadVideo:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        # Code to load the video file and return it in a format that can be processed
        vid = cv2.VideoCapture(self.file_path)
        vid = self.get_frames(vid)
        return vid
    
    # Convert video to frames
    def get_frames(self, cap):
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video or read failed
                break
            frames.append(frame)
        return frames

    
    