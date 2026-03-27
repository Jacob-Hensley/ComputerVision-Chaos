## preprocess.py
## Date Created: 03-27-2026
## Date Last Modified: 03-27-2026

## Description: This file will hold a class that will be responsible for
#  preprocessing and postprocessing of the loaded video file and returning it
#  in a format that can be worked with by the rest of the application.

import cv2

class ProcessVideo:
    def __init__(self):
        pass
    
    def reduce_resolution(self, cap, scale_percent=50):
        reduced_frames = []
        for frame in cap:
            if frame is None:
                continue
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            reduced_frames.append(resized)
        return reduced_frames
        
    def convert_to_grayscale(self, video):
        gray_frames = []
        for frame in video:                    # Loop through each frame
            if frame is None:
                continue
            # frame should be a numpy array here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)

        return gray_frames

    def smooth_video(self, video):
        # Code to smooth the video file and return it in a format that can be processed
        smooth_frames = []
        for frame in video:
            if frame is None:
                continue
            smooth = cv2.GaussianBlur(frame, (5, 5), 0)
            smooth_frames.append(smooth)
        return smooth_frames