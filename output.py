## output.py
## Date Created: 03-27-2026
## Date Last Modified: 03-27-2026

## Description: This file will hold the output formatting and display logic
#  for the analyzed pendulum data.

class Output:
    def __init__(self, angles):
        self.angles = angles
    
    def display_output(self):
        # Code to display the output in a readable format
        for angle in self.angles:
            print(f"Angle: {angle}")
    
    def save_output(self, file_name):
        # Code to save the output to a text file
        with open(file_name, 'w') as f:
            for angle in self.angles:
                f.write(f"Angle: {angle}\n")