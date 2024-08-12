import cv2
import numpy as np
from datetime import datetime

def create_black_screen_with_details(message, frame, fps):
    black_screen = np.zeros_like(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"{message}\nTime: {current_time}\nFPS: {fps:.2f}"
    y0, dy = 40, 40
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(black_screen, line, (50, y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return black_screen

def log_details(identity, emotion, log_file):
    with open(log_file, 'a') as f:
        if emotion is not None:
            f.write(f"Name: {identity}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Emotion: {emotion}\n")

