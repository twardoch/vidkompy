#!/usr/bin/env python3
"""Debug script to extract and save frames for visual inspection."""

import cv2


def extract_debug_frames():
    """Extract first frame from each video for inspection."""

    # Extract first frame from FG video
    fg_cap = cv2.VideoCapture("tests/fg1.mp4")
    ret, fg_frame = fg_cap.read()
    if ret:
        cv2.imwrite("debug_fg_frame.jpg", fg_frame)
    fg_cap.release()

    # Extract first frame from BG video
    bg_cap = cv2.VideoCapture("tests/bg1.mp4")
    ret, bg_frame = bg_cap.read()
    if ret:
        cv2.imwrite("debug_bg_frame.jpg", bg_frame)
    bg_cap.release()


if __name__ == "__main__":
    extract_debug_frames()
