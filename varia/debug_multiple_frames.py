#!/usr/bin/env python3
"""Debug script to test template matching on multiple frames."""

import cv2
import numpy as np


def test_multiple_frames():
    """Test template matching on multiple frames from the videos."""

    # Open videos
    fg_cap = cv2.VideoCapture("tests/fg1.mp4")
    bg_cap = cv2.VideoCapture("tests/bg1.mp4")

    fg_total = int(fg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_total = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Test frames at different positions in the videos
    test_positions = [0, fg_total // 4, fg_total // 2, 3 * fg_total // 4, fg_total - 1]

    for _i, frame_pos in enumerate(test_positions):
        if frame_pos >= min(fg_total, bg_total):
            continue

        # Extract frames
        fg_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        ret_fg, fg_frame = fg_cap.read()
        ret_bg, bg_frame = bg_cap.read()

        if not (ret_fg and ret_bg):
            continue

        # Convert to grayscale
        fg_gray = cv2.cvtColor(fg_frame, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(bg_gray, fg_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Test specific positions
        positions_to_test = [(0, 109), (0, 221)]
        correlations = {}

        for x, y in positions_to_test:
            if y + fg_gray.shape[0] <= bg_gray.shape[0]:
                bg_patch = bg_gray[y : y + fg_gray.shape[0], x : x + fg_gray.shape[1]]
                bg_norm = cv2.normalize(
                    bg_patch.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX
                )
                fg_norm = cv2.normalize(
                    fg_gray.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX
                )
                correlation = np.corrcoef(bg_norm.flatten(), fg_norm.flatten())[0, 1]
                correlations[(x, y)] = correlation

    fg_cap.release()
    bg_cap.release()


if __name__ == "__main__":
    test_multiple_frames()
