#!/usr/bin/env python3
"""Debug script to test template matching manually."""

import cv2
import numpy as np


def debug_template_matching():
    """Test template matching at different positions."""

    # Load frames
    fg_frame = cv2.imread("debug_fg_frame.jpg", cv2.IMREAD_GRAYSCALE)
    bg_frame = cv2.imread("debug_bg_frame.jpg", cv2.IMREAD_GRAYSCALE)

    # Test template matching
    result = cv2.matchTemplate(bg_frame, fg_frame, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Test specific positions manually
    positions_to_test = [(0, 109), (0, 221), (0, 198)]

    for x, y in positions_to_test:
        # Extract patch from BG at this position
        if (
            y + fg_frame.shape[0] <= bg_frame.shape[0]
            and x + fg_frame.shape[1] <= bg_frame.shape[1]
        ):
            bg_patch = bg_frame[y : y + fg_frame.shape[0], x : x + fg_frame.shape[1]]

            # Calculate correlation by comparing pixels directly
            if bg_patch.shape == fg_frame.shape:
                # Normalize both patches
                bg_norm = cv2.normalize(
                    bg_patch.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX
                )
                fg_norm = cv2.normalize(
                    fg_frame.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX
                )

                # Calculate correlation coefficient
                np.corrcoef(bg_norm.flatten(), fg_norm.flatten())[0, 1]
            else:
                pass
        else:
            pass


if __name__ == "__main__":
    debug_template_matching()
