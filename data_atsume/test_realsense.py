import time
from typing import Tuple

import pyrealsense2 as rs
import numpy as np
import cv2


def record_realsense(duration_seconds: float = 1.0, fps: int = 30, size: Tuple[int, int] = (640, 480)) -> None:
    """Record color video (MP4) and save a depth image from a RealSense camera.

    Parameters
    ----------
    duration_seconds: float
        How many seconds to record (default 1.0).
    fps: int
        Target frames per second for recording (default 30).
    size: Tuple[int,int]
        Width and height for the color stream.
    """

    width, height = size

    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Start streaming
    pipeline.start(config)

    # Prepare video writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("capture.mp4", fourcc, float(fps), (width, height))

    start_time = time.time()
    end_time = start_time + duration_seconds
    last_depth_frame = None
    frames_written = 0

    try:
        while time.time() < end_time:
            # Use a shorter timeout and ignore occasional timeouts
            try:
                frames = pipeline.wait_for_frames(1000)  # ms
            except RuntimeError:
                # "Frame didn't arrive" – skip and continue until time window ends
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                # Need at least color to write video
                continue

            # Color frame -> BGR image
            color_image = np.asanyarray(color_frame.get_data())

            # Write color frame to video
            out.write(color_image)
            frames_written += 1

            # Keep last depth frame for saving after loop (if available)
            if depth_frame:
                last_depth_frame = np.asanyarray(depth_frame.get_data())

        # Save last depth frame if we got one
        if last_depth_frame is not None:
            # Raw 16-bit PNG (depth in millimeters)
            cv2.imwrite("depth_raw.png", last_depth_frame)

            # Create a colorized visualization (8-bit)
            depth_vis = cv2.convertScaleAbs(last_depth_frame, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite("depth_colormap.png", depth_colormap)

        print(f"Finished recording: wrote {frames_written} frames to capture.mp4")

    finally:
        # Release resources
        out.release()
        pipeline.stop()


if __name__ == "__main__":
    # Record 1 second at 30 fps by default
    record_realsense(duration_seconds=3.0, fps=30, size=(640, 480))
