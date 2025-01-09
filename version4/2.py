import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Define the Region of Interest (ROI) for the brick
        roi_x, roi_y, roi_w, roi_h = 200, 200, 100, 100  # Example ROI
        depth_image = np.asanyarray(depth_frame.get_data())

        # Extract ROI and calculate average depth
        roi = depth_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        avg_depth = np.mean(roi)

        if avg_depth == 0:
            print("No brick detected.")
            continue

        print(f"Distance to brick: {avg_depth} mm")

        # Update perspective transform or calibration based on avg_depth
        # For example, recompute the perspective transform:
        # pts_src and pts_dst should be updated dynamically
        # matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

        # Visualize the depth image and ROI
        cv2.rectangle(depth_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 255, 2)
        cv2.imshow("Depth Image", depth_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
