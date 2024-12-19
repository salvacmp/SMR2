import cv2
import pyrealsense2 as rs
import numpy as np

# Checkerboard parameters
checkerboard_size = (6, 9)  # Internal corners (rows, columns)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    print("Press 'q' to exit the visualization.")
    while True:
        # Capture frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.resize(color_image, (1280,960), 
               interpolation = cv2.INTER_LINEAR)
        # Convert to grayscale for checkerboard detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, checkerboard_size, None)

        if ret:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

            # Draw corners and labels
            for idx, corner in enumerate(corners):
                x, y = int(corner[0, 0]), int(corner[0, 1])
                # Draw a circle on the corner
                cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)  # Green circle
                # Add a label next to the corner
                cv2.putText(color_image, str(idx + 1), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw checkerboard overlay
            cv2.drawChessboardCorners(color_image, checkerboard_size, corners, ret)

        # Display the image
        cv2.imshow("Checkerboard Visualization", color_image)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
