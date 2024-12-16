import cv2
import pyrealsense2 as rs
import numpy as np

# Set up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Function to get mouse click coordinates
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: ({x}, {y})")

# Create a window and set the mouse callback
cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("RealSense", mouse_callback)
camera_matrix = np.array([[1.07547194e+03, 0.00000000e+00, 3.04854337e+02],
                          [0.00000000e+00, 1.07168396e+03, 2.15872880e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs = np.array([[-3.00103243e-01, 1.53284927e+01, 1.00493178e-02, -8.89411997e-03, -9.81515375e-01]])
while True:
    # Wait for a frame from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    h, w = color_image.shape[:2]
    new_camera_matrix, undist_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(color_image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = undist_roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    frame = undistorted_image
    # Display the color image in the window
    cv2.imshow("RealSense", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the pipeline and close the window
pipeline.stop()
cv2.destroyAllWindows()
