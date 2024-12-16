import cv2
import pyrealsense2 as rs
import numpy as np

# Load the camera matrix and distortion coefficients from the npz file
def load_calibration_data(filename="E:/1_SMR2/SMR2/UR Script/average_calibration.npz"):
    try:
        data = np.load(filename)
        camera_matrix = data['camera_matrix']
        distortion_coeffs = data['distortion_coeffs']
        print("Calibration data loaded successfully.")
        return camera_matrix, distortion_coeffs
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None, None

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable both color and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # Depth stream

# Start the pipeline
pipeline.start(config)

# Load calibration data
camera_matrix, distortion_coeffs = load_calibration_data()

if camera_matrix is None or distortion_coeffs is None:
    print("Exiting due to missing or invalid calibration data.")
    pipeline.stop()
    exit()

# Main loop to capture frames, undistort, and display them
while True:
    # Wait for a new frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # Convert to numpy array
    color_image = np.asanyarray(color_frame.get_data())
    h, w = color_image.shape[:2]
    new_camera_matrix, undist_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
    # Undistort the image using the calibration data
    undistorted_image = cv2.undistort(color_image, camera_matrix, distortion_coeffs, None, new_camera_matrix)
    x, y, w, h = undist_roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    # Display the original and undistorted images
    cv2.imshow('Original Color Image', color_image)
    cv2.imshow('Undistorted Image', undistorted_image)

    # Break the loop on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up and stop the pipeline
cv2.destroyAllWindows()
pipeline.stop()
