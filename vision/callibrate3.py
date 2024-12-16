import pyrealsense2 as rs
import numpy as np
import cv2

# Checkerboard dimensions
CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Allow the camera to adjust to lighting conditions
for _ in range(30):
    pipeline.wait_for_frames()

# Variables to store calibration results for averaging
camera_matrices = []
distortion_coeffs = []
rvecs_list = []
tvecs_list = []

capture_count = 4
captures = 0

try:
    while captures < capture_count:
        # Get frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)

            cv2.drawChessboardCorners(color_image, CHECKERBOARD, corners2, ret)

            # Measure height from camera to table
            height, width = depth_image.shape
            center_x, center_y = width // 2, height // 2
            height_from_camera = depth_frame.get_distance(center_x, center_y)

            print(f"Detected Height from camera to table: {height_from_camera:.2f} meters")

            # Perform camera calibration
            ret, camera_matrix, distortion_coeffs_, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            if ret:
                camera_matrices.append(camera_matrix)
                distortion_coeffs.append(distortion_coeffs_)
                rvecs_list.append(rvecs)
                tvecs_list.append(tvecs)

                captures += 1
                print(f"Capture {captures} done. Camera Matrix:\n{camera_matrix}")
                print(f"Distortion Coefficients:\n{distortion_coeffs_}")
            else:
                print(f"Calibration failed for capture {captures + 1}.")
        else:
            print("Checkerboard pattern not found. Please ensure the pattern is in view.")

        cv2.imshow('Color Frame', color_image)
        cv2.imshow('Depth Frame', depth_image)
        cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()

    # Perform the averaging of calibration results
    if camera_matrices:
        avg_camera_matrix = np.mean(camera_matrices, axis=0)
        avg_distortion_coeffs = np.mean(distortion_coeffs, axis=0)

        # Optionally, average rvecs and tvecs
        avg_rvecs = np.mean(rvecs_list, axis=0)
        avg_tvecs = np.mean(tvecs_list, axis=0)

        print("Average Camera Matrix (Intrinsic Parameters):")
        print(avg_camera_matrix)
        print("\nAverage Distortion Coefficients:")
        print(avg_distortion_coeffs)

        print("\nAverage Rotation Vectors:")
        print(avg_rvecs)

        print("\nAverage Translation Vectors:")
        print(avg_tvecs)

        # Optionally save the results to a file
        np.savez("averaged_calibration_4captures.npz", 
                 camera_matrix=avg_camera_matrix, 
                 distortion_coeffs=avg_distortion_coeffs,
                 rvecs=avg_rvecs, 
                 tvecs=avg_tvecs)

        print("\nCalibration data averaged and saved.")
