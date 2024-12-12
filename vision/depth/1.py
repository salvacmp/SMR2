import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import math
#RobotHeight [0.25009, 0.19872] 
#cameraHeight [1.086, 1.132]
eye_points = np.float32([
    (300, 423),  # Bottom-left (BL)
    (423, 423),  # Bottom-right (BR)
    (423, 200),  # Top-right (TR)
    (300, 200),  # Top-left (TL)
])

hand_points = np.float32([
    (-0.464567, -0.37354),  # Bottom-left (BL)
    (-0.458792, -0.605187),  # Bottom-right (BR)
    (-0.029369, -0.611195),  # Top-right (TR)
    (-0.030541, -0.372615),  # Top-left (TL)
])

roi_polygon = [(228, 186), (238, 468), (480, 468), (471, 186)]
perspective_matrix = cv2.getPerspectiveTransform(eye_points, hand_points)

def map_to_hand_plane(point, matrix):
    src_point = np.array([[point]], dtype=np.float32)
    dst_point = cv2.perspectiveTransform(src_point, matrix)
    return dst_point[0, 0]

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def convert_camera_to_robot(camera_z):
    """
    Convert a Z-axis position from the camera's coordinate system to the robot's coordinate system.

    Args:
        camera_z (float): Z position in the camera's coordinate system.
        robot_height (float): Z position of the robot's base in its own coordinate system.
        camera_height (float): Z position of the camera's base in the robot's coordinate system.

    Returns:
        float: Z position in the robot's coordinate system.
    """
    # Calculate the transformation from camera to robot
    offset = 1.086 - 0.25009
    robot_position_z = camera_z - offset

    return robot_position_z

def run_webcam_inference_with_depth(confidence_threshold=0.55):
    model_path = "E:/1_SMR2/SMR2/UR Script/vision/best.pt"
    model = YOLO(model_path)
    
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    print("Press 'q' to quit the program.")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model(frame)
        annotated_frame = frame.copy()
        depth_annotated_frame = depth_image.copy()

        cv2.polylines(annotated_frame, [np.array(roi_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        obbs = results[0].obb
        if obbs is not None:
            for obb in obbs.data.cpu().numpy():
                x_center, y_center, width, height, angle, confidence, class_id = obb
                if confidence >= confidence_threshold and is_point_in_polygon((x_center, y_center), roi_polygon):
                    hand_x, hand_y = map_to_hand_plane((x_center, y_center), perspective_matrix)

                    # Draw center point
                    cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)

                    # Get depth value at center point
                    depth_value = depth_frame.get_distance(int(x_center), int(y_center)) / depth_scale
                    cv2.putText(
                        annotated_frame,
                        f"Depth: {depth_value:.3f}mm",
                        (int(x_center) + 10, int(y_center) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )

                    # Display depth value on depth frame
                    cv2.putText(
                        depth_image,
                        f"{depth_value:.3f}m",
                        (int(x_center), int(y_center)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        255,
                        1
                    )

        # Convert depth image to a viewable format
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display both frames
        cv2.imshow("Webcam Interface", annotated_frame)
        cv2.imshow("Depth View", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_inference_with_depth(confidence_threshold=0.55)
