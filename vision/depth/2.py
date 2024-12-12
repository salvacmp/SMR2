import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import math

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

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def calculate_highest_point(depth_frame, roi_polygon):
    """
    Calculate the highest average depth point (minimum depth) inside the ROI.
    :param depth_frame: Depth frame from RealSense.
    :param roi_polygon: The region of interest polygon.
    :return: Average depth of the highest points within a small range.
    """
    heights = []
    for y in range(depth_frame.height):
        for x in range(depth_frame.width):
            if is_point_in_polygon((x, y), roi_polygon):
                depth = depth_frame.get_distance(x, y)
                if depth > 0:  # Valid depth
                    heights.append(depth)

    if not heights:
        return None
    # print(repr(heights))
    # Find the minimum depth (highest point) and average nearby values within 1 cm
    min_depth = min([0.7])
    # close_points = [d for d in heights if abs(d - min_depth) <= 0.01]
    # average_depth = sum(close_points) / len(close_points)

    return min_depth

def blackout_if_below_depth(frame, depth_frame, roi_polygon, highest_average_depth, tolerance=0.02):
    """
    Blackout the ROI on the frame if depth values fall below the threshold.
    :param frame: RGB frame.
    :param depth_frame: Depth frame from RealSense.
    :param roi_polygon: The region of interest polygon.
    :param highest_average_depth: The highest average depth in the ROI.
    :param tolerance: The tolerance below which to blackout the ROI (in meters).
    """
    mask = np.zeros_like(frame, dtype=np.uint8)
    for y in range(depth_frame.height):
        for x in range(depth_frame.width):
            if is_point_in_polygon((x, y), roi_polygon):
                depth = depth_frame.get_distance(x, y)
                if depth > (highest_average_depth + tolerance):
                    # Blackout region
                    cv2.fillPoly(mask, [np.array(roi_polygon, np.int32)], (0, 0, 0))

    # Apply the mask to blackout the ROI
    frame = cv2.add(frame, mask)
    return frame

def run_webcam_inference_with_depth(confidence_threshold=0.55):
    model_path = "best.pt"
    model = YOLO(model_path)
    
    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

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
        results = model(frame)
        annotated_frame = frame.copy()

        cv2.polylines(annotated_frame, [np.array(roi_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate the highest average depth in the ROI
        highest_average_depth = calculate_highest_point(depth_frame, roi_polygon)
        if highest_average_depth is not None:
            annotated_frame = blackout_if_below_depth(
                annotated_frame, depth_frame, roi_polygon, highest_average_depth, tolerance=0.02
            )

        # Parse results and add center points for OBBs
        obbs = results[0].obb  # Access the OBB object
        if obbs is not None:
            for obb in obbs.data.cpu().numpy():
                x_center, y_center, width, height, angle, confidence, class_id = obb
                if confidence >= confidence_threshold and is_point_in_polygon((x_center, y_center), roi_polygon):
                    # Draw the center point
                    cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
                    cv2.putText(
                        annotated_frame,
                        f"Depth OK",
                        (int(x_center) + 10, int(y_center) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

        # Display the annotated frame
        cv2.imshow("Webcam Interface", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_inference_with_depth(confidence_threshold=0.55)
