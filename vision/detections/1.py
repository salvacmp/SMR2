import cv2
from ultralytics import YOLO
import time
import numpy as np
import math
import pyrealsense2 as rs

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

def get_obb_corners(x_center, y_center, width, height, angle):
    """
    Calculate the four corners of an Oriented Bounding Box (OBB).
    :param x_center: X-coordinate of the OBB center.
    :param y_center: Y-coordinate of the OBB center.
    :param width: Width of the OBB.
    :param height: Height of the OBB.
    :param angle: Rotation angle of the OBB in radians.
    :return: List of (x, y) tuples for the four corners.
    """
    # Calculate half dimensions
    half_width = width / 2
    half_height = height / 2
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Define the corner points relative to the center
    corners = np.array([
        [-half_width, -half_height],  # Bottom-left
        [ half_width, -half_height],  # Bottom-right
        [ half_width,  half_height],  # Top-right
        [-half_width,  half_height],  # Top-left
    ])
    
    # Rotate and translate the corners to the global coordinate system
    rotated_corners = np.dot(corners, rotation_matrix.T) + [x_center, y_center]
    
    return rotated_corners

def map_to_hand_plane(point, matrix):
    """
    Maps a point from the eye plane to the hand plane using the perspective matrix.
    :param point: (x, y) tuple in the eye plane.
    :param matrix: Perspective transformation matrix.
    :return: (x, y) tuple in the hand plane.
    """
    src_point = np.array([[point]], dtype=np.float32)  # Shape: (1, 1, 2)
    dst_point = cv2.perspectiveTransform(src_point, matrix)  # Shape: (1, 1, 2)
    return dst_point[0, 0]  # Extract (x, y)
def calculate_brick_rotation(width, height, angle):
    """
    Calculate the brick's rotation relative to the bottom line.
    :param width: Width of the OBB.
    :param height: Height of the OBB.
    :param angle: Orientation angle of the OBB in degrees.
    :return: Adjusted angle (0° for short side down, 90° for long side down).
    """
    # Normalize the angle to the range [0, 180]
    normalized_angle = angle % 180

    if width < height:  # Short side pointing down
        rotation = normalized_angle
    else:  # Long side pointing down
        rotation = (normalized_angle + 90) % 180

    return rotation

def angleOffset(Angle):
    if Angle < 0 :
        Angle * -1
    if Angle >= 85 and Angle <= 180 :
        return Angle #OK
    if Angle <85 and Angle >=60:
        return Angle + 90 #OK
    if Angle < 60 and Angle >= 0:
        return 180-Angle 
def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using cv2.pointPolygonTest.
    :param point: Tuple (x, y) representing the point.
    :param polygon: List of points [(x1, y1), (x2, y2), ...] representing the polygon.
    :return: True if the point is inside the polygon, False otherwise.
    """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def run_webcam_inference_with_centers(confidence_threshold=0.55):

    """
    Run YOLOv8 model on a webcam feed with center point marking for each detection.
    :param model_path: Path to the trained YOLOv8 model (e.g., 'yolov8n.pt' or your trained model file).
    :param confidence_threshold: Minimum confidence for detections to be displayed.
    """
    model_path = "E:/1_SMR2/SMR2/UR Script/vision/best.pt"
    # Load the trained YOLOv8 model
    model = YOLO(model_path)
    
    # Open webcam (default camera index is 2 for external cameras)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    print("Press 'q' to quit the webcam.")
    def radians_to_degrees(radians):
        # Convert radians to degrees
        degrees = math.degrees(radians)
        # Normalize to 0-360 degrees
        normalized_degrees = (degrees % 360 + 360) % 360
        return normalized_degrees
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        print(int(frame_width))
        # Perform inference
        results = model(frame)
        annotated_frame = frame.copy()
        # annotated_frame = results[0].plot()
        cv2.circle(annotated_frame, (164,423), 5, (0, 255, 0), -1)
        points = []
        
        # annotated_frame = results[0].plot()
        cv2.polylines(annotated_frame, [np.array(roi_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Parse results and add center points for OBBs
        obbs = results[0].obb  # Access the OBB object
        if obbs is not None:
            for obb in obbs.data.cpu().numpy():
                # Extract OBB details
                x_center, y_center, width, height, angle, confidence, class_id = obb
                #detection square
                if confidence >= confidence_threshold:
                    if is_point_in_polygon((x_center, y_center), roi_polygon):
                        corners = get_obb_corners(x_center, y_center, width, height, angle)
                        cornerList = []

                        # Collect y-coordinates along with their index
                        for i, corner in enumerate(corners):
                            cornerList.append((corner[1], i))

                        # Sort the corners based on their y-coordinates (ascending)
                        sorted_corners = sorted(cornerList, key=lambda x: x[0])

                        # Get the three bottom-most corners
                        bottom_three = sorted_corners[-3:]

                        # Extract the most bottom point
                        bottom_point_index = bottom_three[-1][1]
                        bottom_point = corners[bottom_point_index]

                        # Draw the bottom-most point
                        cv2.circle(annotated_frame, (int(bottom_point[0]), int(bottom_point[1])), 5, (255, 0, 0), -1)

                        # Calculate distances and store the points
                        distances = []
                        for y_coord, index in bottom_three[:-1]:  # Skip the bottom-most point
                            current_point = corners[index]

                            # Calculate the distance
                            distance = np.sqrt((current_point[0] - bottom_point[0]) ** 2 + (current_point[1] - bottom_point[1]) ** 2)
                            distances.append((distance, current_point))

                        # Sort distances to get the longest side
                        distances.sort(reverse=True, key=lambda x: x[0])
                        longest_side_distance, longest_side_point = distances[0]

                        # Calculate the angle for the longest side
                        dx = longest_side_point[0] - bottom_point[0]
                        dy = longest_side_point[1] - bottom_point[1]
                        angle = math.degrees(math.atan2(dy, dx))  # Angle in degrees
                        if angle < 0: angle *=-1
                        # Display the longest side and its angle
                        cv2.line(
                            annotated_frame,
                            (int(bottom_point[0]), int(bottom_point[1])),
                            (int(longest_side_point[0]), int(longest_side_point[1])),
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            annotated_frame,
                            f"Angle: {angle:.2f}°",
                            (int((longest_side_point[0] + bottom_point[0]) / 2), int((longest_side_point[1] + bottom_point[1]) / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1
                        )

                        # Draw a horizontal line at the bottom-most point’s y-coordinate
                        cv2.line(
                            annotated_frame,
                            (0, int(bottom_point[1])),
                            (frame_width, int(bottom_point[1])),
                            (0, 0, 255),
                            2
                        )

                        
                    # Simulate coordinates with the origin at the bottom-left
                        hand_x, hand_y = map_to_hand_plane((x_center, y_center), perspective_matrix)
                        brick_rotation = calculate_brick_rotation(width, height, angle)

                        print(repr(brick_rotation))
                        cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
                        
                        cv2.line(annotated_frame, (int(x_center), int(y_center)), (int(x_center), 10), (0, 255, 0), 2)
                        # cv2.putText(
                        #     annotated_frame,
                        #     f"Rotation: {int(np.degrees(angle) % 360)}° | {angleOffset(int(np.degrees(angle)% 360))}° | NC: {radians_to_degrees(angle) }",
                        #     (int(x_center) + 10, int(y_center) - 30),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5,
                        #     (255, 0, 0),
                        #     1
                        # )
                        # Display the simulated coordinates and class name
                        class_name = results[0].names[int(class_id)]
                        # cv2.putText(
                        #     annotated_frame,
                        #     f"Hand: ({hand_x:.3f}, {hand_y:.3f})",
                        #     (int(x_center) + 10, int(y_center) - 10),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5,
                        #     (0, 255, 0),
                        #     1
                        # )
                        print(repr(hand_x))
        # Display the annotated frame
        cv2.imshow("Webcam Interface", annotated_frame)
        # time.sleep(1)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # Path to the trained YOLO model
    trained_model_path = "best.pt"  # Replace with your model's path

    # Run inference on webcam
    run_webcam_inference_with_centers( confidence_threshold=0.5)
