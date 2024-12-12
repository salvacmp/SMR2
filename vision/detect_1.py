import cv2
from ultralytics import YOLO
import time
import numpy as np
import moveRobot


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

perspective_matrix = cv2.getPerspectiveTransform(eye_points, hand_points)

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

def angleOffset(Angle):
    if Angle < 0 :
        Angle * -1
    if Angle >= 85 and Angle <= 180 :
        return Angle #OK
    if Angle <85 and Angle >=60:
        return Angle + 90 #OK
    if Angle < 60 and Angle >= 0:
        return 180-Angle 

def run_webcam_inference_with_centers(confidence_threshold=0.55):

    """
    Run YOLOv8 model on a webcam feed with center point marking for each detection.
    :param model_path: Path to the trained YOLOv8 model (e.g., 'yolov8n.pt' or your trained model file).
    :param confidence_threshold: Minimum confidence for detections to be displayed.
    """
    model_path = "best.pt"
    # Load the trained YOLOv8 model
    model = YOLO(model_path)
    
    # Open webcam (default camera index is 2 for external cameras)
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    print("Press 'q' to quit the webcam.")
    
    for x in range(5):
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Perform inference
        results = model(frame)
        annotated_frame = frame.copy()
        cv2.circle(annotated_frame, (164,423), 5, (0, 255, 0), -1)
        # qcd = cv2.QRCodeDetector()
        # ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(annotated_frame)
        # print(decoded_info)
        # if ret_qr:
        #     for s, p in zip(decoded_info, points):
        #         if s:
        #             print(s)
        #             color = (0, 255, 0)
        #         else:
        #             color = (0, 0, 255)
        #         cv2.polylines(annotated_frame, [p.astype(int)], True, color, 8)
        points = []

        # Parse results and add center points for OBBs
        obbs = results[0].obb  # Access the OBB object
        if obbs is not None:
            for obb in obbs.data.cpu().numpy():
                # Extract OBB details
                x_center, y_center, width, height, angle, confidence, class_id = obb

                if confidence >= confidence_threshold:
                    # Simulate coordinates with the origin at the bottom-left
                    hand_x, hand_y = map_to_hand_plane((x_center, y_center), perspective_matrix)

                    # simulated_x = int(164)
                    # simulated_y = int(423)
                    # Draw the center point
                    cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
                    
                    # Display the simulated coordinates and class name
                    class_name = results[0].names[int(class_id)]
                    cv2.putText(
                        annotated_frame,
                        f"Hand: ({hand_x:.3f}, {hand_y:.3f})",
                        (int(x_center) + 10, int(y_center) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
                    print(repr(hand_x))
                    if hand_x is not None and x==4 :
                        print("Final")
                        angle_converted = ((np.degrees(angle) % 360))
                        angle_counted = angleOffset(angle_converted)
                        print(f"Real angle: {angle_converted} | Offset Angle: {angle_counted}")
                        print([hand_x,hand_y, int( angle_counted)])
                        points.append([hand_x,hand_y, int(angle_counted)])
                    
        # if(x == 2):
        #     print(repr(points))
        #     print(len(points))
        # for m in range(len(points)):
        #     moveRobot.sendMove(points[m][0],points[m][1])
        #     time.sleep(5)
                                    

        # Display the annotated frame
        cv2.imshow("Webcam Interface", annotated_frame)
        # time.sleep(1)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return points

# if __name__ == "__main__":
#     # Path to the trained YOLO model
#     trained_model_path = "best.pt"  # Replace with your model's path

#     # Run inference on webcam
#     run_webcam_inference_with_centers(trained_model_path, confidence_threshold=0.5)
