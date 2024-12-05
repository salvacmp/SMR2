import math
import cv2
from ultralytics import YOLO

# Helper function to calculate OBB points
def calculate_obb_points(x_center, y_center, width, height, angle):
    """
    Calculate the four corner points of an oriented bounding box (OBB).
    :param x_center: X-coordinate of the OBB center.
    :param y_center: Y-coordinate of the OBB center.
    :param width: Width of the OBB.
    :param height: Height of the OBB.
    :param angle: Orientation angle of the OBB in degrees.
    :return: List of 4 corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """
    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Half dimensions
    half_width = width / 2
    half_height = height / 2

    # Rotation matrix components
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Calculate the four corners relative to the center
    corners = [
        (
            x_center - half_width * cos_a + half_height * sin_a,
            y_center - half_width * sin_a - half_height * cos_a
        ),  # Top-left
        (
            x_center + half_width * cos_a + half_height * sin_a,
            y_center + half_width * sin_a - half_height * cos_a
        ),  # Top-right
        (
            x_center + half_width * cos_a - half_height * sin_a,
            y_center + half_width * sin_a + half_height * cos_a
        ),  # Bottom-right
        (
            x_center - half_width * cos_a - half_height * sin_a,
            y_center - half_width * sin_a + half_height * cos_a
        )   # Bottom-left
    ]

    return corners
def run_webcam_inference_with_obb(model_path, confidence_threshold=0.79):
    # Load the trained YOLOv8 OBB model
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(2)  # Change camera index if needed
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to quit the webcam.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Perform inference
        results = model(frame)

        # Get the annotated frame
        

        # Parse OBB data
        obbs = results[0].obb  # Access OBB information
        if obbs is not None:
            for obb in obbs.data.cpu().numpy():  # Move to CPU and convert to NumPy
                # Extract OBB details
                x_center, y_center, width, height, angle, confidence, class_id = obb

                if confidence >= 0.79:
                    # Calculate OBB points
                    points = calculate_obb_points(x_center, y_center, width, height, angle)
                    annotated_frame = results[0].plot()

                    # Draw the four points and lines connecting them
                    for i in range(4):
                        # Draw each corner
                        cv2.circle(annotated_frame, (int(points[i][0]), int(points[i][1])), 5, (0, 255, 255), -1)


                        # Draw lines between consecutive points (and close the box)
                        

        # Display the frame with OBB
        cv2.imshow("YOLOv8 OBB Webcam Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    trained_model_path = "best.pt"  # Replace with your model's path
    run_webcam_inference_with_obb(trained_model_path, confidence_threshold=0.5)
