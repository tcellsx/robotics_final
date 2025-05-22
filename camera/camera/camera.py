import cv2
import numpy as np
import math
import time
from PIL import Image, ImageDraw, ImageFont
from collections import deque


# Function: for supporting text display
def cv2AddText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # Check if OpenCV image type
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Create an object that can draw on the given image
    draw = ImageDraw.Draw(img)
    # Font format - using Arial for English
    fontStyle = ImageFont.truetype(
        "arial.ttf", textSize, encoding="utf-8")
    # Draw text
    draw.text(position, text, textColor, font=fontStyle)
    # Convert back to OpenCV format
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# Calculate focal length using known distance and object size
def calculate_focal_length(pixel_width, known_width, known_distance):
    return (pixel_width * known_distance) / known_width


# Calculate distance using focal length, actual width and pixel width
def calculate_distance(focal_length, known_width, pixel_width):
    # Prevent division by zero error
    if pixel_width == 0:
        return 0
    return (known_width * focal_length) / pixel_width


# Draw text on image
def draw_text(img, text, position, color=(0, 255, 0), size=30):
    return cv2AddText(img, text, position, color, size)


# Check if drum hit conditions are met - using modified range
def check_drum_hit_condition(left_distance, right_distance):
    # Left point to camera distance within 44-47 cm range
    left_condition = 40 <= left_distance <= 44
    # Right point to camera distance within 58-61 cm range
    right_condition = 51 <= right_distance <= 56

    # Return True only if both conditions are met
    return left_condition and right_condition


# Smooth distance measurements
class DistanceFilter:
    def __init__(self, window_size=10):
        self.left_distances = deque(maxlen=window_size)
        self.right_distances = deque(maxlen=window_size)

    def update(self, left_distance, right_distance):
        self.left_distances.append(left_distance)
        self.right_distances.append(right_distance)

    def get_filtered_distances(self):
        # Return 0 if queue is empty
        if not self.left_distances or not self.right_distances:
            return 0, 0

        # Calculate average as smoothed distance
        left_avg = sum(self.left_distances) / len(self.left_distances)
        right_avg = sum(self.right_distances) / len(self.right_distances)

        return left_avg, right_avg


# State stability management
class DrumHitStateManager:
    def __init__(self, stability_threshold=5):
        self.stability_threshold = stability_threshold
        self.consecutive_hit_frames = 0
        self.consecutive_no_hit_frames = 0
        self.can_hit_drum = False

    def update(self, condition_met):
        if condition_met:
            self.consecutive_hit_frames += 1
            self.consecutive_no_hit_frames = 0

            # Only change state if condition met for multiple consecutive frames
            if self.consecutive_hit_frames >= self.stability_threshold:
                self.can_hit_drum = True
        else:
            self.consecutive_hit_frames = 0
            self.consecutive_no_hit_frames += 1

            # Only change state if condition not met for multiple consecutive frames
            if self.consecutive_no_hit_frames >= self.stability_threshold:
                self.can_hit_drum = False

        return self.can_hit_drum

    def get_stability_percentage(self):
        if self.can_hit_drum:
            # Already in hittable state, return 100%
            return 100
        else:
            # Calculate percentage to reach stable state
            return min(100, (self.consecutive_hit_frames / self.stability_threshold) * 100)


# Main program
def main():
    # Initialize camera - using index 1 (external camera)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open external camera, trying internal camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open any camera")
            exit()

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Pixel to actual distance conversion ratio
    PIXEL_TO_CM = 0.1  # Example value: 10 pixels = 1 cm (needs calibration)

    # Create window
    cv2.namedWindow("Red Point Distance Measurement", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Red Point Distance Measurement", 800, 600)

    # For storing two red objects' positions
    left_point = None
    right_point = None

    print("Press 'q' to exit")
    print("Press 'c' to calibrate distance between points")
    print("Press 's' to calibrate camera to red point distance (very important!)")

    # Calibration mode
    calibration_mode = False
    calibration_distance = 10  # Default calibration distance (cm)

    # Object size calibration parameters
    object_size_calibration = False
    known_width = 5.0  # Known width of red object (cm), needs calibration
    focal_length = None  # Will be calculated through calibration

    # Initialize distance filter and state manager
    distance_filter = DistanceFilter(window_size=10)  # 10 frame sliding window
    state_manager = DrumHitStateManager(stability_threshold=8)  # Need 8 consecutive frames to change state

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Cannot get frame")
            break

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red HSV range
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create red mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations on mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Result frame
        result = frame.copy()

        # Reset positions
        red_objects = []

        # Process found contours
        if len(contours) >= 1:
            # Sort contours by area
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

            # Process at most two largest red objects
            for i in range(min(2, len(contours_sorted))):
                if cv2.contourArea(contours_sorted[i]) > 100:  # Area threshold
                    # Calculate contour center
                    M = cv2.moments(contours_sorted[i])
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Get bounding rectangle of contour
                        x, y, w, h = cv2.boundingRect(contours_sorted[i])

                        # Store center point, area and size
                        red_objects.append({
                            "center": (cX, cY),
                            "area": cv2.contourArea(contours_sorted[i]),
                            "rect": (x, y, w, h),
                            "width": w,
                            "height": h
                        })

                        # Draw rectangle and center point
                        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(result, (cX, cY), 5, (255, 0, 0), -1)

                        # Mark area in small text
                        cv2.putText(result, f"Area: {int(cv2.contourArea(contours_sorted[i]))}",
                                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # If two red objects detected, calculate distance
        if len(red_objects) == 2:
            # Sort by X coordinate to determine left and right points
            if red_objects[0]["center"][0] < red_objects[1]["center"][0]:
                left_point = red_objects[0]
                right_point = red_objects[1]
            else:
                left_point = red_objects[1]
                right_point = red_objects[0]

            # Label on screen with smaller text
            cv2.putText(result, "L", (left_point["center"][0] + 10, left_point["center"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(result, "R", (right_point["center"][0] + 10, right_point["center"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Draw connecting line
            cv2.line(result, left_point["center"], right_point["center"], (0, 0, 255), 1)

            # Calculate pixel distance
            pixel_distance = math.sqrt(
                (left_point["center"][0] - right_point["center"][0]) ** 2 +
                (left_point["center"][1] - right_point["center"][1]) ** 2
            )

            # If in calibration mode, update ratio
            if calibration_mode and pixel_distance > 0:
                PIXEL_TO_CM = calibration_distance / pixel_distance
                print(f"Calibration complete! 1 cm = {1 / PIXEL_TO_CM:.4f} pixels")
                calibration_mode = False

            # Calculate actual distance between objects (cm)
            real_distance = pixel_distance * PIXEL_TO_CM

            # Display distance between objects (smaller font)
            result = draw_text(result, f"Distance: {real_distance:.1f} cm", (10, 20), (0, 0, 255), 18)

            # Display distance at midpoint of connecting line
            mid_x = (left_point["center"][0] + right_point["center"][0]) // 2
            mid_y = (left_point["center"][1] + right_point["center"][1]) // 2
            result = draw_text(result, f"{real_distance:.1f}", (mid_x, mid_y), (0, 0, 255), 14)

            # Calculate camera to object distance (if focal length is calibrated)
            left_distance = 0
            right_distance = 0

            if focal_length is not None:
                # Calculate distance to left red point
                left_distance = calculate_distance(focal_length, known_width, left_point["width"])
                # Calculate distance to right red point
                right_distance = calculate_distance(focal_length, known_width, right_point["width"])

                # Update distance filter
                distance_filter.update(left_distance, right_distance)

                # Get smoothed distances
                left_filtered, right_filtered = distance_filter.get_filtered_distances()

                # Display camera to object distances (show smoothed distances)
                result = draw_text(result, f"L-Cam: {left_filtered:.1f} cm", (10, 40), (0, 255, 255), 16)
                result = draw_text(result, f"R-Cam: {right_filtered:.1f} cm", (10, 60), (0, 255, 255), 16)

                # Check if drum hit conditions are met - using smoothed distances
                condition_met = check_drum_hit_condition(left_filtered, right_filtered)

                # Update state manager
                can_hit_drum = state_manager.update(condition_met)
                stability_percentage = state_manager.get_stability_percentage()

                # Display if drum can be hit
                if can_hit_drum:
                    # Display at center top of screen
                    result = draw_text(result, "READY TO HIT!", (result.shape[1]//2 - 60, 20), (0, 255, 0), 20)
                    # Add color indicator bar at bottom
                    cv2.rectangle(result, (0, result.shape[0] - 15), (result.shape[1], result.shape[0]), (0, 255, 0), -1)
                else:
                    # Display at center top of screen
                    result = draw_text(result, "NOT READY", (result.shape[1]//2 - 45, 20), (0, 0, 255), 20)
                    # Add red indicator bar at bottom
                    cv2.rectangle(result, (0, result.shape[0] - 15), (result.shape[1], result.shape[0]), (0, 0, 255), -1)

                # Display stability indicator bar
                bar_width = int((result.shape[1] - 80) * stability_percentage / 100)
                cv2.rectangle(result, (40, result.shape[0] - 35), (40 + bar_width, result.shape[0] - 25), (0, 255, 255), -1)
                cv2.rectangle(result, (40, result.shape[0] - 35), (result.shape[1] - 40, result.shape[0] - 25), (255, 255, 255), 1)

                # Display drum hit condition status - using smoothed distances
                left_status = "✓" if 44 <= left_filtered <= 47 else "✗"
                right_status = "✓" if 58 <= right_filtered <= 61 else "✗"

                status_text = f"L[44-47cm]: {left_status} ({left_filtered:.1f}cm)"
                result = draw_text(result, status_text, (10, result.shape[0] - 60), (255, 255, 255), 14)

                status_text = f"R[58-61cm]: {right_status} ({right_filtered:.1f}cm)"
                result = draw_text(result, status_text, (10, result.shape[0] - 80), (255, 255, 255), 14)
            else:
                # If focal length not calibrated, prompt user
                result = draw_text(result, "Press 's' to calibrate camera distance", (10, 40), (0, 255, 255), 16)
        else:
            # If not detecting two red objects
            result = draw_text(result, "Need two red objects", (10, 20), (0, 0, 255), 18)

        # Display calibration status (smaller, moved to corner)
        if calibration_mode:
            result = draw_text(result, "Calibration mode - Place objects at exact distance", (10, 80), (255, 0, 0), 14)
            result = draw_text(result, f"Calibration distance: {calibration_distance} cm", (10, 100), (255, 0, 0), 14)
        elif object_size_calibration:
            result = draw_text(result, "Size calibration - Place object at exact distance", (10, 80), (255, 0, 0), 14)
            result = draw_text(result, f"Known width: {known_width} cm", (10, 100), (255, 0, 0), 14)
            result = draw_text(result, f"Known distance: {calibration_distance} cm", (10, 120), (255, 0, 0), 14)

        # Display focal length status
        if focal_length is not None:
            result = draw_text(result, f"F: {focal_length:.1f}", (result.shape[1] - 80, 20), (255, 255, 0), 14)

        # Show results
        cv2.imshow("Red Point Distance Measurement", result)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Enter object distance calibration mode
            calibration_mode = True
            calibration_input = input("Enter actual distance between red points (cm): ")
            try:
                calibration_distance = float(calibration_input)
                print(f"Keep the two red points at a distance of {calibration_distance} cm, then press any key...")
                input()
            except ValueError:
                print("Invalid input, using default value of 10 cm")
                calibration_distance = 10
        elif key == ord('s'):
            # Enter object size calibration mode (for calculating camera to object distance)
            object_size_calibration = True
            width_input = input("Enter actual width of red point (cm): ")
            distance_input = input("Enter actual distance from red point to camera (cm): ")
            try:
                known_width = float(width_input)
                calibration_distance = float(distance_input)
                print(f"Place the red point with width {known_width} cm at {calibration_distance} cm from camera, then press any key...")
                input()

                # If red object detected, use first one for calibration
                if len(red_objects) > 0:
                    pixel_width = red_objects[0]["width"]
                    focal_length = calculate_focal_length(pixel_width, known_width, calibration_distance)
                    print(f"Focal length calibration complete! Focal length = {focal_length:.2f}")
                else:
                    print("No red object detected, calibration failed")

                object_size_calibration = False
            except ValueError:
                print("Invalid input, calibration failed")
                object_size_calibration = False

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()