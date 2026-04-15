import cv2
import numpy as np
import time
from detect_traffic import PretrainedVehicleDetector


class TrafficLight:
    """Represents one traffic light for a lane."""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

    def __init__(self, name):
        self.name = name
        self.state = self.RED
        self.timer = 0

    def set_state(self, state, duration):
        self.state = state
        self.timer = duration


class TrafficLightController:
    """
    Adaptive traffic light controller.
    Allocates green time proportional to vehicle density per lane.
    """

    MIN_GREEN_TIME = 10   # seconds
    MAX_GREEN_TIME = 60   # seconds
    YELLOW_TIME = 3       # seconds
    BASE_GREEN_TIME = 15  # seconds per cycle when no vehicles detected

    def __init__(self, lane_names):
        self.lights = {name: TrafficLight(name) for name in lane_names}
        self.lane_names = lane_names
        self.current_green_index = 0
        self.phase = "GREEN"  # GREEN or YELLOW
        self.phase_end_time = time.time()

    def calculate_green_times(self, vehicle_counts):
        """Calculate green time for each lane based on vehicle count."""
        total = sum(vehicle_counts.values())
        green_times = {}

        for lane in self.lane_names:
            if total == 0:
                green_times[lane] = self.BASE_GREEN_TIME
            else:
                ratio = vehicle_counts.get(lane, 0) / total
                green_time = self.MIN_GREEN_TIME + ratio * (self.MAX_GREEN_TIME - self.MIN_GREEN_TIME)
                green_times[lane] = int(green_time)

        return green_times

    def update(self, vehicle_counts):
        """Update traffic light states based on current time and vehicle counts."""
        current_time = time.time()

        if current_time >= self.phase_end_time:
            if self.phase == "GREEN":
                # Switch to yellow
                self.phase = "YELLOW"
                self.phase_end_time = current_time + self.YELLOW_TIME
                current_lane = self.lane_names[self.current_green_index]
                self.lights[current_lane].set_state(TrafficLight.YELLOW, self.YELLOW_TIME)

            elif self.phase == "YELLOW":
                # Set current lane to red
                current_lane = self.lane_names[self.current_green_index]
                self.lights[current_lane].set_state(TrafficLight.RED, 0)

                # Move to next lane
                self.current_green_index = (self.current_green_index + 1) % len(self.lane_names)
                next_lane = self.lane_names[self.current_green_index]

                # Calculate green times
                green_times = self.calculate_green_times(vehicle_counts)
                duration = green_times[next_lane]

                self.lights[next_lane].set_state(TrafficLight.GREEN, duration)
                self.phase = "GREEN"
                self.phase_end_time = current_time + duration

        return self.get_states()

    def get_states(self):
        return {name: light.state for name, light in self.lights.items()}

    def get_remaining_time(self):
        return max(0, int(self.phase_end_time - time.time()))


def draw_traffic_lights(frame, states, remaining_time, vehicle_counts, x_offset=10, y_offset=10):
    """Draw traffic light status panel on the frame."""
    panel_width = 280
    panel_height = 60 + len(states) * 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset),
                  (x_offset + panel_width, y_offset + panel_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "TRAFFIC CONTROL PANEL", (x_offset + 15, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Timer: {remaining_time}s", (x_offset + 15, y_offset + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    color_map = {
        TrafficLight.RED: (0, 0, 255),
        TrafficLight.YELLOW: (0, 255, 255),
        TrafficLight.GREEN: (0, 255, 0),
    }

    for i, (lane, state) in enumerate(states.items()):
        y = y_offset + 80 + i * 70
        color = color_map.get(state, (255, 255, 255))

        # Light circle
        cv2.circle(frame, (x_offset + 30, y + 15), 15, color, -1)
        cv2.circle(frame, (x_offset + 30, y + 15), 15, (255, 255, 255), 2)

        # Lane info
        count = vehicle_counts.get(lane, 0)
        cv2.putText(frame, f"{lane}", (x_offset + 55, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f"{state} | Vehicles: {count}", (x_offset + 55, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return frame


def draw_zones(frame, zones):
    """Draw detection zones on frame."""
    for zone_name, (x1, y1, x2, y2) in zones.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, zone_name, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return frame


# ============================================================
# MAIN APPLICATION
# ============================================================

def run_with_video(video_source=0):
    """
    Run traffic light control system.
    video_source: 0 for webcam, or path to video file
    """
    # Initialize detector (pretrained — no training needed!)
    detector = PretrainedVehicleDetector(model_size="yolov8n.pt")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {video_source}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define 4 detection zones (adjust coordinates to your video/camera)
    mid_x = frame_width // 2
    mid_y = frame_height // 2
    zones = {
        "North": (mid_x, 0, frame_width, mid_y),
        "South": (0, mid_y, mid_x, frame_height),
        "East": (mid_x, mid_y, frame_width, frame_height),
        "West": (0, 0, mid_x, mid_y),
    }

    # Initialize traffic controller
    controller = TrafficLightController(lane_names=list(zones.keys()))

    # Start first green phase
    controller.phase = "GREEN"
    controller.phase_end_time = time.time() + controller.BASE_GREEN_TIME
    first_lane = controller.lane_names[0]
    controller.lights[first_lane].set_state(TrafficLight.GREEN, controller.BASE_GREEN_TIME)

    print("=" * 50)
    print("  ADAPTIVE TRAFFIC LIGHT CONTROL SYSTEM")
    print("  Press 'q' to quit | Press 's' to screenshot")
    print("=" * 50)

    vehicle_counts = {zone: 0 for zone in zones}

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect vehicles
        detections, annotated_frame = detector.detect_vehicles(frame, confidence=0.4)

        # Count vehicles per zone
        vehicle_counts = detector.count_vehicles_in_zones(detections, zones)

        # Update traffic light controller
        states = controller.update(vehicle_counts)
        remaining = controller.get_remaining_time()

        # Draw zones
        annotated_frame = draw_zones(annotated_frame, zones)

        # Draw traffic light panel
        annotated_frame = draw_traffic_lights(
            annotated_frame, states, remaining, vehicle_counts
        )

        # Total vehicle count
        total = sum(vehicle_counts.values())
        cv2.putText(annotated_frame, f"Total Vehicles: {total}",
                    (frame_width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display
        cv2.imshow("Traffic Light Control System", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("screenshot.jpg", annotated_frame)
            print("Screenshot saved!")

    cap.release()
    cv2.destroyAllWindows()


def run_with_images(image_paths):
    """Run detection on a list of images (for testing)."""
    detector = PretrainedVehicleDetector(model_size="yolov8n.pt")

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Cannot read: {img_path}")
            continue

        detections, annotated = detector.detect_vehicles(frame, confidence=0.4)
        print(f"\n{img_path}: {len(detections)} vehicles detected")
        for det in detections:
            print(f"  - {det['class_name']} (conf: {det['confidence']:.2f})")

        cv2.imshow("Detection", annotated)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # --- Option 1: Run with webcam ---
     run_with_video(0)

    # --- Option 2: Run with a traffic video file ---
   # run_with_video("traffic_video.mp4")

    # --- Option 3: Test on images ---
    # run_with_images(["test1.jpg", "test2.jpg"])