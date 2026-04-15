import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path="runs/detect/traffic_detector/weights/best.pt"):
        """Initialize the YOLO vehicle detector."""
        self.model = YOLO(model_path)
        self.class_names = ["car", "truck", "bus", "motorcycle"]
        self.colors = {
            "car": (0, 255, 0),
            "truck": (255, 0, 0),
            "bus": (0, 165, 255),
            "motorcycle": (255, 255, 0),
        }

    def detect_vehicles(self, frame, confidence=0.5):
        """
        Detect vehicles in a frame.
        Returns: detections list, annotated frame
        """
        results = self.model(frame, conf=confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

                    # Draw bounding box
                    color = self.colors.get(cls_name, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return detections, frame

    def count_vehicles_in_zones(self, detections, zones):
        """
        Count vehicles in each defined zone (lane).
        zones: dict of {zone_name: (x1, y1, x2, y2)}
        Returns: dict of {zone_name: count}
        """
        zone_counts = {zone: 0 for zone in zones}

        for det in detections:
            cx = (det["bbox"][0] + det["bbox"][2]) // 2
            cy = (det["bbox"][1] + det["bbox"][3]) // 2

            for zone_name, (zx1, zy1, zx2, zy2) in zones.items():
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_counts[zone_name] += 1
                    break

        return zone_counts


# --- Use pretrained COCO model (NO training needed) ---
class PretrainedVehicleDetector:
    """Use YOLOv8 pretrained on COCO dataset — works out of the box."""

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_size="yolov8n.pt"):
        self.model = YOLO(model_size)
        self.colors = {
            "car": (0, 255, 0),
            "truck": (255, 0, 0),
            "bus": (0, 165, 255),
            "motorcycle": (255, 255, 0),
        }

    def detect_vehicles(self, frame, confidence=0.5):
        results = self.model(frame, conf=confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self.VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_name = self.VEHICLE_CLASSES[cls_id]

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    })

                    color = self.colors.get(cls_name, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return detections, frame

    def count_vehicles_in_zones(self, detections, zones):
        zone_counts = {zone: 0 for zone in zones}
        for det in detections:
            cx = (det["bbox"][0] + det["bbox"][2]) // 2
            cy = (det["bbox"][1] + det["bbox"][3]) // 2
            for zone_name, (zx1, zy1, zx2, zy2) in zones.items():
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_counts[zone_name] += 1
                    break
        return zone_counts