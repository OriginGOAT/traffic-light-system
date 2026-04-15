from ultralytics import YOLO

def train_model():
    # Load a pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")  # nano version (fast), use yolov8s.pt for better accuracy

    # Train on your custom dataset
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="traffic_detector",
        patience=10,
        save=True,
        device=0  # use 'cpu' if no GPU
    )

    print("Training complete!")
    print(f"Best model saved at: runs/detect/traffic_detector/weights/best.pt")

if __name__ == "__main__":
    train_model()