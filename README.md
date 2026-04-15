# 🚦 AI Traffic Light Control System (YOLO + OpenCV)

An intelligent traffic light control system that dynamically adjusts signal timings based on real-time vehicle density using computer vision.

This project leverages **YOLO (Ultralytics)** for object detection and **OpenCV** for video processing to detect vehicles and optimize traffic flow.

---

## 🧠 Problem Statement

Traditional traffic signals operate on fixed timers, which leads to:

* Unnecessary waiting during low traffic
* Congestion during peak hours
* Inefficient traffic flow

This system solves that by:
👉 Detecting vehicle density in real-time
👉 Dynamically adjusting traffic light durations

---

## ⚙️ Features

* 🚗 Real-time vehicle detection using YOLOv8
* 📊 Traffic density estimation per lane
* ⏱️ Dynamic signal timing based on congestion
* 🎥 Works with video input / live feed
* 🧩 Modular design (detection + control logic separated)

---

## 🏗️ Tech Stack

* **Python**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **NumPy**

---

## 📁 Project Structure

```
traffic_light_system/
│── dataset/               # Training / testing data
│── detect_traffic.py      # Vehicle detection logic
│── traffic_controller.py  # Traffic light decision logic
│── train.py               # Model training (optional)
│── data.yaml              # YOLO dataset config
│── yolov8n.pt             # Pretrained YOLO model
│── requirements.txt
│── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/traffic-light-system.git
cd traffic-light-system

pip install -r requirements.txt
```

---

## ▶️ Usage

Run vehicle detection:

```bash
python detect_traffic.py
```

Run traffic controller:

```bash
python traffic_controller.py
```

---

## 📊 How It Works

1. Video frames are captured using OpenCV
2. YOLO detects vehicles (cars, bikes, trucks, buses)
3. Vehicle count is calculated per frame
4. Traffic density is estimated
5. Signal timing is adjusted dynamically

---

## 📈 Future Improvements

* Multi-lane tracking with region-based detection
* Reinforcement learning for signal optimization
* Integration with real-time CCTV feeds
* Cloud deployment (AWS / GCP)
* Dashboard for traffic analytics

---

## ⚠️ Limitations

* Works best in controlled camera angles
* Accuracy depends on YOLO model quality
* No real-world hardware integration yet

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

MIT License

---

## 💡 Author

Built by ORIGIN — focused on AI + systems + real-world problem solving.
