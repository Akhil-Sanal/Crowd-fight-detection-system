# Smart Crowd Fight Detection System Using Ultralytics YOLO

---

## 🚀 Overview

This project leverages **Ultralytics YOLO** for real-time fight detection in crowded environments. By integrating AI with surveillance systems, it enhances security by identifying violent activities and triggering alerts. The system can process live video feeds, detect fights, and send real-time notifications.

---

## ⚙️ Installation & Setup

### Step 1: Environment Setup

1. Install Python and create a virtual environment:
   ```sh
   python -m venv fight-detection-env
   source fight-detection-env/bin/activate  # On Windows use: fight-detection-env\Scripts\activate
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

---

## 📂 Step 2: Collect & Prepare Dataset

### 📌 Dataset Selection

- Use public datasets like:
  - [🏒 Hockey Fight Dataset](https://www.crcv.ucf.edu/data/hockey)
  - [📹 UCF Crime Dataset](https://www.crcv.ucf.edu/data/ucf-crime)
- Alternatively, create a **custom dataset** by annotating fight scenes.

### 🎯 Data Annotation

- Use **Roboflow** or **LabelImg** to annotate fight scenes.
- Convert annotations to **YOLO format** (TXT files with bounding box coordinates).

---

## 🎓 Step 3: Train YOLO Model

### 🛠 Clone the Ultralytics Repository
```sh
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
```

### 🎯 Train YOLO on the Fight Detection Dataset
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLO model
model.train(data="data.yaml", epochs=50, imgsz=640)
```

#### 📝 `data.yaml` Format
```yaml
train: path/to/train/images
val: path/to/val/images
nc: 2  # Number of classes (fight, no fight)
names: ["No Fight", "Fight"]
```

---

## 🎥 Step 4: Real-Time Fight Detection

### 🖥️ Run the Trained Model on Live Video
```python
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  # Load trained model
cap = cv2.VideoCapture(0)  # Use webcam or video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = int(box.cls[0])

            if label == 1:  # If fight is detected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"⚠️ Fight Detected {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Fight Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📢 Step 5: Alert & Notification System

- **📨 Send SMS or Email Alerts** when a fight is detected using **Twilio** or **SMTP**.
- **📹 Connect with CCTV security systems** for automated responses.

---

## 🚀 Step 6: Deployment & Optimization

- **Convert Model** to **ONNX or TensorRT** for faster inference.
- **Deploy on Edge Devices** like **Jetson Nano** or **Raspberry Pi**.
- **Integrate with Flask/Django** to create a **web-based fight detection dashboard**.

---

## 🏆 Conclusion

This **Smart Crowd Fight Detection System** provides an **AI-powered solution** for enhancing **public safety**. By detecting violent activities in real-time and triggering alerts, it enables rapid response to security threats.

---


