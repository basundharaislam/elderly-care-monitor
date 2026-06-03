# 🏥 Elderly Care AI Monitor

> Real-time fall detection, posture analysis, and inactivity monitoring — no wearables required.

Built as part of my internship application to Japan's tech and research ecosystem,
this project addresses one of Japan's most critical challenges:
**30% of Japan's population is over 65**, and fall-related injuries are a leading
cause of hospitalization among the elderly.

---

## 🎯 Problem Statement

Traditional elderly care monitoring requires expensive wearable sensors or
constant human supervision. This system uses only a standard camera to detect:

- Falls in real time
- Poor posture before it becomes dangerous
- Prolonged inactivity that may indicate distress
- Multiple people simultaneously

No wearables. No special hardware. Just a camera.

---

## 🚀 Live Demo

> 📹 Demo video coming soon

---

## ✨ Features

| Feature | Description |
|---|---|
| 🚨 Fall detection | Detects falls and confirms after 2 seconds to avoid false alarms |
| 🧍 Posture scoring | Rates posture as Good / Bad / Very Bad in real time |
| ⚠️ Inactivity alert | Triggers if no movement detected for 60 seconds |
| 👥 Multi-person | Tracks multiple people simultaneously with separate color boxes |
| 🔊 Sound alarm | Beeps immediately when fall confirmed |
| 📧 Email alerts | Sends formatted HTML alert email to caregiver instantly |
| 📊 Live risk graph | Shows fall risk score over last 60 frames |
| 📋 Session history | Saves every session with fall count and duration |
| ⬇️ CSV download | Download full event log after each session |
| 🌙 Dark UI | Professional dark theme dashboard built with Streamlit |

---

## 🧠 How It Works
1. **YOLOv3** detects people in each frame with confidence scores
2. Bounding box position determines if person is upright or fallen
3. A **2-second confirmation timer** prevents false alarms from crouching
4. **Posture score** is calculated from bounding box height ratio
5. **Motion tracking** compares box positions between frames for inactivity detection
6. All events are logged with timestamps and exportable as CSV

---

## 🛠️ Tech Stack

- **Python 3.13**
- **OpenCV** — camera feed and YOLOv3 inference
- **YOLOv3** — real-time person detection (COCO dataset)
- **Streamlit** — live dashboard UI
- **Pandas / NumPy** — data processing
- **winsound** — alarm system (Windows built-in)
- **smtplib** — email alerts via Gmail SMTP

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/basundharaislam/elderly-care-monitor.git
cd elderly-care-monitor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLO model files
Download these 3 files and place them in the project folder:

- **yolov3.weights** (236MB): https://pjreddie.com/media/files/yolov3.weights
- **yolov3.cfg**: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
- **coco.names**: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

### 4. Configure email alerts (optional)
In `dashboard.py`, update these two lines with your Gmail and app password:
```python
SENDER_EMAIL    = "your_gmail@gmail.com"
SENDER_PASSWORD = "your_app_password"
```

### 5. Run the dashboard
```bash
streamlit run dashboard.py
```

---

## 📁 Project Structure
---

## 🎮 How to Use

| Action | Result |
|---|---|
| Stand normally | ✅ "Safe" in green |
| Slouch forward | ⚠️ "Bad posture" warning |
| Crouch briefly | 🟠 "Low position" — no alarm |
| Stay on floor 2s | 🚨 Fall confirmed — alarm + email |
| No movement 60s | ⚠️ Inactivity alert — email sent |
| Press Stop | 📊 Session report + CSV download |

---

## 📈 Future Improvements

- [ ] Train custom fall detection model on UR Fall Detection Dataset
- [ ] Add SMS alerts via Twilio
- [ ] Deploy on Raspberry Pi for low-cost home installation
- [ ] Add night vision / infrared camera support
- [ ] Mobile app companion for caregivers
- [ ] Integrate with smart home systems (Google Home, Alexa)

---

## 👤 Author

**Basundhara Islam**
Final year B.Tech CSE student | AI/ML & Computer Vision
📧 basundharaislam.11s@gmail.com
🔗 [GitHub](https://github.com/basundharaislam) | [LinkedIn](https://linkedin.com/in/basundharaislam-45636b237)

---

## 📄 License

MIT License — free to use, modify, and distribute.