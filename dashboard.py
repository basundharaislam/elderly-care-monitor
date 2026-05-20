import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
import winsound
import threading
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---- Page config ----
st.set_page_config(
    page_title="Elderly Care AI Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Dark theme CSS ----
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card {
        background: #1c1f2e;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #8b8fa8; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 600; color: #ffffff; }
    .metric-value.safe   { color: #00c896; }
    .metric-value.warn   { color: #ffa500; }
    .metric-value.danger { color: #ff4b4b; }
    .flash-alert {
        background: #ff4b4b22;
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        color: #ff4b4b;
        animation: pulse 1s infinite;
        margin: 10px 0;
    }
    @keyframes pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .safe-banner {
        background: #00c89622;
        border: 2px solid #00c896;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #00c896;
        margin: 10px 0;
    }
    .warn-banner {
        background: #ffa50022;
        border: 2px solid #ffa500;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        color: #ffa500;
        margin: 10px 0;
    }
    .sidebar-section {
        background: #1c1f2e;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# YOLO SETUP
# ======================================================
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names   = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

net, classes, output_layers = load_yolo()

def detect_people(frame):
    h, w  = frame.shape[:2]
    blob  = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes       = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores     = detection[5:]
            class_id   = np.argmax(scores)
            confidence = scores[class_id]

            # Only people (class 0) above 50% confidence
            if class_id == 0 and confidence > 0.5:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x  = cx - bw // 2
                y  = cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    # Remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    final_boxes = []
    final_confs = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_confs.append(confidences[i])

    return final_boxes, final_confs

# ======================================================
# HELPER FUNCTIONS
# ======================================================
def play_alarm():
    def beep():
        for _ in range(3):
            winsound.Beep(880, 300)
            winsound.Beep(660, 300)
    threading.Thread(target=beep, daemon=True).start()

def play_inactivity_beep():
    def beep():
        winsound.Beep(440, 800)
    threading.Thread(target=beep, daemon=True).start()

def get_posture(box_h, frame_h):
    ratio = box_h / frame_h
    if ratio > 0.55:
        return "Good",     "#00c896"
    elif ratio > 0.35:
        return "Bad",      "#ffa500"
    else:
        return "Very bad", "#ff4b4b"

HISTORY_FILE = "session_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_session(data):
    history = load_history()
    history.append(data)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ---- Email ----
SENDER_EMAIL    = "basundharaislam.11s@gmail.com"       # ← your Gmail
SENDER_PASSWORD = "rpsrstiyzkkxwxgx"          # ← your app password

def send_alert_email(caregiver_email, event, timestamp):
    st.toast(f"📧 Email sent to {caregiver_email}", icon="📧")

    def send():
        try:
            msg            = MIMEMultipart("alternative")
            msg["Subject"] = f"ALERT: {event} — Elderly Care Monitor"
            msg["From"]    = SENDER_EMAIL
            msg["To"]      = caregiver_email

            text = f"Event: {event}\nTime: {timestamp}\n\nPlease check on the person immediately."
            html = f"""
<html><body style="font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;">
  <div style="background:white;border-radius:12px;padding:24px;max-width:500px;margin:auto;">
    <h2 style="color:#ff4b4b;">🚨 Elderly Care Alert</h2>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="padding:8px;color:#888;">Event</td>
          <td style="padding:8px;font-weight:bold;">{event}</td></tr>
      <tr style="background:#f9f9f9;">
          <td style="padding:8px;color:#888;">Time</td>
          <td style="padding:8px;">{timestamp}</td></tr>
      <tr><td style="padding:8px;color:#888;">Date</td>
          <td style="padding:8px;">{datetime.now().strftime("%Y-%m-%d")}</td></tr>
    </table>
    <p style="margin-top:20px;color:#555;">
      This is an automated alert from your <strong>Elderly Care AI Monitor</strong>.<br>
      Please check on the person immediately.
    </p>
    <div style="margin-top:20px;padding:12px;background:#ff4b4b22;
                border-left:4px solid #ff4b4b;border-radius:4px;">
      <strong style="color:#ff4b4b;">Immediate attention may be required.</strong>
    </div>
  </div>
</body></html>"""

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html,  "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, caregiver_email, msg.as_string())

            print(f"Email sent to {caregiver_email}")

        except Exception as e:
            print(f"Email error: {e}")

    threading.Thread(target=send, daemon=True).start()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Detection**")
    FALL_THRESHOLD   = st.slider("Fall confirm time (s)", 1.0, 5.0, 2.0, 0.5)
    INACTIVITY_LIMIT = st.slider("Inactivity alert (s)",  10,  120,  60,    5)
    CONFIDENCE_MIN   = st.slider("Min detection confidence", 0.3, 0.9, 0.5, 0.05)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Alerts**")
    sound_on  = st.toggle("Sound alarm",            value=True)
    email_on  = st.toggle("Email alerts",           value=True)
    caregiver = st.text_input("Caregiver email", "caregiver@example.com")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Session history**")
    history   = load_history()
    if history:
        st.success(f"{len(history)} past session(s)")
        show_hist = st.button("View history")
    else:
        st.info("No past sessions yet")
        show_hist = False
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#8b8fa8;'>
    Elderly Care AI Monitor<br>
    Powered by YOLOv3 + OpenCV<br>
    No wearables required
    </div>""", unsafe_allow_html=True)

# ======================================================
# SESSION HISTORY PAGE
# ======================================================
if show_hist:
    st.markdown("## 📊 Session History")
    history = load_history()
    if history:
        rows = [{
            "Session"           : f"Session {i+1}",
            "Date"              : s.get("date",              "—"),
            "Duration"          : s.get("duration",          "—"),
            "Falls"             : s.get("falls",              0),
            "Inactivity alerts" : s.get("inactivity_alerts",  0),
        } for i, s in enumerate(history)]

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("#### Falls per session")
        st.bar_chart(pd.DataFrame(
            {"Falls": [r["Falls"] for r in rows]},
            index=[r["Session"] for r in rows]
        ))

        st.markdown("#### Inactivity alerts per session")
        st.bar_chart(pd.DataFrame(
            {"Inactivity alerts": [r["Inactivity alerts"] for r in rows]},
            index=[r["Session"] for r in rows]
        ))

        if st.button("Clear all history"):
            os.remove(HISTORY_FILE)
            st.success("History cleared.")
    else:
        st.info("No sessions saved yet.")
    st.stop()

# ======================================================
# MAIN PAGE
# ======================================================
st.markdown("# 🏥 Elderly Care AI Monitor")
st.markdown("Real-time multi-person fall, posture & inactivity detection — no wearables needed")
st.divider()

# Metric cards
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
status_box     = mc1.empty()
falls_box      = mc2.empty()
floor_time_box = mc3.empty()
posture_box    = mc4.empty()
session_box    = mc5.empty()

alert_banner = st.empty()

left_col, right_col = st.columns([3, 2])
with left_col:
    st.markdown("#### Live feed")
    frame_display = st.empty()

with right_col:
    st.markdown("#### Fall risk score (last 60 frames)")
    risk_chart_area = st.empty()
    st.markdown("#### Event log")
    log_display = st.empty()
    log_display.info("Monitoring started — no events yet.")

st.divider()
stop_button = st.button("⏹ Stop monitoring", type="primary")

# ======================================================
# STATE
# ======================================================
cap = cv2.VideoCapture(0)

fall_start_time    = None
fall_confirmed     = False
floor_start_time   = None
last_motion_time   = datetime.now()
inactivity_alerted = False
inactivity_count   = 0
fall_count         = 0
fall_log           = []
risk_history       = []
session_start      = datetime.now()
prev_boxes         = []

BOX_COLORS = [
    (0,   200,   0),
    (255, 165,   0),
    (200,   0, 200),
    (0,   200, 255),
]

# ======================================================
# MAIN LOOP
# ======================================================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        st.error("Camera disconnected.")
        break

    small   = cv2.resize(frame, (640, 480))
    frame_h = small.shape[0]
    frame_w = small.shape[1]
    now     = datetime.now()
    elapsed = int((now - session_start).total_seconds())
    mins    = elapsed // 60
    secs    = elapsed % 60

    # ---- YOLO detection ----
    boxes, confs = detect_people(small)
    num_people   = len(boxes)

    # ---- Motion detection ----
    if num_people > 0 and len(prev_boxes) > 0:
        try:
            motion = np.mean(np.abs(
                np.array(boxes[:len(prev_boxes)]) -
                np.array(prev_boxes[:len(boxes)])
            ))
            if motion > 5:
                last_motion_time   = now
                inactivity_alerted = False
        except Exception:
            last_motion_time = now
    elif num_people > 0:
        last_motion_time = now

    prev_boxes = list(boxes)

    # ---- Inactivity check ----
    seconds_inactive = (now - last_motion_time).total_seconds()
    if (seconds_inactive >= INACTIVITY_LIMIT
            and not inactivity_alerted
            and num_people > 0):
        inactivity_alerted = True
        inactivity_count  += 1
        ts = now.strftime("%H:%M:%S")
        fall_log.append({
            "Time"   : ts,
            "Event"  : "Inactivity alert",
            "Detail" : f"No movement for {int(seconds_inactive)}s"
        })
        if sound_on:
            play_inactivity_beep()
        if email_on:
            send_alert_email(caregiver, "Inactivity alert", ts)

    # ---- Default frame state ----
    overall_status   = "No person detected"
    overall_class    = "warn"
    overall_color_cv = (128, 128, 128)
    posture_label    = "—"
    floor_seconds    = 0
    risk_score       = 0

    # ---- Per person ----
    for i, (x, y, bw, bh) in enumerate(boxes):

        color_cv = BOX_COLORS[i % len(BOX_COLORS)]
        conf     = int(confs[i] * 100)

        # Draw box
        cv2.rectangle(small, (x, y), (x+bw, y+bh), color_cv, 2)

        # Label
        cv2.putText(
            small, f"Person {i+1}  {conf}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cv, 2
        )

        # Posture
        posture_label, _ = get_posture(bh, frame_h)
        cv2.putText(
            small, f"Posture: {posture_label}",
            (x, y + bh + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1
        )

        # Fall detection
        frame_center_y = frame_h // 2
        person_is_low  = y > frame_center_y

        if person_is_low:
            risk_score = max(risk_score, 75)

            if floor_start_time is None:
                floor_start_time = now
            floor_seconds = (now - floor_start_time).total_seconds()

            if fall_start_time is None:
                fall_start_time = now
            seconds_down = (now - fall_start_time).total_seconds()

            if seconds_down >= FALL_THRESHOLD:
                risk_score = 100
                if not fall_confirmed:
                    fall_confirmed  = True
                    fall_count     += 1
                    ts              = now.strftime("%H:%M:%S")
                    fall_log.append({
                        "Time"   : ts,
                        "Event"  : f"Fall — Person {i+1}",
                        "Detail" : f"Down for {round(seconds_down,1)}s"
                    })
                    if sound_on:
                        play_alarm()
                    if email_on:
                        send_alert_email(
                            caregiver,
                            f"Fall detected (Person {i+1})",
                            ts
                        )

                overall_status   = "FALL DETECTED"
                overall_class    = "danger"
                overall_color_cv = (0, 0, 255)

            else:
                remaining        = round(FALL_THRESHOLD - seconds_down, 1)
                overall_status   = f"Low position ({remaining}s)"
                overall_class    = "warn"
                overall_color_cv = (0, 165, 255)
                risk_score       = max(risk_score, 60)

        else:
            if i == 0:
                fall_start_time  = None
                fall_confirmed   = False
                floor_start_time = None
                if overall_status != "FALL DETECTED":
                    overall_status   = "Safe"
                    overall_class    = "safe"
                    overall_color_cv = (0, 200, 0)
                    risk_score       = max(risk_score, 10)

    # ---- Top bar on video ----
    cv2.rectangle(small, (0, 0), (frame_w, 48), (14, 17, 23), -1)
    cv2.putText(
        small,
        f"Status: {overall_status}   People: {num_people}   "
        f"Falls: {fall_count}   {mins:02d}:{secs:02d}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, overall_color_cv, 2
    )

    # ---- Risk history ----
    risk_history.append(int(risk_score))
    if len(risk_history) > 60:
        risk_history.pop(0)

    # ---- Update UI ----
    status_box.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value {overall_class}">{overall_status}</div>
        </div>""", unsafe_allow_html=True)

    falls_box.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Falls detected</div>
            <div class="metric-value {'danger' if fall_count > 0 else 'safe'}">{fall_count}</div>
        </div>""", unsafe_allow_html=True)

    floor_time_box.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Time on floor</div>
            <div class="metric-value warn">{round(floor_seconds,1)}s</div>
        </div>""", unsafe_allow_html=True)

    posture_box.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Posture</div>
            <div class="metric-value">{posture_label}</div>
        </div>""", unsafe_allow_html=True)

    session_box.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Session time</div>
            <div class="metric-value">{mins:02d}:{secs:02d}</div>
        </div>""", unsafe_allow_html=True)

    # Alert banner
    if "FALL" in overall_status:
        alert_banner.markdown(
            '<div class="flash-alert">🚨 FALL DETECTED — Immediate attention required</div>',
            unsafe_allow_html=True
        )
    elif seconds_inactive >= INACTIVITY_LIMIT and num_people > 0:
        alert_banner.markdown(
            '<div class="warn-banner">⚠️ INACTIVITY ALERT — No movement detected</div>',
            unsafe_allow_html=True
        )
    else:
        alert_banner.markdown(
            '<div class="safe-banner">✅ Monitoring active — all clear</div>',
            unsafe_allow_html=True
        )

    # Video
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    frame_display.image(rgb_frame, use_container_width=True)

    # Risk chart
    risk_chart_area.line_chart(
        pd.DataFrame({"Fall risk score": risk_history}),
        use_container_width=True
    )

    # Event log
    if fall_log:
        log_display.dataframe(
            pd.DataFrame(fall_log),
            use_container_width=True
        )

    if stop_button:
        break

# ======================================================
# SESSION ENDED
# ======================================================
cap.release()

end_elapsed = int((datetime.now() - session_start).total_seconds())
dur_str     = f"{end_elapsed // 60:02d}:{end_elapsed % 60:02d}"

save_session({
    "date"              : session_start.strftime("%Y-%m-%d %H:%M"),
    "duration"          : dur_str,
    "falls"             : fall_count,
    "inactivity_alerts" : inactivity_count,
})

st.divider()
st.success("✅ Session ended and saved.")

ca, cb, cc = st.columns(3)
ca.metric("Total falls",       fall_count)
cb.metric("Inactivity alerts", inactivity_count)
cc.metric("Session duration",  dur_str)

if fall_log:
    st.subheader("Full session report")
    df_final = pd.DataFrame(fall_log)
    st.dataframe(df_final, use_container_width=True)
    st.download_button(
        label     = "⬇️ Download report as CSV",
        data      = df_final.to_csv(index=False),
        file_name = f"session_{session_start.strftime('%Y%m%d_%H%M%S')}.csv",
        mime      = "text/csv"
    )
else:
    st.info("No fall events this session.")