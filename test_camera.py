import cv2
import csv
from datetime import datetime

# Load built-in person detector
body_detector = cv2.HOGDescriptor()
body_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam
cap = cv2.VideoCapture(0)

# These variables track fall timing
# We don't want to alert on a single crouch — only sustained falls
fall_start_time = None        # when did the fall position start?
fall_confirmed = False        # has it lasted long enough to be a real fall?
FALL_TIME_THRESHOLD = 2.0     # seconds person must be down before alert triggers

# This list stores all fall events during the session
fall_log = []

# Count total falls this session
fall_count = 0

# Track session start
session_start = datetime.now()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    small = cv2.resize(frame, (640, 480))

    # Detect people
    boxes, weights = body_detector.detectMultiScale(
        small,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )

    # Default status when no person detected
    status = "No person detected"
    status_color = (128, 128, 128)    # gray

    for (x, y, w, h) in boxes:

        # Draw box around person
        cv2.rectangle(
            small,
            (x, y),
            (x + w, y + h),
            (0, 200, 0),
            2
        )

        # Frame center (vertical midpoint of screen)
        frame_center_y = small.shape[0] // 2

        # Check if person is in fallen position
        # y is the TOP of the detected box
        # If the top of the box is below screen center, person is low down
        person_is_low = y > frame_center_y

        if person_is_low:
            # Start timing the fall if not already timing
            if fall_start_time is None:
                fall_start_time = datetime.now()

            # How many seconds have they been down?
            seconds_down = (datetime.now() - fall_start_time).total_seconds()

            if seconds_down >= FALL_TIME_THRESHOLD:
                # Fall confirmed — they've been down long enough
                if not fall_confirmed:
                    # First time confirming this fall — log it
                    fall_confirmed = True
                    fall_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    fall_log.append({
                        "Time": timestamp,
                        "Event": "Fall detected",
                        "Duration so far (s)": round(seconds_down, 1)
                    })
                    print(f"FALL DETECTED at {timestamp}")

                status = f"FALL DETECTED"
                status_color = (0, 0, 255)    # red

            else:
                # They're low but not long enough yet — show warning
                remaining = round(FALL_TIME_THRESHOLD - seconds_down, 1)
                status = f"Low position... ({remaining}s)"
                status_color = (0, 165, 255)    # orange

        else:
            # Person is upright — reset fall tracking
            fall_start_time = None
            fall_confirmed = False
            status = "Safe"
            status_color = (0, 200, 0)    # green

        # Show status just above the detection box
        cv2.putText(
            small,
            status,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

    # Top bar — black background strip for readability
    cv2.rectangle(small, (0, 0), (640, 80), (0, 0, 0), -1)

    # Session info at top
    elapsed = int((datetime.now() - session_start).total_seconds())
    minutes = elapsed // 60
    seconds = elapsed % 60

    cv2.putText(
        small,
        f"Status: {status}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    cv2.putText(
        small,
        f"Falls: {fall_count}   Session: {minutes:02d}:{seconds:02d}   Press Q to quit",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )

    cv2.imshow("Elderly Care Monitor", small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save fall log to CSV when session ends
if fall_log:
    filename = f"fall_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Time", "Event", "Duration so far (s)"])
        writer.writeheader()
        writer.writerows(fall_log)
    print(f"\nSession ended. {fall_count} fall(s) logged to {filename}")
else:
    print("\nSession ended. No falls detected.")

cap.release()
cv2.destroyAllWindows()