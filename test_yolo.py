import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names   = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open camera
cap = cv2.VideoCapture(0)

print("YOLO loaded. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (640, 480))
    h, w  = small.shape[:2]

    # Convert frame to blob (YOLO input format)
    blob = cv2.dnn.blobFromImage(
        small, 1/255.0, (416, 416),
        swapRB=True, crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes      = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores     = detection[5:]
            class_id   = np.argmax(scores)
            confidence = scores[class_id]

            # Only detect PEOPLE (class_id 0 in COCO dataset)
            # and only if confidence is above 50%
            if class_id == 0 and confidence > 0.5:
                # Convert center coordinates to corner coordinates
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x  = cx - bw // 2
                y  = cy - bh // 2

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))

    # Apply NMS to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_count = 0

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf          = int(confidences[i] * 100)
            person_count += 1

            cv2.rectangle(small, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(
                small, f"Person {conf}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    cv2.putText(
        small,
        f"People detected: {person_count}   Press Q to quit",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 0) if person_count > 0 else (0, 0, 255), 2
    )

    cv2.imshow("YOLO Person Detector", small)
    print(f"People: {person_count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()