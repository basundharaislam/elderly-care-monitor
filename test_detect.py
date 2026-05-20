import cv2

body_detector = cv2.HOGDescriptor()
body_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

print("Camera opened. Stand back so your FULL body is visible. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to standard size
    small = cv2.resize(frame, (640, 480))

    # Most aggressive detection settings possible
    boxes, weights = body_detector.detectMultiScale(
        small,
        winStride=(2, 2),
        padding=(16, 16),
        scale=1.01
    )

    # Print result in terminal every frame
    print(f"People detected: {len(boxes)}   Boxes: {boxes}")

    # Draw boxes if any found
    for (x, y, w, h) in boxes:
        cv2.rectangle(small, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show count on screen
    cv2.putText(
        small,
        f"People detected: {len(boxes)}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if len(boxes) > 0 else (0, 0, 255),
        2
    )

    cv2.imshow("Detection Test", small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
