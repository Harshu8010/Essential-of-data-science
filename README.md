# Essential-of-data-science
import cv2
import numpy as np

# Capture video stream
cap = cv2.VideoCapture('0')  # Use 0 for webcam, or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Region of Interest (ROI) Selection
    mask = np.zeros_like(edges)
    roi = np.array([[(100, frame.shape[0]), (frame.shape[1]//2-50, frame.shape[0]//2+50),
                     (frame.shape[1]//2+50, frame.shape[0]//2+50), (frame.shape[1]-100, frame.shape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=5)

    # Draw Lane Lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Display Output
    cv2.imshow('0', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
