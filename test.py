import cv2

for i in range(10):  # Try indices from 0 to 9
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        break
    cap.release()
    print(f"Camera {i} is available.")
