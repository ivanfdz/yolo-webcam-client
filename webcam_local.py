"""
YOLO running locally against your webcam.
Everything runs on your own machine, no remote server involved.

Usage:
    python webcam_local.py
    python webcam_local.py --model yolov8n.pt
    python webcam_local.py --camera 1

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import cv2
import time
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO local webcam inference")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model (default: yolov8n.pt)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("Error: could not open the camera")
        return

    print(f"Model: {args.model}")
    print("Press 'q' to quit")

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('YOLO Webcam', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
