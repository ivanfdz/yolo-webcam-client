"""
Local client that sends webcam frames to a remote YOLO server
and displays the returned frame with the detections drawn on it.

Usage:
    python webcam_client.py --url http://<server-ip>:5000/process

Requirements:
    pip install opencv-python requests numpy
"""

import argparse
import cv2
import requests
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(description="Webcam client for remote YOLO inference")
    parser.add_argument("--url", required=True, help="Inference server endpoint (e.g. http://1.2.3.4:5000/process)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--width", type=int, default=640, help="Width of the frame to send")
    parser.add_argument("--height", type=int, default=480, help="Height of the frame to send")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: could not open the camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Connecting to {args.url}")
    print("Press 'q' to quit")

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from the camera")
            break

        # Encode frame as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)

        try:
            # Send to the remote server
            response = requests.post(
                args.url,
                data=buffer.tobytes(),
                headers={'Content-Type': 'application/octet-stream'},
                timeout=10
            )

            if response.status_code == 200:
                # Decode the result
                result = cv2.imdecode(
                    np.frombuffer(response.content, np.uint8),
                    cv2.IMREAD_COLOR
                )
                if result is not None:
                    # Show FPS
                    fps_counter += 1
                    elapsed = time.time() - fps_start
                    if elapsed >= 1.0:
                        fps_display = fps_counter / elapsed
                        fps_counter = 0
                        fps_start = time.time()

                    cv2.putText(result, f"FPS: {fps_display:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('YOLO Remote Inference', result)
            else:
                print(f"Server error: {response.status_code}")
                cv2.imshow('YOLO Remote Inference', frame)

        except requests.exceptions.ConnectionError:
            print("Cannot reach the server. Retrying...")
            cv2.putText(frame, "NO CONNECTION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('YOLO Remote Inference', frame)
        except requests.exceptions.Timeout:
            print("Request timed out")
            cv2.imshow('YOLO Remote Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Client closed")


if __name__ == '__main__':
    main()
