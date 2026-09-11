"""
Cliente local para enviar frames de la webcam a la EC2 con YOLO
y mostrar el resultado con las detecciones.

Uso:
    python webcam_client.py --url http://<ip-ec2>:5000/process

Requisitos:
    pip install opencv-python requests numpy
"""

import argparse
import cv2
import requests
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(description="Webcam client for remote YOLO inference")
    parser.add_argument("--url", required=True, help="URL del endpoint EC2 (ej: http://1.2.3.4:5000/process)")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0)")
    parser.add_argument("--quality", type=int, default=80, help="Calidad JPEG 1-100 (default: 80)")
    parser.add_argument("--width", type=int, default=640, help="Ancho del frame a enviar")
    parser.add_argument("--height", type=int, default=480, help="Alto del frame a enviar")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Conectando a {args.url}")
    print("Pulsa 'q' para salir")

    fps_counter = 0
    fps_start = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame de la cámara")
            break

        # Codifica frame como JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)

        try:
            # Envía al servidor EC2
            response = requests.post(
                args.url,
                data=buffer.tobytes(),
                headers={'Content-Type': 'application/octet-stream'},
                timeout=10
            )

            if response.status_code == 200:
                # Decodifica resultado
                result = cv2.imdecode(
                    np.frombuffer(response.content, np.uint8),
                    cv2.IMREAD_COLOR
                )
                if result is not None:
                    # Muestra FPS
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
                print(f"Error del servidor: {response.status_code}")
                cv2.imshow('YOLO Remote Inference', frame)

        except requests.exceptions.ConnectionError:
            print("No se puede conectar al servidor. Reintentando...")
            cv2.putText(frame, "SIN CONEXION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('YOLO Remote Inference', frame)
        except requests.exceptions.Timeout:
            print("Timeout en la petición")
            cv2.imshow('YOLO Remote Inference', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cliente cerrado")


if __name__ == '__main__':
    main()
