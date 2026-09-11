# yolo-webcam-client

Real-time object detection on your webcam with YOLOv8, in two modes: everything local, or
streaming frames to a remote server (a GPU EC2 instance, for example) that runs inference and
returns the annotated image.

These are two independent scripts with no dependency on each other:

| Script | Where inference runs | What it's for |
| --- | --- | --- |
| `webcam_local.py` | on your machine | trying YOLO with no infrastructure: loads the model, reads the camera, draws the boxes |
| `webcam_client.py` | on a remote server | your machine only captures and displays; the model lives on the server |

In both cases the window shows the measured FPS, and `q` closes it.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`webcam_local.py` needs `ultralytics`, which pulls in PyTorch (~2 GB). If you only plan to use
remote mode, `opencv-python`, `requests` and `numpy` are enough.

On macOS, the first run asks for camera permission for the terminal you launch the script from.
If the dialog never appears, enable it manually under System Settings → Privacy & Security →
Camera.

## Local mode

```bash
python webcam_local.py
python webcam_local.py --model yolov8s.pt --camera 1 --conf 0.35
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | `yolov8n.pt` | weights to load. If the file is missing, `ultralytics` downloads it |
| `--camera` | `0` | capture device index |
| `--conf` | `0.5` | confidence threshold; detections below it are discarded |

`yolov8n` is the smallest of the family (nano) and the one that gives usable FPS on CPU. With a
GPU available, `yolov8s`/`yolov8m` trade speed for accuracy.

## Remote mode

```bash
python webcam_client.py --url http://<server-ip>:5000/process
python webcam_client.py --url http://1.2.3.4:5000/process --quality 60 --width 1280 --height 720
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--url` | *required* | inference server endpoint |
| `--camera` | `0` | capture device index |
| `--quality` | `80` | JPEG quality of the frame sent (1-100) |
| `--width` / `--height` | `640` / `480` | resolution requested from the camera |

The loop is: capture frame → encode as JPEG → `POST` to the server → decode the response →
display. Lowering `--quality` and the resolution is the direct lever for reducing latency,
because request body size is what dominates per-frame time.

Network errors don't kill the process: on `ConnectionError` it draws "NO CONNECTION" over the raw
frame and keeps retrying; on timeout (10 s) it shows the unannotated frame and carries on.

### Server contract

The server is not in this repo. The client expects something very simple:

- `POST` to `--url` with `Content-Type: application/octet-stream`
- body: the bytes of a JPEG
- `200` response with the bytes of an annotated JPEG
- any other status code is logged to the console and the client shows the original frame

A minimal compatible server, using Flask and ultralytics:

```python
from flask import Flask, request, Response
from ultralytics import YOLO
import cv2, numpy as np

app = Flask(__name__)
model = YOLO("yolov8n.pt")

@app.post("/process")
def process():
    frame = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    annotated = model(frame, verbose=False)[0].plot()
    _, buf = cv2.imencode(".jpg", annotated)
    return Response(buf.tobytes(), mimetype="image/jpeg")

app.run(host="0.0.0.0", port=5000)
```

### Security note

The protocol is plain HTTP with no authentication. Fine for a test on a trusted network, but
exposing that port to the internet means publishing an anonymous endpoint that burns your GPU
and carries your camera feed in the clear. For anything real: SSH tunnel or VPN to the server,
or TLS plus a token in the request, and the security group restricted to your IP.

## Files

```
webcam_local.py    local inference with ultralytics
webcam_client.py   capture + display; a remote server does the inference
requirements.txt   dependencies
```

Weights (`*.pt`) are kept out of version control: `ultralytics` downloads them on first run.
