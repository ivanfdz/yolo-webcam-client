# yolo-webcam-client

Detección de objetos en tiempo real sobre la webcam con YOLOv8, en dos modos: todo en local, o
enviando los frames a un servidor remoto (una EC2 con GPU, por ejemplo) que hace la inferencia y
devuelve la imagen ya anotada.

Son dos scripts independientes, sin dependencias entre ellos:

| Script | Dónde corre la inferencia | Para qué sirve |
| --- | --- | --- |
| `webcam_local.py` | en tu máquina | probar YOLO sin infra: carga el modelo, lee la cámara y pinta las cajas |
| `webcam_client.py` | en un servidor remoto | tu máquina solo captura y muestra; el modelo vive en el servidor |

En los dos casos la ventana muestra los FPS reales medidos, y se cierra con `q`.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`webcam_local.py` necesita `ultralytics` (que arrastra PyTorch, ~2 GB). Si solo vas a usar el
modo remoto, te basta con `opencv-python`, `requests` y `numpy`.

En macOS, la primera ejecución pide permiso de cámara para el terminal desde el que lanzas el
script. Si no aparece el diálogo, actívalo a mano en Ajustes del Sistema → Privacidad y
seguridad → Cámara.

## Modo local

```bash
python webcam_local.py
python webcam_local.py --model yolov8s.pt --camera 1 --conf 0.35
```

| Flag | Default | Qué hace |
| --- | --- | --- |
| `--model` | `yolov8n.pt` | pesos a cargar. Si el fichero no existe, `ultralytics` lo descarga solo |
| `--camera` | `0` | índice del dispositivo de captura |
| `--conf` | `0.5` | umbral de confianza; por debajo, la detección se descarta |

`yolov8n` es el más pequeño de la familia (nano) y es el que da FPS decentes en CPU. Si tienes
GPU disponible, `yolov8s`/`yolov8m` mejoran la precisión a costa de velocidad.

## Modo remoto

```bash
python webcam_client.py --url http://<ip-del-servidor>:5000/process
python webcam_client.py --url http://1.2.3.4:5000/process --quality 60 --width 1280 --height 720
```

| Flag | Default | Qué hace |
| --- | --- | --- |
| `--url` | *obligatorio* | endpoint del servidor de inferencia |
| `--camera` | `0` | índice del dispositivo de captura |
| `--quality` | `80` | calidad JPEG del frame que se envía (1-100) |
| `--width` / `--height` | `640` / `480` | resolución que se pide a la cámara |

El bucle es: capturar frame → codificar a JPEG → `POST` al servidor → decodificar la respuesta →
mostrar. Bajar `--quality` y la resolución es la palanca directa para reducir latencia, porque
lo que domina el tiempo por frame es el tamaño del cuerpo de la petición.

Los errores de red no matan el proceso: ante `ConnectionError` pinta "SIN CONEXION" sobre el
frame crudo y sigue reintentando; ante timeout (10 s) muestra el frame sin anotar y continúa.

### Contrato con el servidor

El servidor no está en este repo. El cliente espera algo muy simple:

- `POST` al `--url` con `Content-Type: application/octet-stream`
- cuerpo: los bytes de un JPEG
- respuesta `200` con los bytes de un JPEG ya anotado
- cualquier otro código se registra en consola y el cliente muestra el frame original

Un servidor mínimo compatible, con Flask y ultralytics:

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

### Nota de seguridad

El protocolo va en HTTP plano y sin autenticación. Vale para una prueba en una red de confianza,
pero si expones ese puerto a internet estás publicando un endpoint anónimo que consume tu GPU y
por el que viaja el vídeo de tu cámara en claro. Para cualquier uso real: túnel SSH o VPN al
servidor, o TLS más un token en la petición, y el grupo de seguridad restringido a tu IP.

## Ficheros

```
webcam_local.py    inferencia en local con ultralytics
webcam_client.py   captura + display; la inferencia la hace el servidor remoto
requirements.txt   dependencias
```

Los pesos (`*.pt`) están fuera del control de versiones: `ultralytics` los descarga en la
primera ejecución.
