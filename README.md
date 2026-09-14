# Detección y Reconocimiento de Placas Vehiculares — YOLOv8 + EasyOCR + FastAPI

Sistema de detección automática de placas vehiculares con reconocimiento de caracteres (OCR), desplegado como una API REST en **AWS EC2** y accesible **desde el celular** mediante una página web responsiva servida por el mismo backend.

## Características

- **Detección de placas** con un modelo **YOLOv8** entrenado por transfer learning (`best.pt`).
- **Reconocimiento de texto (OCR)** con **EasyOCR**, mejorado con preprocesamiento múltiple del recorte (deskew, Otsu, adaptativo, CLAHE) y selección del resultado por mayor confianza.
- **Interfaz web móvil**: el usuario sube o captura una foto desde el navegador del celular y recibe la placa leída + lectura por voz (TTS).
- **Optimizado para instancias pequeñas**: redimensionado de imágenes a `1280px` (en el servidor y en el navegador), límite de hilos de CPU y `imgsz=640` para reducir memoria y latencia en una `t3.micro`.

## Tecnologías

| Capa | Tecnología |
|------|-----------|
| Backend | FastAPI + Uvicorn |
| Detección | Ultralytics YOLOv8 (`best.pt`) |
| OCR | EasyOCR (idioma `en`) |
| Imágenes | OpenCV (`cv2`) |
| Deep learning | PyTorch (CPU) |
| Nube | AWS EC2 (Ubuntu) + systemd |

## Estructura del proyecto

```
.
├── app.py              # API FastAPI (endpoints + interfaz web embebida)
├── modelo/
│   └── best.pt         # Modelo YOLOv8 entrenado (clase: pl_license_plate)
├── carroprueba.JPG     # Imagen de prueba
├── README.md
└── .gitignore
```

## Instalación local

Requisitos: **Python 3.10+**.

```bash
# 1. Clonar el repositorio
git clone <tu-repo-url>
cd Deployment-Mobile-Yolo-main

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade ultralytics fastapi uvicorn easyocr opencv-python-headless pillow numpy python-multipart

# 4. Ejecutar el servidor
python app.py
```

El servidor queda disponible en `http://localhost:8080`.

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Página web (subir o capturar foto desde el celular). |
| `GET` | `/health` | Verifica que el servicio esté activo (`{"status":"ok"}`). |
| `POST` | `/predict/` | Envía imagen como `multipart/form-data` (campo `file`). |
| `POST` | `/predict_json/` | Envía imagen en Base64 dentro de un JSON (usado por la web). |

### Respuesta (ambos endpoints)

```json
{
  "success": true,
  "placas": ["ABC123"],
  "num_placas": 1,
  "image": "<imagen con detecciones en Base64>",
  "message": "OK"
}
```

### Ejemplos de uso

**cURL (multipart):**

```bash
curl -X POST -F "file=@carroprueba.JPG" http://<IP>:8080/predict/
```

**cURL (JSON con Base64):**

```bash
curl -X POST http://<IP>:8080/predict_json/ \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<tu_base64>"}'
```

**Python:**

```python
import requests, base64

b64 = base64.b64encode(open("carroprueba.JPG", "rb").read()).decode()
r = requests.post("http://<IP>:8080/predict_json/", json={"image_base64": b64})
print(r.json()["placas"])   # -> ["ABC123"]
```

## Configuración (variables de entorno)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | Ruta del modelo YOLO. |
| `OCR_LANGS` | `en` | Idiomas de EasyOCR (separados por coma). |
| `CONF_THRESH` | `0.25` | Umbral de confianza de detección. |
| `MAX_DIM` | `1280` | Tamaño máximo de imagen antes de la inferencia. |
| `PORT` | `8080` | Puerto del servidor. |

## Despliegue en AWS EC2

> Resumen del procedimiento probado en una instancia Ubuntu `t3.micro` con 4 GiB de swap.

1. **Lanzar una instancia Ubuntu** (recomendado: mínimo `t3.small`/`t2.medium`; `t3.micro` funciona para uso ligero). Abrir el puerto `8080` en el grupo de seguridad.
2. **Conectar por SSH** y preparar el sistema:

   ```bash
   sudo apt update
   sudo apt install -y libgl1 libglib2.0-0 python3-pip python3-venv
   ```

3. **Subir los archivos** (`app.py`, `modelo/best.pt`):

   ```bash
   scp -i "tu_llave.pem" app.py modelo/best.pt ubuntu@<IP>:/home/ubuntu/proyecto/
   ```

4. **Instalar dependencias en la instancia:**

   ```bash
   cd /home/ubuntu/proyecto
   python3 -m venv venv
   source venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install ultralytics fastapi uvicorn easyocr opencv-python-headless pillow numpy python-multipart
   ```

5. **(Opcional) Servicio con systemd** para arranque automático (`/etc/systemd/system/placas.service`):

   ```ini
   [Unit]
   Description=FastAPI YOLO Placas
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/proyecto
   Environment=YOLO_CONFIG_DIR=/home/ubuntu/proyecto/.ultralytics
   ExecStart=/home/ubuntu/proyecto/venv/bin/python3 /home/ubuntu/proyecto/app.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now placas
   ```

6. **Abrir la web** desde cualquier dispositivo: `http://<IP>:8080`.

## Notas de rendimiento

- El frontend redimensiona la foto a `1280px` antes de subirla (más rápido y liviano).
- El servidor redimensiona la imagen recibida a `MAX_DIM=1280` antes de la inferencia.
- Limitar hilos de CPU (`OMP_NUM_THREADS=2`) y usar `imgsz=640` reducen el uso de memoria en instancias pequeñas.
- Para volumen de uso alto se recomienda una instancia con **2 GiB+ de RAM** (p. ej. `t3.small`).

## Prueba rápida

Con la imagen incluida `carroprueba.JPG` se detecta la placa `JNU540`:

<img width="737" height="1600" alt="image" src="https://github.com/user-attachments/assets/b18a0d74-2817-4b07-93f9-ce43c6fd9b0e" />
<img width="737" height="1600" alt="image" src="https://github.com/user-attachments/assets/b0882a83-c41c-4b76-a548-6bcbfc0c4d9e" />
<img width="737" height="1600" alt="image" src="https://github.com/user-attachments/assets/f4da4a89-ee59-465a-82ab-f752bc851d3f" />




```bash
curl -X POST -F "file=@carroprueba.JPG" http://localhost:8080/predict/
```
