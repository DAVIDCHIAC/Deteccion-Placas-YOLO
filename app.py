#!/usr/bin/env python3
# app.py -- FastAPI + YOLOv8 + EasyOCR para detección de placas
# Requiere: fastapi uvicorn ultralytics easyocr opencv-python-headless pillow numpy python-multipart

import os
# Limitar hilos para no saturar el CPU (t3.micro) y reducir memoria
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["TORCH_NUM_THREADS"] = "2"
os.environ["YOLO_VERBOSE"] = "false"

import logging
import base64
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import torch

# -------------------------
# Config / Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-plates")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")  # Ruta al modelo YOLO
OCR_LANGS = os.getenv("OCR_LANGS", "en").split(",")
CONF_THRESH = float(os.getenv("CONF_THRESH", 0.25))
RETURN_IMAGE = True  # Devolver imagen con detecciones
MAX_DIM = int(os.getenv("MAX_DIM", 1280))  # Redimensionar antes de YOLO

# -------------------------
# App init
# -------------------------
app = FastAPI(title="YOLOv8 - Detector de Placas (OCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Cargar modelo y OCR
# -------------------------
logger.info("Cargando modelo YOLOv8 desde %s ...", MODEL_PATH)
torch.set_num_threads(2)
model = YOLO(MODEL_PATH)
logger.info("Modelo YOLOv8 cargado correctamente.")

logger.info("Inicializando EasyOCR con idiomas: %s", OCR_LANGS)
reader = easyocr.Reader(OCR_LANGS, gpu=False)
logger.info("EasyOCR listo.")

# -------------------------
# Helpers
# -------------------------
def preprocess_roi(roi_bgr: np.ndarray, mode: str = "otsu") -> np.ndarray:
    """Genera una version mejorada del ROI para OCR (diferentes metodos)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if mode == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    elif mode == "adaptive":
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 8)
        return th
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    else:
        return gray


def deskew(roi_rgb: np.ndarray) -> np.ndarray:
    """Corrige la inclinacion del ROI si es posible."""
    try:
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = cv2.findNonZero(gray)
        if coords is None or len(coords) < 10:
            return roi_rgb
        rect = cv2.minAreaRect(coords)
        angle = rect[2]
        if angle > 45:
            angle = angle - 90
        if abs(angle) < 1.0 or abs(angle) > 15:
            return roi_rgb
        (h, w) = roi_rgb.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(roi_rgb, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return roi_rgb


def ocr_read_text_from_roi(roi_bgr: np.ndarray) -> Optional[str]:
    """Ejecuta EasyOCR sobre varias versiones del ROI y devuelve el mejor texto."""
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_rgb = deskew(roi_rgb)

        candidates_rgb = [roi_rgb]
        for mode in ["otsu", "adaptive", "clahe"]:
            proc = preprocess_roi(roi_bgr, mode)
            proc_rgb = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
            candidates_rgb.append(proc_rgb)

        best_text = None
        best_conf: float = 0.0

        for cand in candidates_rgb:
            result = reader.readtext(cand)
            if not result:
                continue
            candidate = max(result, key=lambda x: x[2])
            conf = float(candidate[2])
            if conf > best_conf:
                best_conf = conf
                best_text = candidate[1]

        if best_text is None:
            return None

        text = "".join(ch for ch in best_text if ch.isalnum())
        return text.upper() if text else None
    except Exception as e:
        logger.exception("OCR error: %s", e)
        return None


def image_to_base64_jpg(img_bgr: np.ndarray) -> str:
    """Convierte imagen BGR a base64 (JPG) redimensionada para respuestas livianas."""
    h, w = img_bgr.shape[:2]
    max_w = 1280
    if w > max_w:
        scale = max_w / float(w)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (max_w, new_h), interpolation=cv2.INTER_LINEAR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buffer).decode('utf-8')


def process_frame(frame: np.ndarray) -> dict:
    """Ejecuta YOLO + OCR sobre un frame y devuelve el resultado."""
    if max(frame.shape[:2]) > MAX_DIM:
        scale = MAX_DIM / float(max(frame.shape[:2]))
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        logger.info("Imagen redimensionada para YOLO a max dim %s", MAX_DIM)
    results = model.predict(source=frame, conf=CONF_THRESH, imgsz=640, verbose=False)
    if not results:
        return {"placas": [], "image": None, "success": True, "message": "Sin detecciones"}

    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.array([])
    confs = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
    clss = r.boxes.cls.cpu().numpy() if len(r.boxes) > 0 else np.array([])

    placas_detectadas: List[str] = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(clss[i]) if len(clss) > i else None
        label = model.names[cls_id] if cls_id is not None and cls_id < len(model.names) else "objeto"
        conf = confs[i] if len(confs) > i else 0

        h, w = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        roi = frame[y1c:y2c, x1c:x2c].copy()

        # OCR solo si el label coincide con "placa"/"plate"/"license"
        if any(k in label.lower() for k in ["placa", "plate", "license"]):
            text_detected = ocr_read_text_from_roi(roi)
            if text_detected:
                placas_detectadas.append(text_detected)
                cv2.putText(frame, text_detected, (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    img_b64 = image_to_base64_jpg(frame) if RETURN_IMAGE else None
    return {
        "success": True,
        "placas": placas_detectadas,
        "num_placas": len(placas_detectadas),
        "image": img_b64,
        "message": "OK" if placas_detectadas else "No se detectaron placas"
    }


# -------------------------
# Pagina web
# -------------------------
@ app.get("/", response_class=HTMLResponse)
def web():
    return HTMLResponse(WEB_HTML)


WEB_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Detector de Placas</title>
<style>
  :root { --azul:#007AFF; --verde:#34C759; --rojo:#FF3B30; --gris:#f5f5f5; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--gris); color: #222; }
  header { background: var(--azul); color: #fff; padding: 16px; text-align: center;
           box-shadow: 0 2px 8px rgba(0,0,0,.15); }
  header h1 { font-size: 20px; }
  header p { font-size: 13px; opacity: .9; margin-top: 4px; }
  .container { max-width: 480px; margin: 0 auto; padding: 16px; }
  .panel { background: #fff; border-radius: 14px; padding: 16px; margin-bottom: 16px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .panel-title { font-size: 14px; font-weight: 700; color: #555; margin-bottom: 12px;
                 text-transform: uppercase; letter-spacing: .5px; }
  .btn { display: block; width: 100%; padding: 14px; border: none; border-radius: 12px;
         font-size: 16px; font-weight: 600; color: #fff; cursor: pointer; text-align: center; }
  .btn-primary { background: var(--azul); }
  .btn-primary:disabled { background: #8eb8e8; cursor: not-allowed; }
  .btn-green { background: var(--verde); }
  .btn-hidden input { display: none; }
  #preview { width: 100%; border-radius: 10px; margin: 12px 0; display: none; }
  .spinner { display: none; text-align: center; padding: 20px 0; }
  .spinner.active { display: block; }
  .loader { border: 4px solid #e0e0e0; border-top: 4px solid var(--azul); border-radius: 50%;
            width: 42px; height: 42px; animation: spin 1s linear infinite; margin: 0 auto 12px; }
  @keyframes spin { 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }
  .result-box { background: #f0f7ff; border: 1px solid #cfe5ff; border-radius: 10px;
                padding: 14px; margin-top: 12px; display: none; }
  .plate { font-size: 26px; font-weight: 800; color: var(--azul); letter-spacing: 3px;
           text-align: center; margin: 8px 0; }
  .plate.speak { color: var(--verde); }
  .status { font-size: 14px; color: #666; margin-top: 8px; text-align: center; }
  .img-result { width: 100%; border-radius: 10px; margin-top: 12px; display: none; }
  .hint { font-size: 13px; color: #777; text-align: center; margin-top: 8px; line-height: 1.4; }
  .btn-sound { background: var(--verde); margin-top: 10px; display: none; }
  .btn-new { background: #8e8e93; margin-top: 10px; }
  footer { text-align: center; font-size: 12px; color: #999; padding: 20px 0; }
</style>
</head>
<body>
<header>
  <h1>🚗 Detector de Placas</h1>
  <p>YOLOv8 + OCR - Lectura de patentes vehiculares</p>
</header>

<div class="container">
  <div class="panel" id="panelCapture">
    <div class="panel-title">📷 Capturar</div>
    <button class="btn btn-primary" onclick="document.getElementById('fileInput').click()">
      Abrir Camara
    </button>
    <input type="file" id="fileInput" accept="image/*" capture="environment"
           style="display:none">
    <img id="preview" alt="Preview">
    <div class="hint">Toma una foto de la placa del vehiculo (frente o trasera).</div>
  </div>

  <div class="spinner" id="loading">
    <div class="loader"></div>
    <div>Procesando imagen...</div>
  </div>

  <div class="panel" id="panelResult" style="display:none">
    <div class="panel-title">✅ Resultado</div>
    <div class="result-box" id="resultBox">
      <div id="platesBox"></div>
      <div class="status" id="statusMsg"></div>
      <img class="img-result" id="imgResult" alt="Imagen procesada">
      <button class="btn btn-sound" id="btnSound" onclick="speakPlates()">🔊 Escuchar</button>
    </div>
    <button class="btn btn-new" onclick="resetApp()">📸 Tomar otra foto</button>
  </div>

  <footer>Detector de Placas Vehiculares v1.0</footer>
</div>

<script>
  const fileInput = document.getElementById('fileInput');
  const preview = document.getElementById('preview');
  const loading = document.getElementById('loading');
  const panelCapture = document.getElementById('panelCapture');
  const panelResult = document.getElementById('panelResult');
  const resultBox = document.getElementById('resultBox');
  const platesBox = document.getElementById('platesBox');
  const statusMsg = document.getElementById('statusMsg');
  const imgResult = document.getElementById('imgResult');
  const btnSound = document.getElementById('btnSound');
  let currentPlates = [];

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const resizedData = await downscaleImage(file);
    const reader = new FileReader();
    reader.onload = async () => {
      preview.src = reader.result;
      preview.style.display = 'block';
      await sendToServer(reader.result);
    };
    reader.readAsDataURL(resizedData);
  });

  function downscaleImage(file, maxDim = 1280) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > maxDim || height > maxDim) {
          const scale = maxDim / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('no blob')),
                      'image/jpeg', 0.75);
      };
      img.onerror = () => reject(new Error('no image'));
      img.src = URL.createObjectURL(file);
    });
  }

  async function sendToServer(dataUrl) {
    panelCapture.style.display = 'none';
    loading.classList.add('active');
    const base64 = dataUrl.split(',')[1];

    try {
      const res = await fetch('/predict_json/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: base64 })
      });

      let data;
      try {
        data = await res.json();
      } catch (e) {
        data = { error: 'Respuesta del servidor invalida (HTTP ' + res.status + ')' };
      }

      loading.classList.remove('active');

      if (!res.ok && !data.error) {
        data.error = 'Servidor respondio HTTP ' + res.status;
      }

      if (data.error) {
        showResult(false, [data.error]);
        return;
      }

      showResult(true, data.placas || [], data.image || null);
    } catch (err) {
      loading.classList.remove('active');
      showResult(false, ['Error de conexion con el servidor.']);
    }
  }

  function showResult(ok, placas, imgB64) {
    currentPlates = placas;
    resultBox.style.display = 'block';
    platesBox.innerHTML = '';

    if (ok && placas.length > 0) {
      placas.forEach((p, i) => {
        const div = document.createElement('div');
        div.className = 'plate' + (i === 0 ? ' speak' : '');
        div.textContent = p;
        platesBox.appendChild(div);
      });
      statusMsg.textContent = placas.length === 1
        ? 'Placa detectada correctamente.'
        : `Se detectaron ${placas.length} placas.`;
      btnSound.style.display = 'block';
    } else if (ok) {
      statusMsg.textContent = 'No se detectaron placas. Intenta de nuevo con mejor angulo e iluminacion.';
      btnSound.style.display = 'none';
    } else {
      statusMsg.textContent = placas[0] || 'Hubo un error.';
      btnSound.style.display = 'none';
    }

    if (imgB64) {
      imgResult.src = 'data:image/jpeg;base64,' + imgB64;
      imgResult.style.display = 'block';
    }

    panelResult.style.display = 'block';
    panelResult.scrollIntoView({ behavior: 'smooth' });
    if (ok && placas.length > 0) speakPlates();
  }

  function speakPlates() {
    if (currentPlates.length === 0) return;
    if (!('speechSynthesis' in window)) return;
    const text = currentPlates.length === 1
      ? 'La placa detectada es ' + currentPlates[0].split('').join(' ')
      : 'Se detectaron ' + currentPlates.length + ' placas: ' + currentPlates.join(', ');
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'es-ES';
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
  }

  function resetApp() {
    currentPlates = [];
    fileInput.value = '';
    preview.style.display = 'none';
    panelResult.style.display = 'none';
    panelCapture.style.display = 'block';
    loading.classList.remove('active');
    resultBox.style.display = 'none';
    imgResult.style.display = 'none';
    platesBox.innerHTML = '';
  }
</script>
</body>
</html>
"""


# -------------------------
# Rutas API
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/")
async def predict(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Recibe una imagen (multipart o base64) y devuelve:
    {
        "success": True,
        "placas": ["ABC123", "XYZ987"],
        "num_placas": 2,
        "image": "...",  # base64 de la imagen procesada
        "message": "OK"
    }
    """
    try:
        logger.info("Peticion recibida en /predict/")

        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
        elif image_base64:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            image_base64 = image_base64.strip()
            try:
                img_data = base64.b64decode(image_base64 + "===")
            except Exception as e:
                logger.error("Base64 invalido: %s", e)
                return {"error": "Base64 invalido o corrupto."}
            nparr = np.frombuffer(img_data, np.uint8)
        else:
            return {"error": "No se recibio ninguna imagen"}

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        logger.info("Procesando imagen con YOLOv8...")
        result = process_frame(frame)
        logger.info("Placas detectadas: %s", result["placas"])
        return result

    except Exception as e:
        logger.exception("Error en /predict/: %s", e)
        return {"error": str(e)}


@app.post("/predict_json/")
async def predict_json(request: Request):
    """Permite enviar imagen como JSON con campo 'image_base64'."""
    try:
        logger.info("Peticion JSON recibida en /predict_json/")
        body = await request.json()
        image_base64 = body.get("image_base64")
        if not image_base64:
            return {"error": "No se recibio ninguna imagen"}

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",")[1]
        image_base64 = image_base64.strip()
        img_data = base64.b64decode(image_base64 + "===")
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "No se pudo decodificar la imagen"}

        logger.info("Procesando imagen JSON con YOLOv8...")
        result = process_frame(frame)
        logger.info("Placas detectadas: %s", result["placas"])
        return result

    except Exception as e:
        logger.exception("Error en /predict_json/: %s", e)
        return {"error": str(e)}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info("Iniciando servidor en 0.0.0.0:%s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)