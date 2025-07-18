from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2
#PARA CORRER LA APP NECESITO EN DOS TERMINALES DIFERENTES CORRER ESTAS DOS LINEAS:
#python -m http.server 5500
#uvicorn main:app --reload --host 0.0.0.0 --port 8000

# -- nuevo: import YOLO de ultralytics --
from ultralytics import YOLO

import supervision as sv

app = FastAPI()

# --- CONFIGURA CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # o pon aquí el origen de tu frontend, ej. ["http://localhost:5500"]
    allow_methods=["*"],            # permite GET, POST, OPTIONS, etc.
    allow_headers=["*"],            # permite Content-Type, Authorization, etc.
)
# -------------------------

# 1. Apunta al nuevo checkpoint
MODEL_PATH = r"D:\proyectos\soybeans\backend\models\best.pt"

# 2. Carga el modelo
model = YOLO(MODEL_PATH)

# 3. Nombres de clase
CLASS_NAMES = model.model.names  # dict {id: nombre}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1) Lee imagen
    data = await file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)

    # 2) Inferencia de detección
    results = model(img_np, conf=0.5)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)   # Nx4
    confidences = results.boxes.conf.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)

    # 3) Máscara global: fuera de cajas → negro
    mask_boxes = np.zeros(img_np.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask_boxes, (x1, y1), (x2, y2), 1, thickness=-1)
    img_masked = np.where(mask_boxes[..., None] == 1, img_np, 0).astype(np.uint8)

    # 4) Para cada caja, segmenta la semilla dentro y la recorta
    #    Construimos una imagen final donde cada ROI contiene solo la semilla
    final = np.zeros_like(img_masked)
    for (x1, y1, x2, y2) in boxes:
        roi = img_masked[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Gris + Otsu
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Encuentra contornos y elige el mayor
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # Crea máscara de la semilla en este ROI
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [cnt], -1, 255, thickness=-1)

        # Aplica esa máscara al ROI original (mantiene color, fuera semilla → negro)
        seed_color = cv2.bitwise_and(roi, roi, mask=seed_mask)
        final[y1:y2, x1:x2] = seed_color

    # 5) Convierte a PIL para anotación
    pil_final = Image.fromarray(final)

    # 6) Prepara detections y etiquetas (sin área por ahora)
    detections = sv.Detections(xyxy=boxes,
                               confidence=confidences,
                               class_id=class_ids)
    labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}"
        for cid, conf in zip(class_ids, confidences)
    ]

    # 7) Anota sobre la imagen donde solo aparecen las semillas
    out = sv.BoxAnnotator().annotate(scene=pil_final.copy(), detections=detections)
    out = sv.LabelAnnotator().annotate(scene=out,
                                       detections=detections,
                                       labels=labels)

    # 8) Devuelve JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")