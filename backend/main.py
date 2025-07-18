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
    H, W = img_np.shape[:2]

    # 2) Detecta con YOLO
    results = model(img_np, conf=0.5)[0]
    boxes       = results.boxes.xyxy.cpu().numpy().astype(int)  # Nx4
    confidences = results.boxes.conf.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)

    # 3) Máscara global: fuera de cajas → negro
    mask_boxes = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask_boxes, (x1, y1), (x2, y2), 1, thickness=-1)
    img_masked = np.where(mask_boxes[..., None] == 1, img_np, 0).astype(np.uint8)

    # 4) Segmenta cada semilla con padding + hull + morfología
    final = np.zeros_like(img_masked)
    pad = 20  # píxeles de margen extra
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    for (x1, y1, x2, y2) in boxes:
        # aplica padding y recorta dentro de la imagen
        x1p = max(x1 - pad, 0)
        y1p = max(y1 - pad, 0)
        x2p = min(x2 + pad, W)
        y2p = min(y2 + pad, H)

        roi = img_masked[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue

        # convierte a gris y binariza
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # encuentra el contorno más grande
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # calcula el convex hull
        hull = cv2.convexHull(cnt)
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, thickness=-1)

        # morfología: cierra huecos y elimina ruido
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # aplica la máscara al ROI en color
        seed_color = cv2.bitwise_and(roi, roi, mask=seed_mask)

        # recorta seed_color al tamaño original de la caja y lo coloca en final
        cx1, cy1 = x1 - x1p, y1 - y1p   # offset dentro del ROI padded
        cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
        final[y1:y2, x1:x2] = seed_color[cy1:cy2, cx1:cx2]

    # 5) Convierte a PIL para anotar
    pil_final = Image.fromarray(final)

    # 6) Prepara detections y etiquetas
    detections = sv.Detections(
        xyxy       = boxes,
        confidence = confidences,
        class_id   = class_ids
    )
    labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}"
        for cid, conf in zip(class_ids, confidences)
    ]

    # 7) Dibuja cajas y etiquetas sobre la imagen segmentada
    out = sv.BoxAnnotator().annotate(scene=pil_final.copy(), detections=detections)
    out = sv.LabelAnnotator().annotate(scene=out, detections=detections, labels=labels)

    # 8) Devuelve JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")