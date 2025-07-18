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
    # 1) Lee la imagen
    data = await file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)
    H, W = img_np.shape[:2]

    # 2) Detecta con YOLO
    results = model(img_np, conf=0.5)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    # 3) Máscara global: fuera de cajas → negro
    mask_boxes = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask_boxes, (x1, y1), (x2, y2), 1, thickness=-1)
    img_masked = np.where(mask_boxes[..., None] == 1, img_np, 0).astype(np.uint8)

    # 4) Segmenta cada semilla (padding + hull + morfología)
    final = np.zeros_like(img_masked)
    pad = 1  # margen extra
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for (x1, y1, x2, y2) in boxes:
        x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
        x2p, y2p = min(x2 + pad, W), min(y2 + pad, H)
        roi = img_masked[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue

        # convierte a gris y binariza
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # encuentra el contorno más grande y hull
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))

        # crea y suaviza la máscara
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, thickness=-1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # aplica la máscara al ROI color
        seed_color = cv2.bitwise_and(roi, roi, mask=seed_mask)

        # recorta sin padding y coloca en final
        cx1, cy1 = x1 - x1p, y1 - y1p
        cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
        final[y1:y2, x1:x2] = seed_color[cy1:cy2, cx1:cx2]

    # 5) Dibuja los boxes sobre la imagen enmascarada
    scene_masked = Image.fromarray(img_masked)
    pil_boxes = sv.BoxAnnotator().annotate(
        scene=scene_masked.copy(),
        detections=sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
    )

    # 6) Convierte pil_boxes a array y superpone semillas sobre cajas
    boxes_np = np.array(pil_boxes)
    mask_seed = final.any(axis=-1)[..., None]
    # Compose: si mask_seed=1 usa final, si no usa boxes_np
    composed = np.where(mask_seed, final, boxes_np)

    # 7) Añade etiquetas
    labels = [f"{CLASS_NAMES[cid]} {conf:.2f}" for cid, conf in zip(class_ids, confidences)]
    out = sv.LabelAnnotator().annotate(
        scene=Image.fromarray(composed),
        detections=sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids),
        labels=labels
    )

    # 8) Devuelve JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
