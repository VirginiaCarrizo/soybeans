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
    # 1) Leer imagen y numpy
    data   = await file.read()
    pil    = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)
    H, W   = img_np.shape[:2]

    # 2) Detectar con YOLOv8
    results     = model(img_np, conf=0.5)[0]
    boxes       = results.boxes.xyxy.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)

    # 3) Crear imagen negra de fondo
    black = np.zeros_like(img_np)

    # 4) Dibujar los bounding boxes sobre fondo negro
    scene_black = Image.fromarray(black)
    box_annot   = sv.BoxAnnotator()
    dets        = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
    pil_boxes   = box_annot.annotate(scene=scene_black.copy(), detections=dets)
    boxes_np    = np.array(pil_boxes)

    # 5) Segmentar cada semilla con padding + hull + morfología
    final = np.zeros_like(img_np)
    pad    = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    for (x1,y1,x2,y2) in boxes:
        # ampliar caja con padding
        x1p, y1p = max(x1-pad,0), max(y1-pad,0)
        x2p, y2p = min(x2+pad,W),   min(y2+pad,H)

        roi = img_np[y1p:y2p, x1p:x2p]
        if roi.size == 0: continue

        # máscara global de caja (para fondo)
        mask_box = np.zeros((H,W), np.uint8)
        cv2.rectangle(mask_box, (x1,y1),(x2,y2), 1, -1)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask_box[y1p:y2p,x1p:x2p])

        # segmentación interna
        gray = cv2.cvtColor(roi_masked, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue

        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, -1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        seed_color = cv2.bitwise_and(roi_masked, roi_masked, mask=seed_mask)

        # recortar al tamaño original de la caja y pegar
        cx1, cy1 = x1 - x1p, y1 - y1p
        cx2, cy2 = cx1 + (x2-x1), cy1 + (y2-y1)
        final[y1:y2, x1:x2] = seed_color[cy1:cy2, cx1:cx2]

    # 6) Superponer semillas segmentadas (final) sobre los boxes
    mask_seed = (final.sum(axis=-1) > 0)[..., None]  # True donde hay semilla
    composed  = np.where(mask_seed, final, boxes_np)

    # 7) (Opcional) Etiquetas
    labels = [f"{CLASS_NAMES[cid]} {conf:.2f}" for cid, conf in zip(class_ids, confidences)]
    out = sv.LabelAnnotator().annotate(
        scene=Image.fromarray(composed),
        detections=dets,
        labels=labels
    )

    # 8) Retornar JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")