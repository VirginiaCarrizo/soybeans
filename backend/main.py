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
    # 1) Leer imagen
    data   = await file.read()
    pil    = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)
    H, W   = img_np.shape[:2]

    # 2) Detectar con YOLO
    results     = model(img_np, conf=0.5)[0]
    boxes       = results.boxes.xyxy.cpu().numpy().astype(int)
    confidences = results.boxes.conf.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)

    # 3) Fondo negro + dibujar boxes
    black      = np.zeros_like(img_np)
    detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
    pil_boxes  = sv.BoxAnnotator().annotate(scene=Image.fromarray(black), detections=detections)
    composed   = np.array(pil_boxes)

    # 4) Parámetros para segmentación
    pad    = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # 5) Por cada caja: segmentar, pegar y medir usando fitEllipse
    for (x1, y1, x2, y2) in boxes:
        # 5.1) ROI padded
        x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
        x2p, y2p = min(x2 + pad, W),   min(y2 + pad, H)
        roi       = img_np[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue

        # 5.2) Enmascarar fuera de box
        mask_box = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, -1)
        roi_masked = cv2.bitwise_and(
            roi,
            roi,
            mask=mask_box[y1p:y2p, x1p:x2p]
        )

        # 5.3) Binarizar y extraer contorno más grande
        gray = cv2.cvtColor(roi_masked, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cnt.shape[0] < 5:
            # fitEllipse requiere al menos 5 puntos
            continue

        # 5.4) Refina máscara con hull y morfología
        hull = cv2.convexHull(cnt)
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, thickness=-1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # 5.5) Aplica máscara para color
        seed_color = cv2.bitwise_and(roi_masked, roi_masked, mask=seed_mask)

        # 5.6) Pega la semilla segmentada sobre composed
        cx1, cy1 = x1 - x1p, y1 - y1p
        cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
        composed[y1:y2, x1:x2] = seed_color[cy1:cy2, cx1:cx2]

        # 5.7) Ajustar coordenadas del contorno al sistema global
        cnt_global = cnt + np.array([[x1p, y1p]])

        # 5.8) Ajustar convex hull global para medición
        hull_global = cv2.convexHull(cnt_global)

        # 5.9) Ajusta unipse mínimo (rotated ellipse) al hull
        ellipse = cv2.fitEllipse(hull_global)
        (cx, cy), (MA, ma), angle = ellipse  # MA=major axis, ma=minor axis

        # 5.10) Calcula puntos finales de la mayor longitud (largo)
        theta = np.deg2rad(angle)
        dx = np.cos(theta) * MA/2
        dy = np.sin(theta) * MA/2
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        cv2.line(composed, p1, p2, (0,255,0), 2)  # largo

        # 5.11) Calcula puntos finales del eje perpendicular (ancho)
        theta_p = theta + np.pi/2
        dx2 = np.cos(theta_p) * ma/2
        dy2 = np.sin(theta_p) * ma/2
        q1 = (int(cx - dx2), int(cy - dy2))
        q2 = (int(cx + dx2), int(cy + dy2))
        cv2.line(composed, q1, q2, (255,0,0), 2)  # ancho (azul)

    # 6) (Opcional) Etiquetas
    out = sv.LabelAnnotator().annotate(
        scene=Image.fromarray(composed),
        detections=detections,
        labels=[f"{CLASS_NAMES[cid]} {conf:.2f}" for cid, conf in zip(class_ids, confidences)]
    )

    # 7) Retornar como JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")