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
    # 1) Leer la imagen
    data   = await file.read()
    pil    = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)
    H, W   = img_np.shape[:2]

    # 2) Detectar con YOLOv8
    results     = model(img_np, conf=0.5)[0]
    boxes       = results.boxes.xyxy.cpu().numpy().astype(int)  # Nx4
    confidences = results.boxes.conf.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)

    # 3) Crear fondo negro y dibujar bounding boxes
    black      = np.zeros_like(img_np)
    detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
    pil_boxes  = sv.BoxAnnotator().annotate(scene=Image.fromarray(black), detections=detections)
    composed   = np.array(pil_boxes)

    # 4) Parámetros para segmentación
    pad    = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # 5) Para cada caja: segmentar semilla, pegar y medir diámetro máximo
    for (x1, y1, x2, y2) in boxes:
        # 5.1) ROI con padding
        x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
        x2p, y2p = min(x2 + pad, W),   min(y2 + pad, H)
        roi       = img_np[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            continue

        # 5.2) Enmascarar fuera del box
        mask_box = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, thickness=-1)
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

        # 5.4) Convex hull + morfología para refinar la máscara
        hull = cv2.convexHull(cnt)
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, thickness=-1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # 5.5) Aplica la máscara refinada al ROI en color
        seed_color = cv2.bitwise_and(roi_masked, roi_masked, mask=seed_mask)

        # 5.6) Pega la semilla segmentada en 'composed'
        cx1, cy1 = x1 - x1p, y1 - y1p
        cx2, cy2 = cx1 + (x2 - x1), cy1 + (y2 - y1)
        composed[y1:y2, x1:x2] = seed_color[cy1:cy2, cx1:cx2]

        # 5.7) Llevar contorno a coordenadas globales y hull global
        cnt_global = cnt + np.array([[x1p, y1p]])
        hull_global = cv2.convexHull(cnt_global)

        # 5.8) Extraer puntos del hull
        pts = hull_global.reshape(-1, 2)

        # 5.9) Encontrar el par de puntos con distancia máxima (diámetro)
        max_d = 0
        p1 = p2 = pts[0]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_d:
                    max_d = d
                    p1, p2 = tuple(pts[i]), tuple(pts[j])

        # 5.10) Dibujar la línea del largo (verde)
        cv2.line(composed, p1, p2, (0,255,0), 2)

        # 5.11) Calcular y dibujar el ancho (azul), eje perpendicular
        v = np.array(p2) - np.array(p1)
        v = v / np.linalg.norm(v)
        perp = np.array([-v[1], v[0]])
        projs = pts.dot(perp)
        idx_min, idx_max = np.argmin(projs), np.argmax(projs)
        q1, q2 = tuple(pts[idx_min]), tuple(pts[idx_max])
        cv2.line(composed, q1, q2, (255,0,0), 2)

    # 6) (Opcional) Añadir etiquetas
    # out = sv.LabelAnnotator().annotate(
    #     scene=Image.fromarray(composed),
    #     detections=detections,
    #     labels=[f"{CLASS_NAMES[cid]} {conf:.2f}" for cid, conf in zip(class_ids, confidences)]
    # )

    # 7) Devolver el resultado como JPEG
    buf = io.BytesIO()
    composed.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")