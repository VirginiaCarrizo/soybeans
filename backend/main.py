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

    # 3) Fondo negro y dibujar cajas
    black = np.zeros_like(img_np)
    dets  = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
    pil_boxes = sv.BoxAnnotator().annotate(scene=Image.fromarray(black), detections=dets)
    composed  = np.array(pil_boxes)

    # 4) Segmentar cada semilla y medir
    pad    = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    # Vamos a superponer cada semilla segmentada + medidas sobre 'composed'
    for (x1,y1,x2,y2) in boxes:
        # + padding
        x1p, y1p = max(x1-pad,0), max(y1-pad,0)
        x2p, y2p = min(x2+pad,W),   min(y2+pad,H)
        roi       = img_np[y1p:y2p, x1p:x2p]
        if roi.size == 0: continue

        # enmascara fuera del box
        mask_box = np.zeros((H,W), np.uint8)
        cv2.rectangle(mask_box, (x1,y1),(x2,y2), 1, -1)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask_box[y1p:y2p,x1p:x2p])

        # segmentación
        gray = cv2.cvtColor(roi_masked, cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        cnt = max(cnts, key=cv2.contourArea)

        hull = cv2.convexHull(cnt)
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, -1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        seed_color = cv2.bitwise_and(roi_masked, roi_masked, mask=seed_mask)

        # recortar al tamaño sin padding y pegar sobre composed
        cx1, cy1 = x1-x1p, y1-y1p
        cx2, cy2 = cx1+(x2-x1), cy1+(y2-y1)
        composed[y1:y2, x1:x2] = seed_color[cy1:cy2, cx1:cx2]

         # 5) Creamos la imagen compuesta partiendo de fondo negro + boxes
    black = np.zeros_like(img_np)
    dets  = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
    scene0  = sv.BoxAnnotator().annotate(scene=Image.fromarray(black.copy()), detections=dets)
    composed = np.array(scene0)

    # 6) Para cada semilla, pegamos su color y dibujamos sus propias líneas
    for (x1,y1,x2,y2) in boxes:
        # --- segmentación idéntica a antes, obtenemos 'seed_color' y 'seed_mask' ---
        x1p, y1p = max(x1-pad,0), max(y1-pad,0)
        x2p, y2p = min(x2+pad,W),   min(y2+pad,H)
        roi       = img_np[y1p:y2p, x1p:x2p]
        mask_box  = np.zeros((H,W), np.uint8)
        cv2.rectangle(mask_box,(x1,y1),(x2,y2),1,-1)
        roi_mask  = cv2.bitwise_and(roi, roi, mask=mask_box[y1p:y2p,x1p:x2p])
        gray      = cv2.cvtColor(roi_mask, cv2.COLOR_RGB2GRAY)
        _, bw     = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cnts,_    = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            continue
        hull      = cv2.convexHull(max(cnts, key=cv2.contourArea))
        seed_mask = np.zeros_like(gray)
        cv2.drawContours(seed_mask, [hull], -1, 255, -1)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        seed_col  = cv2.bitwise_and(roi_mask, roi_mask, mask=seed_mask)

        # pegamos la semilla segmentada
        cx1,cy1 = x1-x1p, y1-y1p
        cx2,cy2 = cx1+(x2-x1), cy1+(y2-y1)
        composed[y1:y2, x1:x2] = seed_col[cy1:cy2, cx1:cx2]

        # calculamos la caja exacta de la semilla dentro del ROI padded
        ys, xs = np.where(seed_mask==255)
        if ys.size==0:
            continue
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        # convertimos a coords globales
        gx1, gy1 = x1p + x_min, y1p + y_min
        gx2, gy2 = x1p + x_max, y1p + y_max

        # dibujamos las líneas de ancho y alto (verde)
        mid_y = gy1 + (gy2-gy1)//2
        mid_x = gx1 + (gx2-gx1)//2

        # ancho: de (gx1,mid_y) a (gx2,mid_y)
        cv2.line(composed, (gx1, mid_y), (gx2, mid_y), (0,255,0), 2)
        # alto: de (mid_x,gy1) a (mid_x,gy2)
        cv2.line(composed, (mid_x, gy1), (mid_x, gy2), (0,255,0), 2)

    #7) (opcional) añade etiquetas
    out = sv.LabelAnnotator().annotate(
        scene=Image.fromarray(composed),
        detections=dets,
        labels=[f"{CLASS_NAMES[cid]} {conf:.2f}" for cid,conf in zip(class_ids,confidences)]
    )

    # 8) devuelve
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")