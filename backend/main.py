from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

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

# 2. Carga el modelo YOLOv8
model = YOLO(MODEL_PATH)

# 3. Extrae nombres de clase desde el propio modelo
CLASS_NAMES = model.model.names  # es un dict {id: nombre}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 4. Lee imagen de la petición
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(img)

    # 5. Inferencia con YOLO
    #    conf=0.5 para umbral de confianza
    results = model(img_np, conf=0.5)[0]

    # 6. Extrae cajas, clases y confianzas
    boxes       = results.boxes.xyxy.cpu().numpy()      # [[x1,y1,x2,y2], ...]
    confidences = results.boxes.conf.cpu().numpy()      # [0.87, 0.75, ...]
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)  # [0, 1, 2, ...]

    # 7. Convierte a Detections de supervision
    detections = sv.Detections(
        xyxy       = boxes,
        confidence = confidences,
        class_id   = class_ids
    )

    # 8. Anotación de cajas y etiquetas
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}"
        for cid, conf in zip(class_ids, confidences)
    ]

    out = box_annotator.annotate(scene=img.copy(), detections=detections)
    out = label_annotator.annotate(scene=out, detections=detections, labels=labels)

    # 9. Devuelve JPEG
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
