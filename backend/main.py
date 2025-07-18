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

# 2. Carga el modelo
model = YOLO(MODEL_PATH)

# 3. Nombres de clase
CLASS_NAMES = model.model.names  # dict {id: nombre}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 4. Lee la imagen enviada
    data = await file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)

    # 5. Inferencia
    results = model(img_np, conf=0.5)[0]

    # 6. Extrae cajas, clases y confianzas
    boxes       = results.boxes.xyxy.cpu().numpy().astype(int)  # Nx4
    confidences = results.boxes.conf.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy().astype(int)

    # 7. Crea máscara: píxeles dentro de cajas = 1, fuera = 0
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)

    # 8. Aplica máscara: fuera de cajas → negro
    masked_np = np.where(mask[..., None] == 1, img_np, 0).astype(np.uint8)
    pil_masked = Image.fromarray(masked_np)

    # 9. Prepara anotaciones con supervision
    detections = sv.Detections(xyxy=boxes,
                               confidence=confidences,
                               class_id=class_ids)
    labels = [
        f"{CLASS_NAMES[cid]} {conf:.2f}"
        for cid, conf in zip(class_ids, confidences)
    ]

    box_annot   = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()

    out = box_annot.annotate(scene=pil_masked.copy(), detections=detections)
    out = label_annot.annotate(scene=out, detections=detections, labels=labels)

    # 10. Devuelve la imagen JPEG resultante
    buf = io.BytesIO()
    out.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")