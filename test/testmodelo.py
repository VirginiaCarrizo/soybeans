#!/usr/bin/env python3
"""
test_yolov8.py

Carga un modelo YOLOv8 desde best.pt, corre inferencia sobre una imagen
y guarda una versión anotada con los bounding boxes y las clases.
"""

import sys
import cv2
from ultralytics import YOLO

def main(model_path: str, img_path: str, out_path: str, imgsz: int = 640,
         conf_thresh: float = 0.90, iou_thresh: float = 0.3):
    # 1) Carga el modelo (ultralytics>=8.0)
    model = YOLO(model_path)

    # 2) Leer imagen de disco
    orig = cv2.imread(img_path)
    if orig is None:
        print(f"ERROR: no se pudo leer la imagen '{img_path}'")
        sys.exit(1)

    # 3) Redimensionar al tamaño de entrada del modelo (sin preservar relación)
    img = cv2.resize(orig, (imgsz, imgsz))

    # 3) Ejecutar inferencia
    #    Devuelve lista de Results; tomamos la primera
    results = model(img, conf=conf_thresh, iou=iou_thresh)[0]

    # 4) Pintar las cajas sobre la imagen
    annotated = results.plot()  # devuelve un numpy array con las anotaciones

    # 5) Guardar resultado
    # cv2.imwrite(out_path, annotated)
    cv2.imshow("rdo", annotated)
    cv2.waitKey(0)

    print(f"Imagen anotada guardada en {out_path}")



model_file = r"D:\proyectos\soybeans\best.pt"
input_img  = r"C:\Users\virginia.carrizo\Downloads\WhatsApp Image 2025-06-03 at 3.25.35 PM.jpeg"
output_img = r"D:\proyectos\soybeans/annotated.jpg"
main(model_file, input_img, output_img)
