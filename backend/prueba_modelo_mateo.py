#!/usr/bin/env python3

import argparse
import os
import torch
from torchvision import transforms
from PIL import Image

try:
    # Intenta usar la API de Ultralytics para modelos YOLOv8
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


def load_model(model_path, device):
    """
    Carga el modelo desde un checkpoint .pt.
    - Si está disponible Ultralytics, usar YOLO(model_path).
    - Si no, usar torch.load y manejar distintos formatos de checkpoint.
    """
    if _HAS_ULTRALYTICS:
        model = YOLO(model_path)
        model.to(device)
    else:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                model = ckpt['model']
            elif 'state_dict' in ckpt:
                raise ValueError(
                    'Checkpoint contiene solo state_dict; debes definir la arquitectura antes de cargar.'
                )
            else:
                raise ValueError('Formato de checkpoint no soportado.')
            model.to(device)
        else:
            model = ckpt.to(device)
    model.eval()
    return model


def preprocess_image(image_path, size=(224, 224)):
    """Carga una imagen y aplica transformaciones estándar."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='Prueba de modelo PyTorch (.pt) con una imagen')
    parser.add_argument('model', type=str, help='Ruta al archivo .pt del modelo')
    parser.add_argument('image', type=str, help='Ruta a la imagen de entrada')
    parser.add_argument('--output-dir', type=str, default='', help='Directorio para guardar imágenes con boxes')
    parser.add_argument('--resize', nargs=2, type=int, metavar=('H','W'),
                        help='Cambiar tamaño de imagen antes de pasar al modelo (altura ancho)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)

    # Preparar directorio de salida
    output_dir = args.output_dir or os.path.dirname(args.image)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]

    if _HAS_ULTRALYTICS:
        # Ejecución con YOLOv8
        dims = tuple(args.resize) if args.resize else None
        results = model(args.image)
        # Asegurar lista de resultados
        results_list = results if isinstance(results, list) else [results]
        for idx, r in enumerate(results_list):
            # Dibuja cajas sobre la imagen
            annotated = r.plot()
            out_fname = f"{base}_pred_{idx}.jpg"
            out_path = os.path.join(output_dir, out_fname)
            Image.fromarray(annotated).save(out_path)
            print(f'Guardado: {out_path}')
    else:
        # Inferencia genérica (clasificación u otro)
        size = tuple(args.resize) if args.resize else (224, 224)
        input_tensor = preprocess_image(args.image, size).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            if hasattr(output, 'ndim') and output.ndim in (1, 2):
                probs = torch.nn.functional.softmax(output.squeeze(), dim=0)
                top_prob, top_class = probs.max(0)
                print(f'Clase predicha: {top_class.item()} con probabilidad {top_prob.item():.4f}')
            else:
                print('Salida del modelo:', output)

if __name__ == '__main__':
    main()
