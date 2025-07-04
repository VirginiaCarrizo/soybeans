import { getSession, TARGET_SIZE } from './modelLoader.js';
import { nonMaxSuppression } from './nms.js';
import { dibujarPredicciones } from './grouping.js';

/**
 * Realiza inferencia ONNX y dibuja resultados.
 * @param {string} lastDataURL - Data URL de la imagen cargada
 */
export async function inferirLocal(lastDataURL) {
  const session = getSession();
  if (!session || !lastDataURL) return;

  const { canvas, ctx } = require('./dom.js');
  // 1) Leer píxeles del canvas
  const imgData = ctx.getImageData(0,0,TARGET_SIZE,TARGET_SIZE).data;
  const input   = new Float32Array(TARGET_SIZE*TARGET_SIZE*3);
  for (let i=0, j=0; i<imgData.length; i+=4, j+=3) {
    input[j]   = imgData[i]   / 255;
    input[j+1] = imgData[i+1] / 255;
    input[j+2] = imgData[i+2] / 255;
  }
  const tensor = new ort.Tensor('float32', input, [1,3,TARGET_SIZE,TARGET_SIZE]);

  // 2) Ejecutar sesión ONNX
  const outputs = await session.run({ images: tensor });

  // 3) Procesar salidas y aplicar NMS
  // ... lógica de parsing de tensor y umbrales ...

  // 4) Dibujar predicciones
  dibujarPredicciones(finalPreds);
}