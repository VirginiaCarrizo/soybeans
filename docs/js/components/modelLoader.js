import * as ort from 'onnxruntime-web';

let session = null;

/**
 * Carga el modelo ONNX y guarda la sesión.
 */
export async function cargarModelo() {
  session = await ort.InferenceSession.create('../assets/models/best.onnx');
  console.log('ONNX cargado. Salidas:', session.outputNames);
}

/**
 * Devuelve la sesión cargada.
 */
export function getSession() {
    return session;
  }