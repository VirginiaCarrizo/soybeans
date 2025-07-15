import { video, canvas, ctx, boton, TARGET_SIZE } from './dom.js';
import { inferirLocal } from './inference.js';

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error('No se pudo acceder a la cámara:', err);
  }
}

function procesarFrame() {
  // 1. Creamos un canvas “offscreen” para capturar el frame
  const off = document.createElement('canvas');
  off.width = off.height = TARGET_SIZE;
  const offCtx = off.getContext('2d');
  offCtx.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);

  // 2. Sacamos el DataURL de ese offscreen
  const dataURL = off.toDataURL('image/jpeg');

  // 3. Enviamos al backend y dejamos que inferirLocal
  //    limpie y pinte en el canvas visible SOLO la imagen anotada:
  inferirLocal(dataURL);
}

// arranca cámara al cargar
initCamera();

// botón “Procesar” ahora sólo dispara la foto+inferencia
boton.addEventListener('click', procesarFrame);
