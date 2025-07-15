import { video, canvas, ctx, boton, TARGET_SIZE } from './dom.js';
import { inferirLocal } from './inference.js';

// 1) Pedir permiso y mostrar cámara
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
  } catch (err) {
    console.error('No se pudo acceder a la cámara:', err);
  }
}

// 2) Capturar un frame y procesar
function procesarFrame() {
  // Ajustar canvas
  canvas.width = canvas.height = TARGET_SIZE;
  // Dibujar el frame actual del video
  ctx.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);
  // Obtener DataURL y enviar a inferirLocal
  const dataURL = canvas.toDataURL('image/jpeg');
  inferirLocal(dataURL);
}

// Cuando se cargue el script arrancamos la cámara
initCamera();

// Asociamos el botón
boton.addEventListener('click', procesarFrame);
