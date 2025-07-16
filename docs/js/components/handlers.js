import { video, canvas, ctx, boton, TARGET_SIZE } from './dom.js';
import { inferirLocal } from './inference.js';

async function initCamera() {
  try {
    const constraints = {
      video: {
        // Fuerza la cámara trasera (“environment”)
        facingMode: { exact: "environment" }
        // Si exact falla en algún dispositivo, podrías usar:
//      facingMode: { ideal: "environment" }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.warn("No pude abrir la cámara trasera, usando la por defecto:", err);
    // Fallback a la cámara por defecto si “environment” no está disponible
    const fallback = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = fallback;
    await video.play();
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
