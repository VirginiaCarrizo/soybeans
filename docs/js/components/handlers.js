import { video, canvas, ctx, boton, TARGET_SIZE } from './dom.js';
import { inferirLocal } from './inference.js';

async function initCamera() {
  try {
    const constraints = {
      video: { facingMode: { exact: "environment" } }
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.warn("No pude abrir trasera, usando por defecto:", err);
    const fallback = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = fallback;
    await video.play();
  }
}

function procesarFrame() {
  const off = document.createElement('canvas');
  off.width = off.height = TARGET_SIZE;
  const offCtx = off.getContext('2d');
  offCtx.drawImage(video, 0, 0, TARGET_SIZE, TARGET_SIZE);
  const dataURL = off.toDataURL('image/jpeg');
  inferirLocal(dataURL);
}

// Arranca la cámara al cargar la página
initCamera();

// Configura el botón de procesar
boton.addEventListener('click', procesarFrame);

// ————————
// Aquí añades el listener para el <input type="file">
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => inferirLocal(reader.result);
  reader.readAsDataURL(file);
});
