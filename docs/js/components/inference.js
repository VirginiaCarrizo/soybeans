import { canvas, ctx, TARGET_SIZE } from './dom.js';

function dataURLtoBlob(dataurl) {
  const [meta, b64] = dataurl.split(',');
  const mime = meta.match(/:(.*?);/)[1];
  const bin  = atob(b64);
  let   len  = bin.length;
  const arr  = new Uint8Array(len);
  while (len--) arr[len] = bin.charCodeAt(len);
  return new Blob([arr], { type: mime });
}

/**
 * Envia la imagen al backend y pinta la respuesta anotada.
 */
export async function inferirLocal(lastDataURL) {
  if (!lastDataURL) return;

  const blob = dataURLtoBlob(lastDataURL);
  const form = new FormData();
  form.append('file', blob, 'input.jpg');

  try {
    const res = await fetch('http://172.31.90.12:8000/predict', {
      method: 'POST',
      body: form
    });
    console.log('[DEBUG] status:', res.status, res.statusText);
    if (!res.ok) {
      console.error('Error de inferencia:', res.statusText);
      return;
    }

    const imgBlob = await res.blob();
    const img     = new Image();
    img.onload = () => {
      canvas.width  = canvas.height = TARGET_SIZE;
      ctx.clearRect(0,0,TARGET_SIZE,TARGET_SIZE);
      ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);
    };
    img.src = URL.createObjectURL(imgBlob);
  } catch (err) {
    // capturamos errores de red o CORS
    console.error('Fetch failed:', err);
  }
}