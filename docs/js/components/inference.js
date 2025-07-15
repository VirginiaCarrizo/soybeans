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
  console.log('[DEBUG] Enviando blob al servidor:', blob);
  console.log('   size:', blob.size, 'type:', blob.type);
  const form = new FormData();
  form.append('file', blob, 'input.jpg');

  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: form
  });
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
}
