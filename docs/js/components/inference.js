import { canvas, ctx, TARGET_SIZE } from './dom.js';

function dataURLtoBlob(dataurl) {
  const [meta, b64] = dataurl.split(',');
  const mime = meta.match(/:(.*?);/)[1];
  const bin  = atob(b64);
  const arr  = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    arr[i] = bin.charCodeAt(i);
  }
  return new Blob([arr], { type: mime });
}

const API_BASE = ''; // si sirves tu API local con FastAPI+Uvicorn en mismo host, deja ''. 
                   // Si la sirves en otro host/puerto, pon 'http://192.168.x.x:8000'

export async function inferirLocal(dataURL) {
  if (!dataURL) return;

  const blob = dataURLtoBlob(dataURL);
  const form = new FormData();
  form.append('file', blob, 'input.jpg');

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: form
    });
    console.log('[DEBUG] status:', res.status, res.statusText);

    if (!res.ok) {
      const text = await res.text();
      console.error('Error de inferencia:', res.status, res.statusText, text);
      return;
    }

    const imgBlob = await res.blob();
    const img     = new Image();
    img.onload = () => {
      // ajusta el canvas al tamaño natural de la imagen si quieres:
      canvas.width  = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    };
    img.src = URL.createObjectURL(imgBlob);

  } catch (err) {
    console.error('Fetch failed:', err);
  }
}
