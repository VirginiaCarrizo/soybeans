import { canvas, ctx } from './dom.js';

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

// Ajusta si tu API corre en otra máquina o puerto
const API_BASE = 'http://localhost:8000';

export async function inferirLocal(dataURL) {
  if (!dataURL) return;

  // 1) Muestra el spinner
  const spinner = document.getElementById('spinner');
  spinner.classList.remove('hidden');

  // 2) Prepara la petición
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
      spinner.classList.add('hidden');
      return;
    }

    // 3) Procesa la respuesta y pinta en el canvas
    const imgBlob = await res.blob();
    const img     = new Image();
    img.onload = () => {
      canvas.width  = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      // 4) Oculta el spinner
      spinner.classList.add('hidden');
    };
    img.src = URL.createObjectURL(imgBlob);

  } catch (err) {
    console.error('Fetch failed:', err);
    spinner.classList.add('hidden');
  }
}
