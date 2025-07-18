import { canvas, ctx, TARGET_SIZE } from './dom.js';
import { inferirLocal } from './inference.js';

const fileInput = document.getElementById('fileInput');

fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;

  // Lee la imagen y convierte a DataURL
  const reader = new FileReader();
  reader.onload = () => {
    // Cuando esté listo, llama a inferirLocal
    inferirLocal(reader.result);
  };
  reader.readAsDataURL(file);
});
