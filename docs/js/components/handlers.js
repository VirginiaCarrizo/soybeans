import { inferirLocal } from './inference.js';

const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => inferirLocal(reader.result);
  reader.readAsDataURL(file);
});
