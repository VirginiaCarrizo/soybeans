import { inferirLocal } from 'inference.js';
import { fileInput, canvas, ctx, sliderConf, labelConf, sliderOv, labelOv, boton } from 'dom.js';

let lastDataURL = null;

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    lastDataURL = reader.result;
    const img = new Image();
    img.src = lastDataURL;
    img.onload = () => {
      canvas.width = canvas.height = TARGET_SIZE;
      ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);
      inferirLocal(lastDataURL);
    };
  };
  reader.readAsDataURL(file);
});

let tid;
sliderConf.addEventListener('input', () => {
  labelConf.textContent = parseFloat(sliderConf.value).toFixed(2);
  clearTimeout(tid);
  tid = setTimeout(() => inferirLocal(lastDataURL), 200);
});
sliderOv.addEventListener('input', () => {
  labelOv.textContent = parseFloat(sliderOv.value).toFixed(2);
  clearTimeout(tid);
  tid = setTimeout(() => inferirLocal(lastDataURL), 200);
});

boton.addEventListener('click', () => inferirLocal(lastDataURL));