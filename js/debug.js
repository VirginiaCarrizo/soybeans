// ------------------------------
// ELEMENTOS DEL DOM
// ------------------------------
const fileInput   = document.getElementById('fileInput');
const canvas      = document.getElementById('canvas');
const boton       = document.getElementById('botonAccion');
const ctx         = canvas.getContext('2d');

const sliderConf  = document.getElementById('sliderConf');
const labelConf   = document.getElementById('labelConf');
const sliderOv    = document.getElementById('sliderOv');
const labelOv     = document.getElementById('labelOv');

const TARGET_SIZE = 640;
let lastDataURL   = null;

// ------------------------------
// ONNX Runtime Web: carga el modelo
// ------------------------------
let session = null;
async function cargarModelo() {
  session = await ort.InferenceSession.create("best.onnx");
  console.log("ONNX cargado. Salidas:", session.outputNames);
}
cargarModelo();

// ------------------------------
// Non-Maximum Suppression (NMS)
// ------------------------------
function nonMaxSuppression(preds, iouThresh) {
  preds.sort((a,b)=>b.confidence - a.confidence);
  const keep = [];
  const iou = (a,b) => {
    const ax0 = a.x - a.width/2, ay0 = a.y - a.height/2;
    const ax1 = ax0 + a.width,    ay1 = ay0 + a.height;
    const bx0 = b.x - b.width/2,  by0 = b.y - b.height/2;
    const bx1 = bx0 + b.width,    by1 = by0 + b.height;
    const xx0 = Math.max(ax0, bx0), yy0 = Math.max(ay0, by0);
    const xx1 = Math.min(ax1, bx1), yy1 = Math.min(ay1, by1);
    const w   = Math.max(0, xx1 - xx0), h = Math.max(0, yy1 - yy0);
    const inter = w * h;
    const union = a.width*a.height + b.width*b.height - inter;
    return inter / union;
  };
  while (preds.length) {
    const p = preds.shift();
    keep.push(p);
    preds = preds.filter(q => iou(p, q) < iouThresh);
  }
  return keep;
}

// ------------------------------
// Agrupar en filas y dibujar IDs
// ------------------------------
function agruparFilas(boxes) {
  const filas = [], TH = 20;
  boxes.forEach(box => {
    let placed = false;
    for (let f of filas) {
      const avgCy = f.reduce((s,b)=>s+b.cy,0)/f.length;
      if (Math.abs(box.cy - avgCy) < TH) {
        f.push(box);
        placed = true;
        break;
      }
    }
    if (!placed) filas.push([box]);
  });
  return filas;
}
function dibujarPredicciones(preds) {
  const boxes = preds.map(p => {
    const cx = p.x, cy = p.y, w = p.width, h = p.height;
    return { ...p, x0: cx - w/2, y0: cy - h/2, cx, cy, w, h };
  });
  const filas = agruparFilas(boxes);
  filas.forEach(f=>f.sort((a,b)=>a.cx - b.cx));
  filas.sort((A,B)=>{
    const aAvg = A.reduce((s,b)=>s+b.cy,0)/A.length;
    const bAvg = B.reduce((s,b)=>s+b.cy,0)/B.length;
    return aAvg - bAvg;
  });
  let id = 1;
  filas.flat().forEach(b => {
    ctx.strokeStyle = 'red';
    ctx.lineWidth   = 2;
    ctx.strokeRect(b.x0, b.y0, b.w, b.h);
    ctx.font      = '16px sans-serif';
    ctx.fillStyle = 'red';
    const tx = b.x0 + 4;
    const ty = (b.y0 - 4 < 16) ? b.y0 + 16 : b.y0 - 4;
    ctx.fillText(id.toString(), tx, ty);
    id++;
  });
}

// ------------------------------
// Inferencia local ONNX
// ------------------------------
async function inferirLocal() {
  if (!session || !lastDataURL) return;

  // 1) Leer píxeles del canvas
  const imgData = ctx.getImageData(0,0,TARGET_SIZE,TARGET_SIZE).data;
  const input   = new Float32Array(TARGET_SIZE*TARGET_SIZE*3);
  for (let i=0, j=0; i<imgData.length; i+=4, j+=3) {
    input[j]   = imgData[i]   / 255;
    input[j+1] = imgData[i+1] / 255;
    input[j+2] = imgData[i+2] / 255;
  }
  const tensor = new ort.Tensor("float32", input, [1,3,TARGET_SIZE,TARGET_SIZE]);

  // 2) Ejecutar sesión ONNX
  const outputs = await session.run({ images: tensor });
  console.log("Inferencia (raw):", outputs);

  // 3) Si solo hay una salida, repartir canales
  const outNames = session.outputNames;
  if (outNames.length === 1) {
    const outTensor = outputs[outNames[0]];
    const { data, dims } = outTensor;
    // dims = [1,6,numBoxes]
    if (dims.length === 3 && dims[1] === 6) {
      const numBoxes = dims[2];
      const confThresh = parseFloat(sliderConf.value);
      const iouThresh  = parseFloat(sliderOv.value);
      const preds = [];
      for (let i = 0; i < numBoxes; i++) {
        const score = data[4 * numBoxes + i];
        if (score < confThresh) continue;
        const cx  = data[0 * numBoxes + i];
        const cy  = data[1 * numBoxes + i];
        const w   = data[2 * numBoxes + i];
        const h   = data[3 * numBoxes + i];
        const cls = data[5 * numBoxes + i];
        preds.push({ x: cx, y: cy, width: w, height: h, confidence: score, class: cls });
      }
      const finalPreds = nonMaxSuppression(preds, iouThresh);

      // 4) Redibujar imagen + resultados
      const img = new Image();
      img.src = lastDataURL;
      await new Promise(r => img.onload = r);
      ctx.clearRect(0,0,TARGET_SIZE,TARGET_SIZE);
      ctx.drawImage(img,0,0,TARGET_SIZE,TARGET_SIZE);
      dibujarPredicciones(finalPreds);
      return;
    }
    console.error("Formato de salida ONNX inesperado:", dims);
  } else {
    console.error("Se esperaban 1 tensor de salida, llegaron:", outNames.length);
  }
}

// ------------------------------
// Manejo de subida de archivo
// ------------------------------
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    lastDataURL = reader.result;
    const img = new Image();
    img.src = lastDataURL;
    img.onload = () => {
      canvas.width  = canvas.height = TARGET_SIZE;
      ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);
      inferirLocal();
    };
  };
  reader.readAsDataURL(file);
});

// ------------------------------
// Sliders (debounce re-inferencia)
// ------------------------------
let tid;
sliderConf.addEventListener('input', () => {
  labelConf.textContent = parseFloat(sliderConf.value).toFixed(2);
  clearTimeout(tid);
  tid = setTimeout(inferirLocal, 200);
});
sliderOv.addEventListener('input', () => {
  labelOv.textContent = parseFloat(sliderOv.value).toFixed(2);
  clearTimeout(tid);
  tid = setTimeout(inferirLocal, 200);
});

// ------------------------------
// Botón “Procesar” manual
// ------------------------------
boton.addEventListener('click', inferirLocal);
