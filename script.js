// ------------------------------
// ELEMENTOS DEL DOM
// ------------------------------
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const boton       = document.getElementById('botonAccion');
const btnExaminar = document.getElementById('btnExaminar');
const ctx         = canvas.getContext('2d');

const sliderConf  = document.getElementById('sliderConf');
const labelConf   = document.getElementById('labelConf');
const sliderOv    = document.getElementById('sliderOv');
const labelOv     = document.getElementById('labelOv');

const TARGET_SIZE = 640;
let streamActual  = null;
let lastDataURL   = null;

// ------------------------------
// ONNX Runtime Web
// ------------------------------
let session = null;
async function cargarModeloLocal() {
  // Asegúrate de que best.onnx esté junto a index.html
  session = await ort.InferenceSession.create("best.onnx");
  console.log("Modelo ONNX cargado.");
  console.log("Claves de salida del modelo ONNX:", session.outputNames);
}
cargarModeloLocal();

// ------------------------------
// Cámara trasera
// ------------------------------
async function iniciarCamaraTrasera() {
  try {
    const s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "environment" } }
    });
    video.srcObject  = s;
    streamActual     = s;
  } catch {
    const s2 = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }
    });
    video.srcObject  = s2;
    streamActual     = s2;
  }
}
function detenerCamara() {
  if (streamActual) {
    streamActual.getTracks().forEach(t => t.stop());
    streamActual = null;
  }
  video.srcObject = null;
}

// ------------------------------
// Non-Maximum Suppression (NMS)
// ------------------------------
function nonMaxSuppression(preds, iouThresh) {
  preds.sort((a,b)=>b.confidence - a.confidence);
  const keep = [];
  const iou = (a,b) => {
    const ax0=a.x-a.width/2, ay0=a.y-a.height/2, ax1=ax0+a.width, ay1=ay0+a.height;
    const bx0=b.x-b.width/2, by0=b.y-b.height/2, bx1=bx0+b.width, by1=by0+b.height;
    const xx0=Math.max(ax0,bx0), yy0=Math.max(ay0,by0);
    const xx1=Math.min(ax1,bx1), yy1=Math.min(ay1,by1);
    const w=Math.max(0,xx1-xx0), h=Math.max(0,yy1-yy0);
    const inter = w*h;
    const union = a.width*a.height + b.width*b.height - inter;
    return inter/union;
  };
  while (preds.length) {
    const p = preds.shift();
    keep.push(p);
    preds = preds.filter(q => iou(p,q) < iouThresh);
  }
  return keep;
}

// ------------------------------
// Agrupar en filas y dibujar IDs
// ------------------------------
function agruparFilas(boxes) {
  const filas = [];
  const TH = 20;
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
  // Convertir predicciones a cajas con (x0,y0,w,h)
  const boxes = preds.map(p => {
    const cx=p.x, cy=p.y, w=p.width, h=p.height;
    return { ...p, x0: cx-w/2, y0: cy-h/2, cx, cy, w, h };
  });
  // Agrupar y ordenar
  const filas = agruparFilas(boxes);
  filas.forEach(f=>f.sort((a,b)=>a.cx-b.cx));
  filas.sort((A,B)=>{
    const aAvg=A.reduce((s,b)=>s+b.cy,0)/A.length;
    const bAvg=B.reduce((s,b)=>s+b.cy,0)/B.length;
    return aAvg - bAvg;
  });
  // Dibujar
  let id=1;
  filas.flat().forEach(b => {
    ctx.strokeStyle='red'; ctx.lineWidth=2;
    ctx.strokeRect(b.x0,b.y0,b.w,b.h);
    ctx.font='16px sans-serif'; ctx.fillStyle='red';
    const tx=b.x0+4, ty=(b.y0-4<16?b.y0+16:b.y0-4);
    ctx.fillText(id.toString(),tx,ty);
    id++;
  });
}

// ------------------------------
// Inferencia local con ONNX
// ------------------------------
async function inferirConModeloLocal() {
  if (!session || !lastDataURL) return;

  // 1) Leer pixels del canvas 640×640
  const imgData = ctx.getImageData(0,0,TARGET_SIZE,TARGET_SIZE).data;
  const input = new Float32Array(TARGET_SIZE*TARGET_SIZE*3);
  for (let i=0,j=0; i<imgData.length; i+=4,j+=3) {
    input[j]   = imgData[i]   / 255;
    input[j+1] = imgData[i+1] / 255;
    input[j+2] = imgData[i+2] / 255;
  }
  const tensor = new ort.Tensor("float32", input, [1,3,TARGET_SIZE,TARGET_SIZE]);

  // 2) Ejecutar sesión ONNX
  const outputs = await session.run({ images: tensor });
  console.log("Resultados de la inferencia:", outputs);
  // 3) Extraer los valores en orden (sin asumir nombres)
  const outVals = Object.values(outputs);
  if (outVals.length < 3) {
    console.error("Esperaba 3 salidas (boxes, scores, classes), encontré:", outVals.length);
    return;
  }
  const [boxesT, scoresT, classesT] = outVals;
  const boxesData  = boxesT.data;
  const scoresData = scoresT.data;
  const clsData    = classesT.data;

  // 4) Filtrar por confianza y aplicar NMS
  const confThresh = parseFloat(sliderConf.value);
  const iouThresh  = parseFloat(sliderOv.value);
  const preds = [];
  for (let i=0; i<scoresData.length; i++) {
    const score = scoresData[i];
    if (score < confThresh) continue;
    const cx = boxesData[i*4], cy = boxesData[i*4+1];
    const w  = boxesData[i*4+2], h  = boxesData[i*4+3];
    preds.push({ x:cx, y:cy, width:w, height:h, confidence:score, class:clsData[i] });
  }
  const finalPreds = nonMaxSuppression(preds, iouThresh);

  // 5) Redibujar imagen + predicciones
  const img = new Image();
  img.src = lastDataURL;
  await new Promise(r => img.onload = r);
  ctx.clearRect(0,0,TARGET_SIZE,TARGET_SIZE);
  ctx.drawImage(img,0,0,TARGET_SIZE,TARGET_SIZE);
  dibujarPredicciones(finalPreds);
}

// ------------------------------
// Sliders (debounce re-inferencia)
// ------------------------------
let timeoutId;
sliderConf.addEventListener('input',()=>{
  labelConf.textContent = parseFloat(sliderConf.value).toFixed(2);
  if (canvas.style.display==='block') {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(inferirConModeloLocal,200);
  }
});
sliderOv.addEventListener('input',()=>{
  labelOv.textContent = parseFloat(sliderOv.value).toFixed(2);
  if (canvas.style.display==='block') {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(inferirConModeloLocal,200);
  }
});

// ------------------------------
// Botón Capturar / Tomar otra
// ------------------------------
boton.addEventListener('click', async ()=>{
  if (boton.textContent==='Capturar') {
    // Capturar frame 640×640
    canvas.width = canvas.height = TARGET_SIZE;
    ctx.drawImage(video,0,0,video.videoWidth,video.videoHeight,0,0,TARGET_SIZE,TARGET_SIZE);
    lastDataURL = canvas.toDataURL('image/png');
    detenerCamara();
    video.style.display='none'; canvas.style.display='block';
    boton.textContent='Tomar otra'; btnExaminar.style.display='inline-block';
    ctx.fillStyle='white'; ctx.fillRect(0,0,150,24);
    ctx.fillStyle='black'; ctx.fillText('Cargando...',8,16);
    await inferirConModeloLocal();
  } else {
    // Volver a preview
    canvas.style.display='none'; ctx.clearRect(0,0,canvas.width,canvas.height);
    btnExaminar.style.display='none';
    await iniciarCamaraTrasera();
    video.style.display='block'; boton.textContent='Capturar';
  }
});

// ------------------------------
// Botón Examinar (pendiente)
// ------------------------------
btnExaminar.addEventListener('click',()=>alert('Funcionalidad “Examinar” pendiente.'));

// ------------------------------
// Iniciar al cargar la página
// ------------------------------
iniciarCamaraTrasera();
