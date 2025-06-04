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

// Para almacenar el stream activo y recordar la última imagen 640×640
let streamActual    = null;
let lastDataURL     = null;

// ------------------------------
// PARÁMETROS ROBoflow
// ------------------------------
const ROBOFLOW_API_KEY  = "BB3sh1D4ta8L9zosEHdl";
const MODEL_ENDPOINT    = "https://detect.roboflow.com/beancount/1";
const TARGET_SIZE       = 640;   // “Resize: Stretch to 640×640”
const ROW_THRESHOLD_PX  = 20;    // Umbral de altura para agrupar en la misma fila

// ------------------------------
// Arrancar la cámara trasera
// ------------------------------
async function iniciarCamaraTrasera() {
  try {
    // Intentar con facingMode exacto
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "environment" } }
    });
    video.srcObject  = stream;
    streamActual     = stream;
  } catch (e) {
    console.warn('No se pudo usar exact:"environment", probando sin exact…', e);
    try {
      const stream2 = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }
      });
      video.srcObject  = stream2;
      streamActual     = stream2;
    } catch (err) {
      alert('No se pudo acceder a la cámara trasera: ' + err.message);
    }
  }
}

// ------------------------------
// Detener la cámara (liberar sensor)
// ------------------------------
function detenerCamara() {
  if (streamActual) {
    streamActual.getTracks().forEach(track => track.stop());
    streamActual = null;
  }
  video.srcObject = null;
}

// ------------------------------
// Agrupar “boxes” en filas usando clustering vertical
// ------------------------------
function agruparFilas(boxes) {
  const filas = [];
  boxes.forEach(box => {
    let colocado = false;
    for (let fila of filas) {
      // Calcular el cy promedio de la fila
      const avgCy = fila.reduce((sum, b) => sum + b.cy, 0) / fila.length;
      if (Math.abs(box.cy - avgCy) < ROW_THRESHOLD_PX) {
        fila.push(box);
        colocado = true;
        break;
      }
    }
    if (!colocado) {
      filas.push([box]);
    }
  });
  return filas;
}

// ------------------------------
// Dibujar predicciones con numeración sobre el canvas
// ------------------------------
function dibujarPredicciones(predicciones) {
  // --- Construir un array “boxes” con propiedades calculadas ---
  const boxes = predicciones.map(pred => {
    const cx = pred.x;
    const cy = pred.y;
    const w  = pred.width;
    const h  = pred.height;
    const x0 = cx - w / 2;
    const y0 = cy - h / 2;
    return { ...pred, x0, y0, cx, cy, w, h };
  });

  // --- Agrupar las cajas en filas ---
  const filasAgrupadas = agruparFilas(boxes);

  // --- Ordenar cada fila por el centro horizontal (cx) y
  //     luego ordenar las filas completas por su altura media (cy promedio) ---
  filasAgrupadas.forEach(fila => {
    fila.sort((a, b) => a.cx - b.cx);
  });
  filasAgrupadas.sort((filaA, filaB) => {
    const avgA = filaA.reduce((sum, b) => sum + b.cy, 0) / filaA.length;
    const avgB = filaB.reduce((sum, b) => sum + b.cy, 0) / filaB.length;
    return avgA - avgB;
  });

  // --- Aplanar las filas en un solo array, asignando ID incremental ---
  let id = 1;
  filasAgrupadas.forEach(fila => {
    fila.forEach(box => {
      // Dibujar el rectángulo
      ctx.strokeStyle = 'red';
      ctx.lineWidth   = 2;
      ctx.strokeRect(box.x0, box.y0, box.w, box.h);

      // Preparar posición del texto (ID)
      ctx.font = '16px sans-serif';
      ctx.fillStyle = 'red';
      const textX = box.x0 + 4;
      const textY = (box.y0 - 4 < 16) ? box.y0 + 16 : box.y0 - 4;

      // Dibujar el ID
      ctx.fillText(id.toString(), textX, textY);
      id += 1;
    });
  });
}

// ------------------------------
// Enviar la imagen 640×640 a Roboflow con thresholds actuales
// ------------------------------
async function enviarARoboflow(dataURL) {
  // Convertir dataURL a Blob
  const blob = await (await fetch(dataURL)).blob();

  // Empaquetar en FormData
  const formData = new FormData();
  formData.append("file", blob, "soybean.png");

  // Leer valores de los sliders
  let confValue = parseFloat(sliderConf.value);
  let ovValue   = parseFloat(sliderOv.value);

  // Clamp para garantizar [0,1]
  confValue = Math.min(Math.max(confValue, 0), 1);
  ovValue   = Math.min(Math.max(ovValue,   0), 1);

  // Construir URL con api_key, confidence y overlap
  const url = `${MODEL_ENDPOINT}`
            + `?api_key=${ROBOFLOW_API_KEY}`
            + `&confidence=${confValue}`
            + `&overlap=${ovValue}`;

  try {
    // Llamada POST a Roboflow Detection API 
    const response = await fetch(url, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      const errorTexto = await response.text();
      console.error(`Roboflow devuelto ${response.status}: ${errorTexto}`);
      throw new Error(`Error ${response.status}`);
    }
    const data = await response.json();

    // Dibujar cajas + numeración sobre la imagen en el canvas
    dibujarPredicciones(data.predictions);
  } catch (err) {
    console.error("Error en inferencia Roboflow:", err);
    ctx.fillStyle = 'red';
    ctx.fillText("Error en inferencia", 10, 20);
  }
}

// ------------------------------
// Re-inferir cuando muevas los sliders, sobre la última imagen capturada
// ------------------------------
function reInfer() {
  if (!lastDataURL) return;
  const img = new Image();
  img.src = lastDataURL;
  img.onload = async () => {
    // Limpiar y dibujar la imagen base 640×640
    ctx.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
    ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);
    // Enviar de nuevo a Roboflow con thresholds actualizados
    await enviarARoboflow(lastDataURL);
  };
}

// ------------------------------
// Debounce: no ejecutar reInfer más de una vez cada 200 ms
// ------------------------------
let debounceTimeout = null;

// Slider “confidence”
sliderConf.addEventListener('input', () => {
  labelConf.textContent = parseFloat(sliderConf.value).toFixed(2);
  if (canvas.style.display === 'block' && lastDataURL) {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(reInfer, 200);
  }
});

// Slider “overlap”
sliderOv.addEventListener('input', () => {
  labelOv.textContent = parseFloat(sliderOv.value).toFixed(2);
  if (canvas.style.display === 'block' && lastDataURL) {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(reInfer, 200);
  }
});

// ------------------------------
// Manejo del botón “Capturar” / “Tomar otra”
// ------------------------------
boton.addEventListener('click', async () => {
  if (boton.textContent === 'Capturar') {
    // —— Etapa 1: capturar frame y estirarlo a 640×640 ——
    canvas.width  = TARGET_SIZE;
    canvas.height = TARGET_SIZE;

    // Dibujar el video “estirado” en el canvas 640×640
    ctx.drawImage(
      video,
      0, 0, video.videoWidth, video.videoHeight,  // rect fuente
      0, 0, TARGET_SIZE, TARGET_SIZE               // rect destino estirado
    );

    // Convertir a dataURL (PNG, sin pérdida)
    const dataURL = canvas.toDataURL('image/png');
    lastDataURL    = dataURL;

    // Detener y ocultar la cámara
    detenerCamara();
    video.style.display  = 'none';
    canvas.style.display = 'block';

    // Cambiar texto del botón principal
    boton.textContent = 'Tomar otra';
    // Mostrar el botón “Examinar”
    btnExaminar.style.display = 'inline-block';

    // Pintar un “Cargando…” previo
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 150, 24);
    ctx.fillStyle = 'black';
    ctx.fillText("Cargando...", 8, 16);

    // —— Etapa 2: enviar la imagen a Roboflow y dibujar cajas + IDs ——
    await enviarARoboflow(dataURL);

  } else {
    // —— Cuando el botón dice “Tomar otra”: revertir al preview ——
    canvas.style.display    = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Ocultar “Examinar”
    btnExaminar.style.display = 'none';

    // Reiniciar la cámara
    await iniciarCamaraTrasera();
    video.style.display = 'block';
    boton.textContent   = 'Capturar';
  }
});

// ------------------------------
// Listener para “Examinar” (por implementar)
// ------------------------------
btnExaminar.addEventListener('click', () => {
  alert('Funcionalidad “Examinar” pendiente de implementar.');
});

// ------------------------------
// Al cargar la página, arrancar la cámara trasera
// ------------------------------
iniciarCamaraTrasera();
