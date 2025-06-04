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
const TARGET_SIZE       = 640;      // “Resize: Stretch to 640×640”

// ------------------------------
// Arrancar la cámara trasera
// ------------------------------
async function iniciarCamaraTrasera() {
  try {
    // Intento con facingMode exacto
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
// Dibujar predicciones con numeración sobre el canvas
// ------------------------------
function dibujarPredicciones(predicciones) {
  // Primero transformamos cada pred a un objeto con x0, y0, cx, cy, w, h
  const boxes = predicciones.map(pred => {
    const cx = pred.x;
    const cy = pred.y;
    const w  = pred.width;
    const h  = pred.height;
    const x0 = cx - w / 2;
    const y0 = cy - h / 2;
    return { ...pred, x0, y0, cx, cy, w, h };
  });

  // Ordenamos de arriba a abajo (por cy) y, dentro de la misma fila, de izquierda a derecha (por cx)
  boxes.sort((a, b) => {
    // Comparamos centro Y
    if (Math.abs(a.cy - b.cy) > 10) {
      return a.cy - b.cy;
    }
    // Si están casi a la misma altura (diferencia < 10 px), comparamos centro X
    return a.cx - b.cx;
  });

  // Dibujamos cada caja en orden y asignamos ID incremental
  boxes.forEach((box, idx) => {
    const id = idx + 1;
    // Rectángulo
    ctx.strokeStyle = 'red';
    ctx.lineWidth   = 2;
    ctx.strokeRect(box.x0, box.y0, box.w, box.h);

    // Texto del ID: lo colocamos un poco arriba de la esquina superior-izquierda
    ctx.font      = '16px sans-serif';
    ctx.fillStyle = 'red';
    // Si y0 es muy pequeño, dibujamos dentro de la caja para que no se salga
    const textX = box.x0 + 4;
    const textY = box.y0 - 4 < 16 ? box.y0 + 16 : box.y0 - 4;
    ctx.fillText(id.toString(), textX, textY);
  });
}

// ------------------------------
// Enviar la imagen 640×640 a Roboflow con thresholds actuales
// ------------------------------
async function enviarARoboflow(dataURL) {
  // 1) Convertir dataURL a Blob
  const blob = await (await fetch(dataURL)).blob();

  // 2) Empaquetar en FormData
  const formData = new FormData();
  formData.append("file", blob, "soybean.png");

  // 3) Leer valores de los sliders en el momento
  const confValue = parseFloat(sliderConf.value);
  const ovValue   = parseFloat(sliderOv.value);

  // 4) Construir URL con api_key, confidence y overlap
  const url = `${MODEL_ENDPOINT}`
            + `?api_key=${ROBOFLOW_API_KEY}`
            + `&confidence=${confValue}`
            + `&overlap=${ovValue}`;

  try {
    // 5) Llamar al endpoint de detección de Roboflow 
    const response = await fetch(url, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Error en Roboflow: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();

    // 6) Dibujar las cajas + IDs sobre la imagen en el canvas
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
    // Limpiamos y dibujamos la imagen base 640×640
    ctx.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
    ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);
    // Enviamos de nuevo a Roboflow con thresholds actualizados
    await enviarARoboflow(lastDataURL);
  };
}

// ------------------------------
// Actualizar etiquetas de los sliders y re-inferir si hace falta
// ------------------------------
sliderConf.addEventListener('input', () => {
  labelConf.textContent = parseFloat(sliderConf.value).toFixed(2);
  if (canvas.style.display === 'block' && lastDataURL) {
    reInfer();
  }
});
sliderOv.addEventListener('input', () => {
  labelOv.textContent = parseFloat(sliderOv.value).toFixed(2);
  if (canvas.style.display === 'block' && lastDataURL) {
    reInfer();
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

    // Dibujamos el video completo “estirado” en el canvas 640×640
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

    // Pintamos un “Cargando…” previo
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
  // Aquí podrías, por ejemplo, abrir un <input type="file"> para subir
  // una imagen de la galería. Por ahora mostramos un alert de ejemplo:
  alert('Funcionalidad “Examinar” pendiente de implementar.');
});

// ------------------------------
// Al cargar la página, arrancar la cámara trasera
// ------------------------------
iniciarCamaraTrasera();
