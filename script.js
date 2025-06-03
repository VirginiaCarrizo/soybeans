// ------------------------------
// ELEMENTOS DEL DOM
// ------------------------------
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const boton       = document.getElementById('botonAccion');
const ctx         = canvas.getContext('2d');

// Sliders y sus labels
const sliderConf  = document.getElementById('sliderConf');
const labelConf   = document.getElementById('labelConf');
const sliderOv    = document.getElementById('sliderOv');
const labelOv     = document.getElementById('labelOv');

// Para almacenar el stream activo
let streamActual    = null;
// Para guardar el dataURL de la última captura (640×640) 
let lastDataURL     = null;

// ------------------------------
// PARÁMETROS DE ROBoflow
// ------------------------------
const ROBOFLOW_API_KEY  = "BB3sh1D4ta8L9zosEHdl";
const MODEL_ENDPOINT    = "https://detect.roboflow.com/beancount/1";

// Tamaño fijo 640×640 (igual que en tu pipeline de entrenamiento)
const TARGET_SIZE       = 640;

// ------------------------------
// Arrancar la cámara trasera
// ------------------------------
async function iniciarCamaraTrasera() {
  try {
    // Intentamos usar facingMode exacto
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
// Detener la cámara (liberar el sensor)
// ------------------------------
function detenerCamara() {
  if (streamActual) {
    streamActual.getTracks().forEach(track => track.stop());
    streamActual = null;
  }
  video.srcObject = null;
}

// ------------------------------
// Dibujar predicciones sobre el canvas
// ------------------------------
function dibujarPredicciones(predicciones) {
  ctx.strokeStyle = 'red';
  ctx.lineWidth   = 2;
  ctx.font        = '16px sans-serif';
  ctx.fillStyle   = 'red';

  predicciones.forEach(pred => {
    // Roboflow entrega (x, y) como centro de la caja en la imagen 640×640
    const cx = pred.x;
    const cy = pred.y;
    const w  = pred.width;
    const h  = pred.height;
    // Convertimos a esquina superior izquierda
    const x0 = cx - w / 2;
    const y0 = cy - h / 2;

    // Dibujar rectángulo
    ctx.strokeRect(x0, y0, w, h);

    // Texto con clase y confianza
    const etiqueta = `${pred.class} (${pred.confidence.toFixed(2)})`;
    ctx.fillText(etiqueta, x0 + 4, y0 + 16);
  });
}

// ------------------------------
// Enviar imagen a Roboflow con los thresholds actuales
// ------------------------------
async function enviarARoboflow(dataURL) {
  // 1) Convertir dataURL a Blob
  const blob = await (await fetch(dataURL)).blob();

  // 2) Empaquetar en FormData
  const formData = new FormData();
  formData.append("file", blob, "soybean.png");

  // 3) Leer valores de sliders en este momento
  const confValue = parseFloat(sliderConf.value);
  const ovValue   = parseFloat(sliderOv.value);

  // 4) Construir URL con api_key, confidence y overlap
  const url = `${MODEL_ENDPOINT}`
            + `?api_key=${ROBOFLOW_API_KEY}`
            + `&confidence=${confValue}`
            + `&overlap=${ovValue}`;

  try {
    // 5) Llamar al endpoint de Roboflow
    const response = await fetch(url, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Error en Roboflow: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    // 6) Dibujar las cajas obtenidas
    dibujarPredicciones(data.predictions);
  } catch (err) {
    console.error("Error en inferencia Roboflow:", err);
    ctx.fillStyle = 'red';
    ctx.fillText("Error en inferencia", 10, 20);
  }
}

// ------------------------------
// Redibujar la misma imagen (640×640) + volver a inferir
// ------------------------------
function reInfer() {
  if (!lastDataURL) return;
  // 1) Creamos un elemento Image para dibujar la base
  const img = new Image();
  img.src = lastDataURL;
  img.onload = async () => {
    // Limpiamos canvas y pintamos la imagen base
    ctx.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
    ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE);

    // 2) Llamamos a Roboflow con la misma dataURL y nuevos thresholds
    await enviarARoboflow(lastDataURL);
  };
}

// ------------------------------
// Listeners en los sliders para actualizar labels y re-inferir
// ------------------------------
sliderConf.addEventListener('input', () => {
  labelConf.textContent = parseFloat(sliderConf.value).toFixed(2);
  // Si ya hay una imagen capturada (canvas visible), hacemos re-inferencia
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
// Manejador del botón “Capturar” / “Tomar otra”
// ------------------------------
boton.addEventListener('click', async () => {
  if (boton.textContent === 'Capturar') {
    // ——— Etapa 1: capturar el frame y estirarlo a 640×640 ———
    canvas.width  = TARGET_SIZE;
    canvas.height = TARGET_SIZE;

    // Dibujamos “estirado” el video completo sobre un canvas 640×640
    ctx.drawImage(
      video,
      0, 0, video.videoWidth, video.videoHeight,  // rect fuente
      0, 0, TARGET_SIZE, TARGET_SIZE               // rect destino estirado
    );

    // Convertimos a dataURL (PNG, sin pérdida)
    const dataURL = canvas.toDataURL('image/png');
    lastDataURL    = dataURL; // guardamos para poder re-inferir con nuevos umbrales

    // Detenemos la cámara y ocultamos el video
    detenerCamara();
    video.style.display  = 'none';
    canvas.style.display = 'block';

    // Cambiamos texto del botón
    boton.textContent = 'Tomar otra';

    // Mostramos un “Cargando...” inicial
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 150, 24);
    ctx.fillStyle = 'black';
    ctx.fillText("Cargando...", 8, 16);

    // ——— Etapa 2: enviamos la imagen a Roboflow y dibujamos cajas ———
    await enviarARoboflow(dataURL);

  } else {
    // ——— Etapa “Tomar otra”: volvemos a mostrar el preview ———
    canvas.style.display = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    await iniciarCamaraTrasera();
    video.style.display = 'block';
    boton.textContent   = 'Capturar';
  }
});

// Al cargar la página, iniciamos la cámara trasera
iniciarCamaraTrasera();
