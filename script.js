// ------------------------------
// ELEMENTOS DEL DOM
// ------------------------------
const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const boton  = document.getElementById('botonAccion');
const ctx    = canvas.getContext('2d');

// Mantiene el stream activo para detenerlo luego
let streamActual = null;

// ------------------------------
// PARÁMETROS ROBoflow
// ------------------------------
const ROBOFLOW_API_KEY  = "BB3sh1D4ta8L9zosEHdl";
const MODEL_ENDPOINT    = "https://detect.roboflow.com/beancount/1";

// Umbrales recomendados (los puedes ajustar según tu modelo)
const CONF_THRESHOLD    = 0.8;   // Ejemplo: 0.5
const OVERLAP_THRESHOLD = 0.2;   // Ejemplo: 0.3

// Tamaño fijo 640×640 (precisamente como “Resize: Stretch to 640×640” en tu entrenamiento)
const TARGET_SIZE = 640;

// ------------------------------
// Función para arrancar la cámara trasera
// ------------------------------
async function iniciarCamaraTrasera() {
  try {
    // Primero probamos con facingMode exacto:
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
// Función para detener la cámara
// ------------------------------
function detenerCamara() {
  if (streamActual) {
    streamActual.getTracks().forEach(track => track.stop());
    streamActual = null;
  }
  video.srcObject = null;
}

// ------------------------------
// Función para dibujar las predicciones sobre el canvas
// ------------------------------
function dibujarPredicciones(predicciones) {
  ctx.strokeStyle = 'red';
  ctx.lineWidth   = 2;
  ctx.font        = '16px sans-serif';
  ctx.fillStyle   = 'red';

  predicciones.forEach(pred => {
    // Roboflow entrega x,y como el centro de la caja en la imagen 640×640
    const cx = pred.x;
    const cy = pred.y;
    const w  = pred.width;
    const h  = pred.height;
    // Convertimos a esquina superior izquierda:
    const x0 = cx - w / 2;
    const y0 = cy - h / 2;

    // Dibujar rectángulo
    ctx.strokeRect(x0, y0, w, h);

    // Texto: clase y confianza
    const etiqueta = `${pred.class} (${pred.confidence.toFixed(2)})`;
    ctx.fillText(etiqueta, x0 + 4, y0 + 16);
  });
}

// ------------------------------
// Función para enviar la imagen redimensionada a Roboflow
// ------------------------------
async function enviarARoboflow(dataURL) {
  // 1) Convertimos el dataURL (PNG) a Blob
  const blob = await (await fetch(dataURL)).blob();

  // 2) Creamos FormData y adjuntamos la imagen
  const formData = new FormData();
  formData.append("file", blob, "soybean.png");

  // 3) Construimos la URL con api_key, confidence y overlap
  const url = `${MODEL_ENDPOINT}`
            + `?api_key=${ROBOFLOW_API_KEY}`
            + `&confidence=${CONF_THRESHOLD}`
            + `&overlap=${OVERLAP_THRESHOLD}`;

  try {
    // 4) Llamada POST a Roboflow Detection API 
    const response = await fetch(url, {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      throw new Error(`Error en Roboflow: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    // 5) Dibujar las cajas sobre la imagen en el canvas
    dibujarPredicciones(data.predictions);
  } catch (err) {
    console.error("Error en inferencia Roboflow:", err);
    ctx.fillStyle = 'red';
    ctx.fillText("Error en inferencia", 10, 20);
  }
}

// ------------------------------
// Manejador del botón “Capturar” / “Tomar otra”
// ------------------------------
boton.addEventListener('click', async () => {
  if (boton.textContent === 'Capturar') {
    // — Etapa 1: capturar el frame y redimensionarlo a 640×640 — 

    // Ajustar el tamaño del canvas exactamente a 640×640
    canvas.width  = TARGET_SIZE;
    canvas.height = TARGET_SIZE;

    // Dibujamos “estirando” el video completo sobre 640×640 (sin respetar aspecto)
    ctx.drawImage(video,
      0, 0, video.videoWidth, video.videoHeight, // fuente
      0, 0, TARGET_SIZE, TARGET_SIZE               // destino estirado a 640×640
    );

    // Convertimos el canvas a dataURL (formato PNG, sin pérdida)
    const dataURL = canvas.toDataURL('image/png');

    // Detenemos la cámara y ocultamos el video
    detenerCamara();
    video.style.display  = 'none';
    canvas.style.display = 'block';

    // Cambiamos el texto del botón
    boton.textContent = 'Tomar otra';

    // Mostramos un “Cargando…” temporal
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 150, 24);
    ctx.fillStyle = 'black';
    ctx.fillText("Cargando...", 8, 16);

    // — Etapa 2: enviar la imagen 640×640 a Roboflow —
    await enviarARoboflow(dataURL);

  } else {
    // — Etapa “Tomar otra”: volvemos a mostrar el preview — 
    canvas.style.display = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    await iniciarCamaraTrasera();
    video.style.display = 'block';

    // Restauramos el texto del botón
    boton.textContent = 'Capturar';
  }
});

// ------------------------------
// Arrancamos la cámara al cargar la página
// ------------------------------
iniciarCamaraTrasera();
