// Elementos del DOM
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const boton = document.getElementById('botonAccion');

// Contexto 2D del canvas
const ctx = canvas.getContext('2d');

// Para almacenar el stream activo
let streamActual = null;

// Credenciales Roboflow
const ROBOFLOW_API_KEY = "BB3sh1D4ta8L9zosEHdl";
const MODEL_ENDPOINT = "https://detect.roboflow.com/beancount/1";
const CONF_THRESHOLD = 0.4;   // mínima confianza (50%)
const OVERLAP_THRESHOLD = 0.30; // NMS: máximo 30% de solape

// ------------------------------
// Iniciar cámara trasera
// ------------------------------
async function iniciarCamaraTrasera() {
  try {
    // Intento con facingMode exacto
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "environment" } }
    });
    video.srcObject = stream;
    streamActual = stream;
  } catch (e) {
    console.warn('No se pudo usar exact:"environment", probamos sin exact…', e);
    try {
      const stream2 = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" }
      });
      video.srcObject = stream2;
      streamActual = stream2;
    } catch (err) {
      alert('No se pudo acceder a la cámara trasera: ' + err.message);
    }
  }
}

// ------------------------------
// Detener cámara
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
  // Asumimos que el canvas ya tiene la imagen capturada
  // Configuramos estilos para rectángulos y texto
  ctx.strokeStyle = 'red';
  ctx.lineWidth = 2;
  ctx.font = '16px sans-serif';
  ctx.fillStyle = 'red';

  predicciones.forEach(pred => {
    // Roboflow usa x,y como centro del box, así que convertimos a esquina superior
    const cx = pred.x;
    const cy = pred.y;
    const w = pred.width;
    const h = pred.height;
    const x0 = cx - w / 2;
    const y0 = cy - h / 2;

    // Dibujar rectángulo
    ctx.strokeRect(x0, y0, w, h);

    // Preparar texto con clase y confiabilidad
    const etiqueta = `${pred.class} (${(pred.confidence).toFixed(2)})`;
    // Dibujamos el texto justo encima del rectángulo
    ctx.fillText(etiqueta, x0 + 4, y0 + 16);
  });
}

// ------------------------------
// Enviar imagen capturada a Roboflow y procesar respuesta
// ------------------------------
async function enviarARoboflow(dataURL) {
    // 1) Convertir dataURL a Blob
    const blob = await (await fetch(dataURL)).blob();
    // 2) Empaquetar en FormData
    const formData = new FormData();
    formData.append("file", blob, "soybean.png");
  
    // 3) Construir la URL con api_key + confidence + overlap
    const url = `${MODEL_ENDPOINT}`
              + `?api_key=${ROBOFLOW_API_KEY}`
              + `&confidence=${CONF_THRESHOLD}`
              + `&overlap=${OVERLAP_THRESHOLD}`;
  
    try {
      // 4) Hacer POST con los thresholds
      const response = await fetch(url, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error(`Error en Roboflow: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      // 5) Dibujar predicciones con 'data.predictions'
      dibujarPredicciones(data.predictions);
    } catch (err) {
      // maneja errores
      ctx.fillStyle = 'red';
      ctx.fillText("Error en inferencia", 10, 20);
      console.error("Error al invocar Roboflow:", err);
    }
  }

// ------------------------------
// Manejador del botón
// ------------------------------
boton.addEventListener('click', async () => {
  if (boton.textContent === 'Capturar') {
    // —— Etapa 1: capturar el frame en el canvas ——
    // Ajustamos tamaño del canvas al video real
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    // Dibujamos el frame actual del video en el canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convertimos a dataURL
    const dataURL = canvas.toDataURL('image/png');

    // Detenemos la cámara
    detenerCamara();

    // Mostramos el canvas y ocultamos el video
    video.style.display = 'none';
    canvas.style.display = 'block';

    // Cambiamos el texto del botón
    boton.textContent = 'Tomar otra';

    // —— Etapa 2: enviar a Roboflow —— 
    // (dibujaremos sobre el mismo canvas donde ya está la foto)
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 150, 24);
    ctx.fillStyle = 'black';
    ctx.fillText("Cargando...", 8, 16);
    await enviarARoboflow(dataURL);
  } else {
    // —— Etapa “Tomar otra”: volver al preview —— 
    // Ocultamos el canvas y borramos su contenido
    canvas.style.display = 'none';
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reiniciamos la cámara
    await iniciarCamaraTrasera();
    video.style.display = 'block';

    // Restauramos texto del botón
    boton.textContent = 'Capturar';
  }
});

// Arrancar cámara al cargar
iniciarCamaraTrasera();
