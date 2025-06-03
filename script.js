// Seleccionamos cada elemento del DOM
const video = document.getElementById('video');
const foto = document.getElementById('foto');
const boton = document.getElementById('botonAccion');
const canvas = document.getElementById('canvas');
const resultado = document.getElementById('resultado');

// Variable para mantener la referencia del stream actual
let streamActual = null;

// Tus credenciales de Roboflow
const ROBOFLOW_API_KEY = "BB3sh1D4ta8L9zosEHdl";
// Identificador del modelo en Roboflow Universe (“ddd-aiw7a/beancount/1”)
const MODEL_ENDPOINT = "https://detect.roboflow.com/beancount/1";

// ======================
// Función para iniciar la cámara trasera
// ======================
async function iniciarCamaraTrasera() {
  try {
    // Intentamos primero con facingMode exacto
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { exact: "environment" } }
    });
    video.srcObject = stream;
    streamActual = stream;
  } catch (e) {
    console.warn('Fallo exact:"environment"; probamos sin exact…', e);
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

// ======================
// Función para detener la cámara (liberar sensor)
// ======================
function detenerCamara() {
  if (streamActual) {
    streamActual.getTracks().forEach(track => track.stop());
    streamActual = null;
  }
  video.srcObject = null;
}

// ======================
// Función que envía la imagen a Roboflow y muestra la respuesta
// ======================
async function enviarARoboflow(dataURL) {
  // Convertimos el dataURL a Blob
  const blob = await (await fetch(dataURL)).blob();

  // Preparamos FormData con la imagen
  const formData = new FormData();
  formData.append("file", blob, "soybean.png");

  try {
    // Hacemos POST al endpoint de Roboflow (HTTP Inference API) :contentReference[oaicite:0]{index=0}
    const response = await fetch(
      `${MODEL_ENDPOINT}?api_key=${ROBOFLOW_API_KEY}`, 
      {
        method: "POST",
        body: formData
      }
    );
    if (!response.ok) {
      throw new Error(`Error en Roboflow: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    // Mostramos el JSON en el <pre>
    resultado.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    resultado.textContent = "Error al invocar Roboflow:\n" + err.message;
  }
}

// ======================
// Evento al hacer click en el botón
// ======================
boton.addEventListener('click', async () => {
  if (boton.textContent === 'Capturar') {
    // —————— Etapa 1: Capturar y mostrar la foto ——————
    // 1) Dibujar el frame del video en el canvas
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2) Convertir a dataURL (PNG)
    const dataURL = canvas.toDataURL('image/png');

    // 3) Ocultar el <video> y mostrar el <img> con la foto capturada
    video.style.display = 'none';
    foto.src = dataURL;
    foto.style.display = 'block';

    // 4) Detener la cámara para liberar la lente
    detenerCamara();

    // 5) Cambiar texto del botón a “Tomar otra”
    boton.textContent = 'Tomar otra';

    // —————— Etapa 2: Enviar la imagen a Roboflow ——————
    resultado.textContent = "Enviando a Roboflow…";
    await enviarARoboflow(dataURL);
  } else {
    // —————— Etapa “Tomar otra”: volver al preview de cámara ——————
    foto.style.display = 'none';
    resultado.textContent = "";
    video.style.display = 'block';
    await iniciarCamaraTrasera();
    boton.textContent = 'Capturar';
  }
});

// Arrancamos la cámara trasera al cargar la página
iniciarCamaraTrasera();
