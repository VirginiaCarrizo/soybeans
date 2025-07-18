// Extraemos el canvas y su contexto
export const canvas = document.getElementById('canvas');
export const ctx    = canvas.getContext('2d');

// (Opcional) Si quieres forzar un tamaño de recorte, ajústalo aquí
export const TARGET_SIZE = 416;
