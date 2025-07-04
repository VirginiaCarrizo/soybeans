/**
 * Agrupa cajas en filas según su coordenada Y.
 */
export function agruparFilas(boxes) {
    const filas = [];
    const TH = 20;
    boxes.forEach(box => {
      let placed = false;
      for (let f of filas) {
        const avgCy = f.reduce((s,b) => s + b.cy, 0) / f.length;
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
  
  /**
   * Dibuja rectángulos y IDs a partir de predicciones.
   */
  export function dibujarPredicciones(preds) {
    const boxes = preds.map(p => {
      const cx = p.x, cy = p.y, w = p.width, h = p.height;
      return { ...p, x0: cx - w/2, y0: cy - h/2, cx, cy, w, h };
    });
    const filas = agruparFilas(boxes);
    filas.forEach(f => f.sort((a,b) => a.cx - b.cx));
    filas.sort((A,B) => {
      const aAvg = A.reduce((s,b) => s + b.cy,0) / A.length;
      const bAvg = B.reduce((s,b) => s + b.cy,0) / B.length;
      return aAvg - bAvg;
    });
    let id = 1;
    const { ctx } = require('../components/dom');
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