/**
 * Non-Maximum Suppression (NMS).
 * @param {Array} preds - Lista de predicciones con {x, y, width, height, confidence}
 * @param {number} iouThresh - Umbral de IOU
 * @returns {Array} - Predicciones filtradas
 */
export function nonMaxSuppression(preds, iouThresh) {
    preds.sort((a,b) => b.confidence - a.confidence);
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