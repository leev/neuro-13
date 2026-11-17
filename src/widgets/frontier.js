// ===== Interactive: rough digit recognizer =====
(function() {
  const canvas = document.getElementById('digit-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const bars = document.getElementById('digit-bars');
  const predEl = document.getElementById('digit-pred');
  const W = canvas.width;
  const H = canvas.height;
  let drawing = false;
  let last = null;

  const SIZE = 28;
  const normCanvas = document.createElement('canvas');
  normCanvas.width = SIZE;
  normCanvas.height = SIZE;
  const normCtx = normCanvas.getContext('2d');

  function inkValue(data, i) {
    return Math.max(data[i], data[i + 1], data[i + 2]);
  }

  function vectorFrom(sourceCtx, width, height) {
    const img = sourceCtx.getImageData(0, 0, width, height);
    const data = img.data;
    let minX = width, minY = height, maxX = -1, maxY = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        if (inkValue(data, i) > 80) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    if (maxX < 0) return null;

    const bw = maxX - minX + 1;
    const bh = maxY - minY + 1;
    const scale = Math.min(20 / bw, 20 / bh);
    const dw = Math.max(1, Math.round(bw * scale));
    const dh = Math.max(1, Math.round(bh * scale));
    const dx = Math.round((SIZE - dw) / 2);
    const dy = Math.round((SIZE - dh) / 2);

    normCtx.fillStyle = '#000';
    normCtx.fillRect(0, 0, SIZE, SIZE);
    normCtx.imageSmoothingEnabled = true;
    normCtx.drawImage(canvasFromImageData(img), minX, minY, bw, bh, dx, dy, dw, dh);

    const small = normCtx.getImageData(0, 0, SIZE, SIZE).data;
    const vec = [];
    let mag = 0;
    for (let i = 0; i < small.length; i += 4) {
      const v = inkValue(small, i) / 255;
      vec.push(v);
      mag += v * v;
    }
    mag = Math.sqrt(mag) || 1;
    return vec.map(v => v / mag);
  }

  function canvasFromImageData(img) {
    const c = document.createElement('canvas');
    c.width = img.width;
    c.height = img.height;
    c.getContext('2d').putImageData(img, 0, 0);
    return c;
  }

  function makeTemplate(digit, fontSize, xShift, yShift) {
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const tctx = c.getContext('2d');
    tctx.fillStyle = '#111827';
    tctx.fillRect(0, 0, W, H);
    tctx.fillStyle = '#f9fafb';
    tctx.textAlign = 'center';
    tctx.textBaseline = 'middle';
    tctx.font = `900 ${fontSize}px Arial, sans-serif`;
    tctx.fillText(String(digit), W / 2 + xShift, H / 2 + yShift);
    return vectorFrom(tctx, W, H);
  }

  const templates = [];
  for (let digit = 0; digit <= 9; digit++) {
    for (const fontSize of [205, 225, 245]) {
      for (const xShift of [-10, 0, 10]) {
        templates.push({ digit, vec: makeTemplate(digit, fontSize, xShift, 8) });
      }
    }
  }

  function clear() {
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, W, H);
    renderScores();
  }

  function point(evt) {
    const rect = canvas.getBoundingClientRect();
    const p = evt.touches ? evt.touches[0] : evt;
    return {
      x: (p.clientX - rect.left) * (W / rect.width),
      y: (p.clientY - rect.top) * (H / rect.height),
    };
  }

  function drawLine(a, b) {
    ctx.strokeStyle = '#f9fafb';
    ctx.lineWidth = 26;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function start(evt) {
    evt.preventDefault();
    drawing = true;
    last = point(evt);
  }
  function move(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const p = point(evt);
    drawLine(last, p);
    last = p;
    renderScores();
  }
  function end() {
    drawing = false;
    last = null;
    renderScores();
  }

  function cosine(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function renderScores() {
    const vec = vectorFrom(ctx, W, H);
    if (!vec) {
      predEl.textContent = '?';
      bars.innerHTML = '<em style="color: var(--muted); font-size: 14px;">Draw a digit to see guesses.</em>';
      return;
    }
    const best = new Map();
    for (const t of templates) {
      const score = cosine(vec, t.vec);
      best.set(t.digit, Math.max(best.get(t.digit) || 0, score));
    }
    const raw = Array.from(best.entries());
    const maxScore = Math.max(...raw.map(([,s]) => s));
    const exp = raw.map(([d, s]) => [d, Math.exp((s - maxScore) * 18)]);
    const total = exp.reduce((sum, [,v]) => sum + v, 0) || 1;
    const probs = exp.map(([d, v]) => [d, v / total]).sort((a, b) => b[1] - a[1]);
    predEl.textContent = probs[0][0];
    bars.innerHTML = probs.slice(0, 5).map(([d, p]) => (
      `<div class="bar-container"><span class="label">digit ${d}</span><div class="bar-track"><div class="bar" style="width:${p * 100}%;"></div></div><span class="pct">${p.toFixed(2)}</span></div>`
    )).join('');
  }

  function demoTwo() {
    clear();
    const pts = [
      {x: 82, y: 80}, {x: 140, y: 42}, {x: 202, y: 72},
      {x: 190, y: 122}, {x: 142, y: 160}, {x: 92, y: 212},
      {x: 210, y: 214},
    ];
    for (let i = 1; i < pts.length; i++) drawLine(pts[i-1], pts[i]);
    renderScores();
  }

  canvas.addEventListener('mousedown', start);
  canvas.addEventListener('mousemove', move);
  window.addEventListener('mouseup', end);
  canvas.addEventListener('touchstart', start, { passive: false });
  canvas.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('touchend', end);
  document.getElementById('digit-clear').addEventListener('click', clear);
  document.getElementById('digit-demo').addEventListener('click', demoTwo);
  clear();
})();

