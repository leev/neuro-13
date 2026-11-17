// ---- shared helpers ----
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

// Render a model's stored numbers as a JSON-style dump, highlighting any
// value whose displayed digits changed since the previous update().
function makeModelDump(elId, opts = {}) {
  const el = document.getElementById(elId);
  const dp = opts.decimals === undefined ? 3 : opts.decimals;
  const rowComments = opts.rowComments || {};   // key -> per-row labels (2-D values)
  const headComments = opts.headComments || {}; // key -> comment line above the key
  let prev = null;

  function fmt(v) { return (v >= 0 ? ' ' : '') + v.toFixed(dp); }
  function num(v, pv) {
    const s = fmt(v);
    const changed = pv !== null && pv !== undefined && fmt(pv) !== s;
    return changed ? '<span class="chg">' + s + '</span>' : s;
  }
  function inlineCmt(rcs) {
    return typeof rcs === 'string' ? ' <span class="cmt">// ' + rcs + '</span>' : '';
  }
  function renderVal(v, pv, indent, tail, rcs) {
    if (typeof v === 'number') return num(v, pv) + tail + inlineCmt(rcs);
    if (!Array.isArray(v[0])) {
      return '[' + v.map((x, i) => num(x, pv ? pv[i] : null)).join(', ') + ']' + tail + inlineCmt(rcs);
    }
    const lines = v.map((row, i) =>
      '[' + row.map((x, j) => num(x, pv ? pv[i][j] : null)).join(', ') + ']'
    );
    return lines.map((line, i) => {
      const last = i === lines.length - 1;
      let s = (i === 0 ? '[' : ' '.repeat(indent + 1)) + line + (last ? ']' + tail : ',');
      if (rcs && rcs[i] !== undefined) s += ' <span class="cmt">// ' + rcs[i] + '</span>';
      return s;
    }).join('\n');
  }
  function update(model) {
    if (!el) return;
    const keys = Object.keys(model);
    const w = Math.max(...keys.map(k => k.length));
    let out = '{\n';
    keys.forEach((k, i) => {
      const tail = i < keys.length - 1 ? ',' : '';
      if (headComments[k]) out += (i ? '\n' : '') + '  <span class="cmt">// ' + headComments[k] + '</span>\n';
      const head = '  "' + k + '": ' + ' '.repeat(w - k.length);
      out += head + renderVal(model[k], prev ? prev[k] : null, head.length, tail, rowComments[k]) + '\n';
    });
    out += '}';
    el.innerHTML = out;
    prev = JSON.parse(JSON.stringify(model));
  }
  return { update, reset() { prev = null; } };
}


// ===== generic single-neuron trainer =====
function makeSingleTrainer(opts) {
  const { data, labels, predIds, statIds, btnIds, lr=0.5 } = opts;
  const dump = makeModelDump(opts.modelId, {
    rowComments: { w1: labels[0], w2: labels[1], bias: 'the built-in lean' },
  });
  let w1, w2, b, epoch;

  function reset() {
    w1 = Math.random()*2 - 1;
    w2 = Math.random()*2 - 1;
    b  = Math.random()*2 - 1;
    epoch = 0;
    dump.reset();
    render();
  }
  function step(n=1) {
    for (let i = 0; i < n; i++) {
      let dw1=0, dw2=0, db=0;
      for (const [[x1,x2], y] of data) {
        const o = sigmoid(w1*x1 + w2*x2 + b);
        const err = y - o;
        const g = err * o * (1-o);
        dw1 += g*x1; dw2 += g*x2; db += g;
      }
      w1 += lr*dw1; w2 += lr*dw2; b += lr*db;
      epoch++;
    }
    render();
  }
  function render() {
    let html = '';
    let total = 0;
    for (let i = 0; i < data.length; i++) {
      const [[x1,x2], y] = data[i];
      const o = sigmoid(w1*x1 + w2*x2 + b);
      const ok = (o>0.5) === (y>0.5);
      total += (y-o)*(y-o);
      const mark = opts.markUnsure && o.toFixed(3) === '0.500'
        ? '<span style="color: var(--muted);">?</span>'
        : ok
          ? '<span class="ok-mark">✓</span>'
          : '<span class="bad-mark">✗</span>';
      html += `${labels[0]}=${x1}, ${labels[1]}=${x2}  →  predicted <strong>${o.toFixed(3)}</strong>  (truth ${y})  ${mark}<br>`;
    }
    document.getElementById(predIds).innerHTML = html;
    document.getElementById(statIds.epoch).textContent = epoch;
    if (statIds.w1) document.getElementById(statIds.w1).textContent = w1.toFixed(2);
    if (statIds.w2) document.getElementById(statIds.w2).textContent = w2.toFixed(2);
    if (statIds.b)  document.getElementById(statIds.b).textContent  = b.toFixed(2);
    if (statIds.err) document.getElementById(statIds.err).textContent = (total/data.length).toFixed(4);
    dump.update({ w1: w1, w2: w2, bias: b });
  }
  document.getElementById(btnIds.step).addEventListener('click', () => step(1));
  document.getElementById(btnIds.fast).addEventListener('click', () => step(1000));
  document.getElementById(btnIds.reset).addEventListener('click', reset);
  reset();
}


// ===== Shared widget: draw a separating line =====
function makeSeparator(containerId, data, featureLabels, opts = {}) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const [startW1, startW2, startB] = opts.start || [0.3, 0.3, 0];
  const nearMiss = opts.nearMiss || 'so close — but no line gets all 4 here';
  const W = 360, H = 280;
  const xMin = -0.3, xMax = 1.3;
  const yMin = -0.3, yMax = 1.3;
  function xToPx(x) { return 40 + ((x - xMin) / (xMax - xMin)) * (W - 60); }
  function yToPx(y) { return H - 40 - ((y - yMin) / (yMax - yMin)) * (H - 60); }

  container.innerHTML = `
    <svg id="${containerId}-svg" viewBox="0 0 ${W} ${H}" style="background:#f9fafb;border-radius:8px;display:block;margin:0 auto 12px;max-width:${W}px;"></svg>
    <div class="slider-row">
      <label>weight 1 (${featureLabels[0]})</label>
      <input type="range" id="${containerId}-w1" min="-3" max="3" step="0.01" value="${startW1}">
      <span class="value" id="${containerId}-w1-v">0.30</span>
    </div>
    <div class="slider-row">
      <label>weight 2 (${featureLabels[1]})</label>
      <input type="range" id="${containerId}-w2" min="-3" max="3" step="0.01" value="${startW2}">
      <span class="value" id="${containerId}-w2-v">0.30</span>
    </div>
    <div class="slider-row">
      <label>bias</label>
      <input type="range" id="${containerId}-b" min="-3" max="3" step="0.01" value="${startB}">
      <span class="value" id="${containerId}-b-v">0.00</span>
    </div>
    <div class="stat-row">
      <span>correct: <strong id="${containerId}-correct">?</strong> / ${data.length}</span>
      <span id="${containerId}-msg" style="color: var(--muted);"></span>
    </div>
  `;

  function predict(x1, x2, w1, w2, b) {
    return (w1*x1 + w2*x2 + b) > 0 ? 1 : 0;
  }

  function render() {
    const w1 = +document.getElementById(`${containerId}-w1`).value;
    const w2 = +document.getElementById(`${containerId}-w2`).value;
    const b  = +document.getElementById(`${containerId}-b`).value;
    document.getElementById(`${containerId}-w1-v`).textContent = w1.toFixed(2);
    document.getElementById(`${containerId}-w2-v`).textContent = w2.toFixed(2);
    document.getElementById(`${containerId}-b-v`).textContent  = b.toFixed(2);

    const svg = document.getElementById(`${containerId}-svg`);
    let html = '';

    // shade decision regions on a coarse grid
    const cell = 8;
    for (let py = 0; py < H; py += cell) {
      for (let px = 0; px < W; px += cell) {
        const x = xMin + (px - 40) / (W - 60) * (xMax - xMin);
        const y = yMin + (H - 40 - py) / (H - 60) * (yMax - yMin);
        const p = predict(x, y, w1, w2, b);
        const fill = p === 1 ? 'rgba(37, 99, 235, 0.08)' : 'rgba(220, 38, 38, 0.08)';
        html += `<rect x="${px}" y="${py}" width="${cell}" height="${cell}" fill="${fill}"/>`;
      }
    }

    // axes
    html += `<line x1="${xToPx(xMin)}" y1="${yToPx(0)}" x2="${xToPx(xMax)}" y2="${yToPx(0)}" stroke="#9ca3af" stroke-width="1"/>`;
    html += `<line x1="${xToPx(0)}" y1="${yToPx(yMin)}" x2="${xToPx(0)}" y2="${yToPx(yMax)}" stroke="#9ca3af" stroke-width="1"/>`;
    html += `<text x="${W-8}" y="${yToPx(0)-6}" font-size="11" fill="#6b7280" text-anchor="end">${featureLabels[0]}</text>`;
    html += `<text x="${xToPx(0)+6}" y="18" font-size="11" fill="#6b7280">${featureLabels[1]}</text>`;
    html += `<text x="${xToPx(0)-10}" y="${yToPx(0)+14}" font-size="10" fill="#6b7280" text-anchor="end">0</text>`;
    html += `<text x="${xToPx(1)}" y="${yToPx(0)+14}" font-size="10" fill="#6b7280" text-anchor="middle">1</text>`;
    html += `<text x="${xToPx(0)-10}" y="${yToPx(1)+4}" font-size="10" fill="#6b7280" text-anchor="end">1</text>`;

    // decision line: w1*x + w2*y + b = 0
    if (Math.abs(w2) > 0.001) {
      const y1plot = -(w1*xMin + b) / w2;
      const y2plot = -(w1*xMax + b) / w2;
      html += `<line x1="${xToPx(xMin)}" y1="${yToPx(y1plot)}" x2="${xToPx(xMax)}" y2="${yToPx(y2plot)}" stroke="#1f2937" stroke-width="3" stroke-dasharray="6,4"/>`;
    } else if (Math.abs(w1) > 0.001) {
      const xLine = -b / w1;
      html += `<line x1="${xToPx(xLine)}" y1="${yToPx(yMin)}" x2="${xToPx(xLine)}" y2="${yToPx(yMax)}" stroke="#1f2937" stroke-width="3" stroke-dasharray="6,4"/>`;
    }

    // data points
    let correct = 0;
    for (const [[x, y], truth] of data) {
      const pred = predict(x, y, w1, w2, b);
      const ok = pred === truth;
      if (ok) correct++;
      const truthColor = truth ? '#2563eb' : '#dc2626';
      const fillColor  = ok ? '#22c55e' : truthColor;
      const ringColor  = ok ? '#16a34a' : '#f97316';
      html += `<circle cx="${xToPx(x)}" cy="${yToPx(y)}" r="16" fill="${fillColor}" stroke="${ringColor}" stroke-width="4"/>`;
      html += `<text x="${xToPx(x)}" y="${yToPx(y)+5}" text-anchor="middle" font-size="13" fill="white" font-weight="700">${truth ? 'Y' : 'N'}</text>`;
    }

    svg.innerHTML = html;
    const cEl = document.getElementById(`${containerId}-correct`);
    cEl.textContent = correct;
    cEl.style.color = correct === data.length
      ? 'var(--good)'
      : correct === data.length - 1 ? 'var(--warn)' : 'var(--bad)';
    const msgEl = document.getElementById(`${containerId}-msg`);
    if (correct === data.length) msgEl.textContent = '✓ all separated by one line!';
    else if (correct === data.length - 1) msgEl.textContent = nearMiss;
    else msgEl.textContent = '';
    msgEl.hidden = !msgEl.textContent;
  }

  ['w1','w2','b'].forEach(name => {
    document.getElementById(`${containerId}-${name}`).addEventListener('input', render);
  });
  render();
}

