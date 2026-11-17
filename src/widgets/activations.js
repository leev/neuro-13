// ===== Interactive: activation function plot =====
(function() {
  const svg = document.getElementById('act-plot');
  if (!svg) return;
  const W = 600, H = 320, pad = 40;
  const xMin = -6, xMax = 6, yMin = -0.4, yMax = 1.4;
  function xToPx(x) { return pad + ((x - xMin) / (xMax - xMin)) * (W - 2*pad); }
  function yToPx(y) { return H - pad - ((y - yMin) / (yMax - yMin)) * (H - 2*pad); }

  let activation = 'sigmoid';
  function f(x) {
    if (activation === 'sigmoid') return 1 / (1 + Math.exp(-x));
    return Math.max(0, x);
  }
  function fp(x) {
    if (activation === 'sigmoid') {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    }
    return x > 0 ? 1 : 0;
  }
  function pathFor(fn, color, width) {
    let d = '';
    for (let px = pad; px <= W - pad; px += 2) {
      const x = xMin + (px - pad) / (W - 2*pad) * (xMax - xMin);
      const y = fn(x);
      const py = yToPx(y);
      d += (px === pad ? 'M' : 'L') + px + ',' + py + ' ';
    }
    return `<path d="${d}" stroke="${color}" stroke-width="${width}" fill="none"/>`;
  }

  function render() {
    const x = +document.getElementById('act-x').value;
    const y = f(x);
    const d = fp(x);
    const dotX = xToPx(x);
    const dotY = yToPx(y);
    const angle = Math.atan(d * ((H - 2*pad)/(yMax - yMin)) / ((W - 2*pad)/(xMax - xMin)));
    const tanLen = 80;
    const tx = Math.cos(angle) * tanLen;
    const ty = Math.sin(angle) * tanLen;

    let html = '';
    // gridlines
    html += `<line x1="${pad}" y1="${yToPx(0)}" x2="${W-pad}" y2="${yToPx(0)}" stroke="#9ca3af" stroke-width="1"/>`;
    html += `<line x1="${xToPx(0)}" y1="${pad}" x2="${xToPx(0)}" y2="${H-pad}" stroke="#9ca3af" stroke-width="1"/>`;
    html += `<line x1="${pad}" y1="${yToPx(1)}" x2="${W-pad}" y2="${yToPx(1)}" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4,4"/>`;
    // labels
    html += `<text x="${W-pad+4}" y="${yToPx(0)+4}" font-size="11" fill="#6b7280">x</text>`;
    html += `<text x="${xToPx(0)+6}" y="${pad-2}" font-size="11" fill="#6b7280">y</text>`;
    html += `<text x="${xToPx(0)-14}" y="${yToPx(1)+4}" font-size="11" fill="#6b7280">1</text>`;
    html += `<text x="${xToPx(0)-14}" y="${yToPx(0)+12}" font-size="11" fill="#6b7280">0</text>`;
    // curves
    html += pathFor(f, '#2563eb', 3);
    html += pathFor(fp, '#f97316', 2.5);
    // tangent
    html += `<line x1="${dotX - tx}" y1="${dotY + ty}" x2="${dotX + tx}" y2="${dotY - ty}" stroke="#f97316" stroke-width="2" stroke-dasharray="5,3"/>`;
    // dot
    html += `<circle cx="${dotX}" cy="${dotY}" r="8" fill="#dc2626" stroke="#7f1d1d" stroke-width="2"/>`;
    // legend
    html += `<g><rect x="${W-pad-160}" y="${pad}" width="160" height="40" fill="white" opacity="0.9" rx="4"/>`;
    html += `<line x1="${W-pad-150}" y1="${pad+12}" x2="${W-pad-130}" y2="${pad+12}" stroke="#2563eb" stroke-width="3"/>`;
    html += `<text x="${W-pad-124}" y="${pad+16}" font-size="12" fill="#1f2937">${activation}</text>`;
    html += `<line x1="${W-pad-150}" y1="${pad+30}" x2="${W-pad-130}" y2="${pad+30}" stroke="#f97316" stroke-width="3"/>`;
    html += `<text x="${W-pad-124}" y="${pad+34}" font-size="12" fill="#1f2937">slope (derivative)</text></g>`;

    svg.innerHTML = html;
    document.getElementById('act-x-v').textContent = x.toFixed(2);
    document.getElementById('act-y').textContent = y.toFixed(3);
    document.getElementById('act-d').textContent = d.toFixed(3);
  }

  document.getElementById('act-x').addEventListener('input', render);
  document.querySelectorAll('[data-act]').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('[data-act]').forEach(x => x.classList.remove('on'));
      t.classList.add('on');
      activation = t.getAttribute('data-act');
      render();
    });
  });
  render();
})();

