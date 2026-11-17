(function convolutionChapter() {
  const N = 28;

  const net = (function networkMath() {
    function unpack(t) {
      if (t.values) return Float32Array.from(t.values);
      const raw = atob(t.q), out = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) out[i] = ((raw.charCodeAt(i) << 24) >> 24) * t.scale;
      return out;
    }

    function load(file) {
      const read = group => Object.fromEntries(Object.entries(group).map(([name, t]) => [name, unpack(t)]));
      const conv = read(file.conv), plain = read(file.plain);
      [conv.filters1, , conv.k1] = file.conv['conv1.weight'].shape;
      [conv.filters2, , conv.k2] = file.conv['conv2.weight'].shape;
      plain.hidden = file.plain['hidden.weight'].shape[0];
      const samples = {};
      for (const [digit, pixels] of Object.entries(file.samples)) samples[digit] = Float32Array.from(atob(pixels), ch => ch.charCodeAt(0) / 255);
      return { conv, plain, samples, measured: file.measured };
    }

    function convRelu(x, channels, size, w, b, filters, k) {
      const pad = (k - 1) / 2, wide = size + 2 * pad;
      const framed = new Float32Array(channels * wide * wide), out = new Float32Array(filters * size * size);
      for (let ch = 0; ch < channels; ch++) {
        for (let r = 0; r < size; r++) {
          framed.set(x.subarray((ch * size + r) * size, (ch * size + r + 1) * size), (ch * wide + r + pad) * wide + pad);
        }
      }
      for (let f = 0; f < filters; f++) {
        for (let r = 0; r < size; r++) {
          for (let c = 0; c < size; c++) {
            let sum = b[f];
            for (let ch = 0; ch < channels; ch++) {
              for (let u = 0; u < k; u++) {
                const xRow = (ch * wide + r + u) * wide + c, wRow = ((f * channels + ch) * k + u) * k;
                for (let v = 0; v < k; v++) sum += framed[xRow + v] * w[wRow + v];
              }
            }
            out[(f * size + r) * size + c] = sum > 0 ? sum : 0;
          }
        }
      }
      return out;
    }

    function maxPool(x, channels, size) {
      const half = size / 2, out = new Float32Array(channels * half * half);
      for (let ch = 0; ch < channels; ch++) {
        for (let r = 0; r < half; r++) {
          for (let c = 0; c < half; c++) {
            const i = (ch * size + 2 * r) * size + 2 * c;
            out[(ch * half + r) * half + c] = Math.max(x[i], x[i + 1], x[i + size], x[i + size + 1]);
          }
        }
      }
      return out;
    }

    function dense(x, w, b, outputs) {
      const out = new Float32Array(outputs);
      for (let o = 0; o < outputs; o++) {
        let sum = b[o];
        for (let i = 0; i < x.length; i++) sum += x[i] * w[o * x.length + i];
        out[o] = sum;
      }
      return out;
    }

    function softmax(scores) {
      const top = Math.max(...scores), e = Array.from(scores, s => Math.exp(s - top));
      const total = e.reduce((a, v) => a + v, 0);
      return e.map(v => v / total);
    }

    function runConv(m, x) {
      const half = N / 2, area = half * half;
      const maps1 = convRelu(x, 1, N, m['conv1.weight'], m['conv1.bias'], m.filters1, m.k1);
      const pooled = maxPool(maps1, m.filters1, N);
      const maps2 = convRelu(pooled, m.filters1, half, m['conv2.weight'], m['conv2.bias'], m.filters2, m.k2);
      const peaks = new Float32Array(m.filters2), peakAt = new Int32Array(m.filters2);
      for (let f = 0; f < m.filters2; f++) {
        for (let i = 0; i < area; i++) {
          if (maps2[f * area + i] > peaks[f]) { peaks[f] = maps2[f * area + i]; peakAt[f] = i; }
        }
      }
      const scores = dense(peaks, m['out.weight'], m['out.bias'], 10);
      return { maps1, pooled, maps2, peaks, peakAt, scores, probs: softmax(scores) };
    }

    function runPlain(m, x) {
      const hidden = dense(x, m['hidden.weight'], m['hidden.bias'], m.hidden).map(v => Math.max(v, 0));
      const scores = dense(hidden, m['out.weight'], m['out.bias'], 10);
      return { scores, probs: softmax(scores) };
    }

    function slide(x, kernel) {
      const side = N - 2, out = new Float32Array(side * side);
      for (let r = 0; r < side; r++) {
        for (let c = 0; c < side; c++) {
          let sum = 0;
          for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) sum += x[(r + u) * N + c + v] * kernel[u * 3 + v];
          out[r * side + c] = sum;
        }
      }
      return out;
    }

    function shift(x, dx, dy) {
      const out = new Float32Array(N * N);
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          const fromR = r - dy, fromC = c - dx;
          if (fromR >= 0 && fromR < N && fromC >= 0 && fromC < N) out[r * N + c] = x[fromR * N + fromC];
        }
      }
      return out;
    }

    function tidy(ink, size) {
      let top = size, left = size, bottom = -1, right = -1, mass = 0, mx = 0, my = 0;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const v = ink[y * size + x];
          if (v < 0.1) continue;
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
          mass += v;
          mx += v * (x + 0.5);
          my += v * (y + 0.5);
        }
      }
      if (bottom < 0) return null;
      const span = Math.max(right - left + 1, bottom - top + 1) / 20;
      const cx = mx / mass, cy = my / mass, out = new Float32Array(N * N);
      for (let ty = 0; ty < N; ty++) {
        // MNIST digits balance on the middle of pixel 14 (measured on the 60,000 training digits), not on the seam between 13 and 14.
        const y0 = cy + (ty - 14.5) * span, y1 = y0 + span;
        for (let tx = 0; tx < N; tx++) {
          const x0 = cx + (tx - 14.5) * span, x1 = x0 + span;
          let sum = 0;
          for (let y = Math.max(0, Math.floor(y0)); y < Math.min(size, y1); y++) {
            const tall = Math.min(y + 1, y1) - Math.max(y, y0);
            for (let x = Math.max(0, Math.floor(x0)); x < Math.min(size, x1); x++) {
              sum += ink[y * size + x] * tall * (Math.min(x + 1, x1) - Math.max(x, x0));
            }
          }
          out[ty * N + tx] = Math.min(1, sum / (span * span));
        }
      }
      return out;
    }

    return { load, runConv, runPlain, slide, shift, tidy };
  })();

  // tools/check_convnet.js loads this file under node to test the exact code the page runs.
  if (typeof module === 'object') module.exports = net;
  if (typeof document === 'undefined') return;

  const $ = id => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const BLUE = [37, 99, 235], RED = [220, 38, 38], PAPER = [255, 255, 255], INK = [27, 31, 39], FRAME = '#ea580c';
  const mix = (from, to, t) => from.map((v, i) => Math.round(v + (to[i] - v) * t));
  const signed = v => mix(PAPER, v >= 0 ? BLUE : RED, Math.pow(Math.min(1, Math.abs(v)), 0.7));
  const inked = v => mix(PAPER, INK, clamp(v, 0, 1));
  const css = ([r, g, b]) => `rgb(${r},${g},${b})`;
  const pct = p => Math.round(p * 100) + '%';

  const scratch = document.createElement('canvas');
  function paint(ctx, values, offset, cols, rows, x, y, w, h, color) {
    scratch.width = cols;
    scratch.height = rows;
    const sctx = scratch.getContext('2d'), img = sctx.createImageData(cols, rows);
    for (let i = 0; i < cols * rows; i++) {
      const rgb = color(values[offset + i]);
      img.data[i * 4] = rgb[0];
      img.data[i * 4 + 1] = rgb[1];
      img.data[i * 4 + 2] = rgb[2];
      img.data[i * 4 + 3] = 255;
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratch, x, y, w, h);
  }

  function showSeen(canvas, image) {
    const { ctx, w, h } = Book.hidpi(canvas);
    paint(ctx, image || new Float32Array(N * N), 0, N, N, 0, 0, w, h, inked);
  }

  function showAnswer(el, probs, truth) {
    const best = probs ? probs.indexOf(Math.max(...probs)) : null;
    el.classList.toggle('cv-right', best !== null && truth !== null && best === truth);
    el.classList.toggle('cv-wrong', best !== null && truth !== null && best !== truth);
    el.querySelector('.cv-digit').textContent = best === null ? '?' : best;
    el.querySelector('.cv-sure').textContent = best === null ? 'draw a digit' : (probs[best] > 0.995 ? 'over 99%' : pct(probs[best])) + ' sure';
  }

  function showBars(el, probs) {
    if (!el.firstChild) {
      el.innerHTML = Array.from({ length: 10 }, (_, d) =>
        `<div class="bar-container"><span class="label">${d}</span><div class="bar-track"><div class="bar"></div></div><span class="pct"></span></div>`).join('');
    }
    const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
    [...el.children].forEach((row, d) => {
      const bar = row.querySelector('.bar');
      bar.style.width = (probs ? probs[d] * 100 : 0) + '%';
      bar.classList.toggle('cv-top', d === best);
      row.querySelector('.pct').textContent = probs ? pct(probs[d]) : '';
    });
  }

  function makePad(canvas, samplesEl, onChange) {
    const size = +canvas.getAttribute('width'), pen = size * 22 / 280, fit = size * 160 / 280;
    const shadow = document.createElement('canvas');
    shadow.width = shadow.height = size;
    const shadowCtx = shadow.getContext('2d', { willReadFrequently: true });
    let strokes = [], active = null, loaded = null, queued = false, samples = {};

    function trace(ctx, place) {
      ctx.strokeStyle = ctx.fillStyle = css(INK);
      ctx.lineWidth = pen;
      ctx.lineCap = ctx.lineJoin = 'round';
      for (const points of strokes) {
        const [x0, y0] = place(points[0]);
        ctx.beginPath();
        ctx.arc(x0, y0, pen / 2, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        for (const p of points) ctx.lineTo(...place(p));
        ctx.stroke();
      }
    }

    // The strokes are first redrawn at one standard size, as in the digits chapter, so the pen
    // is as thick next to the digit as the pens in MNIST were, however small the reader drew.
    function image() {
      if (loaded) return loaded.pixels;
      if (!strokes.length) return null;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const p of strokes.flat()) {
        x0 = Math.min(x0, p.x);
        x1 = Math.max(x1, p.x);
        y0 = Math.min(y0, p.y);
        y1 = Math.max(y1, p.y);
      }
      const grow = fit / Math.max(x1 - x0, y1 - y0, fit / 8), midX = (x0 + x1) / 2, midY = (y0 + y1) / 2;
      shadowCtx.clearRect(0, 0, size, size);
      trace(shadowCtx, p => [size / 2 + (p.x - midX) * grow, size / 2 + (p.y - midY) * grow]);
      const rgba = shadowCtx.getImageData(0, 0, size, size).data, ink = new Float32Array(size * size);
      for (let i = 0; i < ink.length; i++) ink[i] = rgba[i * 4 + 3] / 255;
      return net.tidy(ink, size);
    }

    function refresh() {
      queued = false;
      const { ctx } = Book.hidpi(canvas);
      if (loaded) {
        paint(ctx, loaded.pixels, 0, N, N, 0, 0, size, size, inked);
      } else {
        ctx.fillStyle = css(PAPER);
        ctx.fillRect(0, 0, size, size);
        trace(ctx, p => [p.x, p.y]);
      }
      if (!loaded && !strokes.length) {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '15px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('draw a digit here', size / 2, size / 2);
      }
      onChange(image(), loaded ? loaded.digit : null);
    }

    function queue() {
      if (queued) return;
      queued = true;
      requestAnimationFrame(refresh);
    }

    function show(digit) {
      strokes = [];
      loaded = { digit, pixels: samples[digit] };
      queue();
    }

    function offer(digits, first) {
      samples = digits;
      samplesEl.insertAdjacentHTML('beforeend', '<span>or load a test digit:</span>' +
        Object.keys(digits).map(d => `<button class="secondary" data-digit="${d}">${d}</button>`).join(''));
      if (!strokes.length) show(first);
    }

    canvas.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (loaded) loaded = null;
      active = [Book.pointer(canvas, e)];
      strokes.push(active);
      queue();
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
      if (!active) return;
      active.push(Book.pointer(canvas, e));
      queue();
    });
    const lift = () => { active = null; };
    canvas.addEventListener('pointerup', lift);
    canvas.addEventListener('pointercancel', lift);

    samplesEl.innerHTML = '<button class="secondary cv-wide" data-clear>clear</button>';
    samplesEl.addEventListener('click', e => {
      const btn = e.target.closest('button');
      if (!btn) return;
      if (btn.dataset.digit) return show(+btn.dataset.digit);
      strokes = [];
      loaded = null;
      queue();
    });

    queue();
    return { offer };
  }

  let networks = null;
  function loadNetworks() {
    if (!networks) {
      networks = fetch('py/convnet-weights.json')
        .then(res => {
          if (!res.ok) throw new Error('convnet-weights.json: ' + res.status);
          return res.json();
        })
        .then(net.load);
    }
    return networks;
  }
  const LOAD_FAILED = 'Could not load the trained networks (py/convnet-weights.json). If you opened this page as a file, run "make serve" in the book folder and open http://localhost:8000 instead.';

  (function slideWidget() {
    const inCanvas = $('sf-in');
    if (!inCanvas) return;
    const outCanvas = $('sf-out'), playBtn = $('sf-play'), kernelEl = $('sf-kernel');
    const CELL = 10, OUT = N - 2, ALL = OUT * OUT, PLAY_LABEL = playBtn.textContent, FIRST_ROW_SECONDS = 1.8, REST_SECONDS = 3.2;
    const PRESETS = {
      vertical: [-1, 0, 1, -1, 0, 1, -1, 0, 1],
      horizontal: [-1, -1, -1, 0, 0, 0, 1, 1, 1],
      blur: Array(9).fill(1 / 9),
      sharpen: [0, -1, 0, -1, 5, -1, 0, -1, 0],
      outline: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
      nothing: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    };
    const SHAPES = {
      block: (r, c) => (r >= 8 && r < 20 && c >= 8 && c < 20 ? 1 : 0),
      ring: (r, c) => clamp(1.6 - Math.abs(Math.hypot(r - 13.5, c - 13.5) - 7.5), 0, 1),
      slash: (r, c) => (r > 4 && r < 23 ? clamp(1.5 - Math.abs(r + c - 27) / 1.6, 0, 1) : 0),
    };
    const num = v => String(Math.round(v * 100) / 100).replace('-', '−');

    kernelEl.innerHTML = Array.from({ length: 9 }, (_, i) =>
      `<input type="number" step="0.5" inputmode="decimal" aria-label="filter weight, row ${Math.floor(i / 3) + 1}, column ${i % 3 + 1}">`).join('');
    const fields = [...kernelEl.querySelectorAll('input')];

    let picture = new Float32Array(N * N), kernel = Float32Array.from(PRESETS.vertical), samples = {}, picked = '4';
    let output, peak, chosen = { r: 6, c: 9 }, shown = ALL, run = 0, painting = false, last = null;

    function render() {
      const input = Book.hidpi(inCanvas).ctx;
      paint(input, picture, 0, N, N, 0, 0, N * CELL, N * CELL, inked);
      input.strokeStyle = FRAME;
      input.lineWidth = 2.5;
      input.strokeRect(chosen.c * CELL, chosen.r * CELL, 3 * CELL, 3 * CELL);

      const out = Book.hidpi(outCanvas);
      out.ctx.fillStyle = '#e5e7eb';
      out.ctx.fillRect(0, 0, out.w, out.h);
      const visible = output.map((v, i) => (i < shown ? v / peak : NaN));
      paint(out.ctx, visible, 0, OUT, OUT, CELL, CELL, OUT * CELL, OUT * CELL, v => (Number.isNaN(v) ? [243, 244, 246] : signed(v)));
      out.ctx.strokeStyle = FRAME;
      out.ctx.lineWidth = 2.5;
      out.ctx.strokeRect((chosen.c + 1) * CELL, (chosen.r + 1) * CELL, CELL, CELL);

      const strongest = Math.max(1e-6, ...kernel.map(Math.abs));
      let terms = '';
      for (let u = 0; u < 3; u++) {
        for (let v = 0; v < 3; v++) {
          const pixel = picture[(chosen.r + u) * N + chosen.c + v], product = pixel * kernel[u * 3 + v];
          terms += `<div class="cv-term" style="background:${css(signed(0.35 * product / strongest))}">${pixel.toFixed(2)} × ${num(kernel[u * 3 + v])}<b>= ${num(product)}</b></div>`;
        }
      }
      const total = output[chosen.r * OUT + chosen.c];
      $('sf-where').textContent = `row ${chosen.r + 1}, column ${chosen.c + 1}`;
      $('sf-terms').innerHTML = terms;
      $('sf-sum').innerHTML = `add the nine products → <strong style="background:${css(signed(0.6 * total / peak))}">${num(total)}</strong>`;
    }

    function recompute() {
      output = net.slide(picture, kernel);
      peak = Math.max(1e-6, ...output.map(Math.abs));
      render();
    }

    function stop() {
      run++;
      shown = ALL;
      playBtn.textContent = PLAY_LABEL;
    }

    function play() {
      if (shown < ALL) {
        stop();
        render();
        return;
      }
      const mine = ++run, before = chosen, began = performance.now();
      playBtn.textContent = '■ stop';
      (function frame() {
        if (mine !== run) return;
        const t = (performance.now() - began) / 1000;
        const cells = Math.floor(t < FIRST_ROW_SECONDS ? 1 + OUT * t / FIRST_ROW_SECONDS : OUT + (ALL - OUT) * (t - FIRST_ROW_SECONDS) / REST_SECONDS);
        if (cells >= ALL) {
          chosen = before;
          stop();
          render();
          return;
        }
        shown = cells;
        chosen = { r: Math.floor((cells - 1) / OUT), c: (cells - 1) % OUT };
        render();
        requestAnimationFrame(frame);
      })();
    }

    function setKernel(values) {
      kernel = Float32Array.from(values);
      const strongest = Math.max(1e-6, ...kernel.map(Math.abs));
      fields.forEach((field, i) => {
        if (document.activeElement !== field) field.value = Math.round(kernel[i] * 100) / 100;
        field.style.background = css(signed(0.55 * kernel[i] / strongest));
      });
      stop();
      recompute();
    }

    function setPicture(name) {
      picked = name;
      painting = name === 'draw';
      inCanvas.classList.toggle('cv-paintable', painting);
      $('sf-hint').textContent = painting
        ? 'Drag on the blank picture to draw. Then point at the output map.'
        : 'Point at either picture (or tap) to pick a spot.';
      if (painting) picture = new Float32Array(N * N);
      else if (SHAPES[name]) picture = Float32Array.from({ length: N * N }, (_, i) => SHAPES[name](Math.floor(i / N), i % N));
      else picture = samples[name] || new Float32Array(N * N);
      stop();
      recompute();
    }

    function pick(canvas, e) {
      if (shown < ALL) return;
      const p = Book.pointer(canvas, e);
      chosen = { r: clamp(Math.floor(p.y / CELL) - 1, 0, OUT - 1), c: clamp(Math.floor(p.x / CELL) - 1, 0, OUT - 1) };
      render();
    }

    function dab(p) {
      const pr = p.y / CELL - 0.5, pc = p.x / CELL - 0.5;
      for (let r = Math.max(0, Math.floor(pr - 2)); r <= Math.min(N - 1, pr + 2); r++) {
        for (let c = Math.max(0, Math.floor(pc - 2)); c <= Math.min(N - 1, pc + 2); c++) {
          const ink = clamp(1.7 - Math.hypot(r - pr, c - pc), 0, 1);
          if (ink > picture[r * N + c]) picture[r * N + c] = ink;
        }
      }
    }

    function drag(e) {
      const p = Book.pointer(inCanvas, e), steps = Math.max(1, Math.ceil(Math.hypot(p.x - last.x, p.y - last.y) / (CELL / 2)));
      for (let s = 1; s <= steps; s++) dab({ x: last.x + (p.x - last.x) * s / steps, y: last.y + (p.y - last.y) * s / steps });
      last = p;
      stop();
      recompute();
    }

    inCanvas.addEventListener('pointerdown', e => {
      if (!painting) return pick(inCanvas, e);
      e.preventDefault();
      last = Book.pointer(inCanvas, e);
      drag(e);
      inCanvas.setPointerCapture(e.pointerId);
    });
    inCanvas.addEventListener('pointermove', e => {
      if (!painting) return pick(inCanvas, e);
      if (last) drag(e);
    });
    const lift = () => { last = null; };
    inCanvas.addEventListener('pointerup', lift);
    inCanvas.addEventListener('pointercancel', lift);
    outCanvas.addEventListener('pointermove', e => pick(outCanvas, e));
    outCanvas.addEventListener('pointerdown', e => pick(outCanvas, e));

    function choose(row, attr, apply) {
      row.addEventListener('click', e => {
        const tog = e.target.closest('.toggle');
        if (!tog) return;
        row.querySelectorAll('.toggle').forEach(t => t.classList.toggle('on', t === tog));
        apply(tog.getAttribute(attr));
      });
    }
    choose($('sf-pictures'), 'data-pic', setPicture);
    choose($('sf-presets'), 'data-preset', name => setKernel(PRESETS[name]));
    kernelEl.addEventListener('input', () => {
      $('sf-presets').querySelectorAll('.toggle').forEach(t => t.classList.remove('on'));
      setKernel(fields.map(field => parseFloat(field.value) || 0));
    });
    playBtn.addEventListener('click', play);

    setKernel(PRESETS.vertical);
    loadNetworks().then(loaded => {
      samples = loaded.samples;
      if (samples[picked]) setPicture(picked);
    }, () => {
      const row = $('sf-pictures');
      row.querySelectorAll('.toggle').forEach(t => {
        t.hidden = /^\d$/.test(t.dataset.pic);
        t.classList.toggle('on', t.dataset.pic === 'block');
      });
      setPicture('block');
    });
  })();

  (function filtersWidget() {
    const padCanvas = $('fl-pad');
    if (!padCanvas) return;
    const WIDTH = 640;
    let model = null, current = null, truth = null;

    function tiles(canvas, values, count, side, strongest, perRow, gap) {
      const tile = (WIDTH - (perRow - 1) * gap) / perRow;
      const at = i => [(i % perRow) * (tile + gap), Math.floor(i / perRow) * (tile + gap)];
      // Book.hidpi reads the height attribute once, on its first call, so it has to be right before then.
      if (!canvas.dataset.w) canvas.setAttribute('height', Math.ceil(Math.ceil(count / perRow) * (tile + gap) - gap));
      const { ctx, w, h } = Book.hidpi(canvas);
      ctx.clearRect(0, 0, w, h);
      for (let i = 0; i < count; i++) {
        const [x, y] = at(i);
        paint(ctx, values, i * side * side, side, side, x, y, tile, tile, v => signed(v / strongest));
        ctx.strokeStyle = 'rgba(17, 24, 39, 0.18)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, tile - 1, tile - 1);
      }
      return { ctx, tile, at };
    }

    function render() {
      showSeen($('fl-seen'), current);
      if (!model) return;
      const m = model.conv, half = N / 2;
      const run = current && net.runConv(m, current);
      showAnswer($('fl-answer'), run && run.probs, truth);
      showBars($('fl-bars'), run && run.probs);
      const quiet = new Float32Array(m.filters2 * half * half);
      const maps1 = run ? run.maps1 : new Float32Array(m.filters1 * N * N);
      const strongest1 = Math.max(1e-6, ...maps1);
      tiles($('fl-maps'), maps1, m.filters1, N, strongest1, 8, 8);
      tiles($('fl-pooled'), run ? run.pooled : quiet, m.filters1, half, strongest1, 8, 8);
      const maps2 = run ? run.maps2 : quiet;
      const second = tiles($('fl-maps2'), maps2, m.filters2, half, Math.max(1e-6, ...maps2), 16, 6);
      if (!run) return;
      const cell = second.tile / half;
      second.ctx.strokeStyle = FRAME;
      second.ctx.lineWidth = 1.5;
      for (let f = 0; f < m.filters2; f++) {
        if (run.peaks[f] <= 0) continue;
        const [x, y] = second.at(f), at = run.peakAt[f];
        second.ctx.strokeRect(x + (at % half) * cell - 1, y + Math.floor(at / half) * cell - 1, cell + 2, cell + 2);
      }
    }

    const pad = makePad(padCanvas, $('fl-samples'), (image, digit) => {
      current = image;
      truth = digit;
      render();
    });

    loadNetworks().then(loaded => {
      model = loaded;
      pad.offer(model.samples, 5);
      const w = model.conv['conv1.weight'];
      tiles($('fl-filters'), w, model.conv.filters1, model.conv.k1, Math.max(...w.map(Math.abs)), 8, 8);
      $('fl-status').textContent = '';
      render();
    }, () => { $('fl-status').textContent = LOAD_FAILED; });
  })();

  (function shiftWidget() {
    const padCanvas = $('sh-pad');
    if (!padCanvas) return;
    const slider = $('sh-shift'), dirs = $('sh-dir');
    const DIRS = { right: [1, 0], left: [-1, 0], down: [0, 1], up: [0, -1] };
    let model = null, current = null, truth = null, dir = 'right';

    function render() {
      const k = +slider.value, [dx, dy] = DIRS[dir];
      $('sh-shift-v').textContent = k + ' px';
      const shifted = current && net.shift(current, dx * k, dy * k);
      showSeen($('sh-seen'), shifted);
      if (!model) return;
      for (const kind of ['plain', 'conv']) {
        const run = shifted && (kind === 'plain' ? net.runPlain(model.plain, shifted) : net.runConv(model.conv, shifted));
        showAnswer($(`sh-${kind}-answer`), run && run.probs, truth);
        showBars($(`sh-${kind}-bars`), run && run.probs);
        const right = model.measured[kind][dir][k], total = model.measured.digits;
        $(`sh-${kind}-measured`).innerHTML = `All ${total.toLocaleString()} test digits, ${k ? `pushed ${k} px ${dir}` : 'not pushed'}: ` +
          `<strong>${(100 * right / total).toFixed(1)}%</strong> right`;
      }
    }

    const pad = makePad(padCanvas, $('sh-samples'), (image, digit) => {
      current = image;
      truth = digit;
      render();
    });

    slider.addEventListener('input', render);
    dirs.addEventListener('click', e => {
      const tog = e.target.closest('[data-dir]');
      if (!tog) return;
      dir = tog.dataset.dir;
      dirs.querySelectorAll('.toggle').forEach(t => t.classList.toggle('on', t === tog));
      render();
    });

    loadNetworks().then(loaded => {
      model = loaded;
      pad.offer(model.samples, 3);
      $('sh-status').textContent = '';
      render();
    }, () => { $('sh-status').textContent = LOAD_FAILED; });
  })();
})();
