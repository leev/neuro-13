(function digitsWidget() {
  const SIDE = 28, PIXELS = SIDE * SIDE, HIDDEN = 64, DIGITS = 10;
  const EPOCHS = 10, BATCH = 32, LR = 0.2;
  const INK_BOX = 20, FAINT = 0.1;

  // Four pixels in five are blank paper; listing only the inked ones lets the
  // first layer skip every multiply-by-zero, which is most of the work.
  function inkLists(pixels, count, scale) {
    let inked = 0;
    for (let i = 0; i < pixels.length; i++) if (pixels[i]) inked++;
    const start = new Int32Array(count + 1), where = new Uint16Array(inked), value = new Float32Array(inked);
    let k = 0;
    for (let n = 0; n < count; n++) {
      const base = n * PIXELS;
      for (let i = 0; i < PIXELS; i++) {
        if (pixels[base + i]) { where[k] = i; value[k++] = pixels[base + i] * scale; }
      }
      start[n + 1] = k;
    }
    return { start, where, value };
  }

  function parseDigits(buffer) {
    const bytes = new Uint8Array(buffer), head = new DataView(buffer);
    if (bytes.length < 16 || String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== 'DGT1') {
      throw new Error('digits.bin is not the file this page expects');
    }
    const nTrain = head.getUint32(4, true), nTest = head.getUint32(8, true), n = nTrain + nTest;
    if (head.getUint32(12, true) !== SIDE || bytes.length !== 16 + n * (1 + PIXELS)) throw new Error('digits.bin is incomplete');
    const labels = bytes.subarray(16, 16 + n), pixels = bytes.subarray(16 + n);
    return { nTrain, nTest, labels, pixels, ...inkLists(pixels, n, 1 / 255) };
  }

  function scrambled(data, shuffle) {
    const where = new Uint16Array(data.where.length);
    for (let k = 0; k < where.length; k++) where[k] = shuffle[data.where[k]];
    return { ...data, where };
  }

  function makeNet(random) {
    const gauss = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    const w1 = new Float64Array(PIXELS * HIDDEN), w2 = new Float64Array(HIDDEN * DIGITS);
    for (let i = 0; i < w1.length; i++) w1[i] = gauss() * 0.01;
    for (let i = 0; i < w2.length; i++) w2[i] = gauss() * 0.1;
    return { w1, b1: new Float64Array(HIDDEN), w2, b2: new Float64Array(DIGITS) };
  }

  function makeWork() {
    return {
      g1: new Float64Array(PIXELS * HIDDEN), gb1: new Float64Array(HIDDEN),
      g2: new Float64Array(HIDDEN * DIGITS), gb2: new Float64Array(DIGITS),
      h: new Float64Array(HIDDEN), p: new Float64Array(DIGITS), dh: new Float64Array(HIDDEN),
    };
  }

  function forward(net, where, value, from, to, h, p) {
    const { w1, b1, w2, b2 } = net;
    for (let j = 0; j < HIDDEN; j++) h[j] = b1[j];
    for (let k = from; k < to; k++) {
      const row = where[k] * HIDDEN, v = value[k];
      for (let j = 0; j < HIDDEN; j++) h[j] += v * w1[row + j];
    }
    for (let c = 0; c < DIGITS; c++) p[c] = b2[c];
    for (let j = 0; j < HIDDEN; j++) {
      if (h[j] <= 0) { h[j] = 0; continue; }
      const row = j * DIGITS, a = h[j];
      for (let c = 0; c < DIGITS; c++) p[c] += a * w2[row + c];
    }
    let top = p[0], sum = 0;
    for (let c = 1; c < DIGITS; c++) if (p[c] > top) top = p[c];
    for (let c = 0; c < DIGITS; c++) { p[c] = Math.exp(p[c] - top); sum += p[c]; }
    for (let c = 0; c < DIGITS; c++) p[c] /= sum;
  }

  function batchGradient(net, data, order, from, to, work) {
    const { start, where, value, labels } = data, { w2 } = net;
    const { g1, gb1, g2, gb2, h, p, dh } = work;
    const share = 1 / (to - from);
    let loss = 0;
    g1.fill(0); gb1.fill(0); g2.fill(0); gb2.fill(0);
    for (let n = from; n < to; n++) {
      const ex = order[n], a = start[ex], b = start[ex + 1];
      forward(net, where, value, a, b, h, p);
      loss -= Math.log(p[labels[ex]] + 1e-300);
      p[labels[ex]] -= 1;
      for (let c = 0; c < DIGITS; c++) { p[c] *= share; gb2[c] += p[c]; }
      for (let j = 0; j < HIDDEN; j++) {
        if (h[j] <= 0) { dh[j] = 0; continue; }
        const row = j * DIGITS, act = h[j];
        let blame = 0;
        for (let c = 0; c < DIGITS; c++) { g2[row + c] += act * p[c]; blame += p[c] * w2[row + c]; }
        dh[j] = blame;
        gb1[j] += blame;
      }
      for (let k = a; k < b; k++) {
        const row = where[k] * HIDDEN, v = value[k];
        for (let j = 0; j < HIDDEN; j++) g1[row + j] += v * dh[j];
      }
    }
    return loss * share;
  }

  function step(net, work, lr) {
    const { w1, b1, w2, b2 } = net, { g1, gb1, g2, gb2 } = work;
    for (let i = 0; i < w1.length; i++) w1[i] -= lr * g1[i];
    for (let i = 0; i < w2.length; i++) w2[i] -= lr * g2[i];
    for (let j = 0; j < HIDDEN; j++) b1[j] -= lr * gb1[j];
    for (let c = 0; c < DIGITS; c++) b2[c] -= lr * gb2[c];
  }

  function trainSlice(net, data, order, from, to, work) {
    for (let at = from; at < to; at += BATCH) {
      batchGradient(net, data, order, at, Math.min(at + BATCH, to), work);
      step(net, work, LR);
    }
  }

  function best(p) {
    let top = 0;
    for (let c = 1; c < DIGITS; c++) if (p[c] > p[top]) top = c;
    return top;
  }

  function score(net, data, from, to, work, guess, sure) {
    const { start, where, value, labels } = data, { h, p } = work;
    let right = 0;
    for (let n = from; n < to; n++) {
      forward(net, where, value, start[n], start[n + 1], h, p);
      const top = best(p);
      if (top === labels[n]) right++;
      if (guess) { guess[n - from] = top; sure[n - from] = p[top]; }
    }
    return right / (to - from);
  }

  function read(net, image, work) {
    const { start, where, value } = inkLists(image, 1, 1);
    forward(net, where, value, 0, start[1], work.h, work.p);
    return work.p;
  }

  function shifted(image, dx, dy) {
    const out = new Float32Array(PIXELS);
    for (let y = 0; y < SIDE; y++) {
      for (let x = 0; x < SIDE; x++) {
        const sx = x - dx, sy = y - dy;
        if (sx >= 0 && sx < SIDE && sy >= 0 && sy < SIDE) out[y * SIDE + x] = image[sy * SIDE + sx];
      }
    }
    return out;
  }

  // Each of the 28 x 28 output pixels is the average of the zoom x zoom patch
  // of the drawing it covers; (cx, cy) lands mid-frame. Averaging makes the gray edges.
  function frame28(ink, size, cx, cy, zoom) {
    const stride = size + 1, total = new Float64Array(stride * stride);
    for (let y = 0; y < size; y++) {
      let run = 0;
      for (let x = 0; x < size; x++) {
        run += ink[y * size + x];
        total[(y + 1) * stride + x + 1] = total[y * stride + x + 1] + run;
      }
    }
    const inkAbove = (x, y) => {
      x = Math.min(size, Math.max(0, x)); y = Math.min(size, Math.max(0, y));
      const x0 = Math.min(size - 1, Math.floor(x)), y0 = Math.min(size - 1, Math.floor(y)), fx = x - x0, fy = y - y0;
      const at = y0 * stride + x0;
      return total[at] * (1 - fx) * (1 - fy) + total[at + 1] * fx * (1 - fy) +
        total[at + stride] * (1 - fx) * fy + total[at + stride + 1] * fx * fy;
    };
    const out = new Float32Array(PIXELS), half = SIDE / 2;
    for (let ty = 0; ty < SIDE; ty++) {
      for (let tx = 0; tx < SIDE; tx++) {
        const x0 = cx + (tx - half) * zoom, x1 = x0 + zoom, y0 = cy + (ty - half) * zoom, y1 = y0 + zoom;
        const sum = inkAbove(x1, y1) - inkAbove(x0, y1) - inkAbove(x1, y0) + inkAbove(x0, y0);
        out[ty * SIDE + tx] = Math.min(1, Math.max(0, sum / (zoom * zoom)));
      }
    }
    return out;
  }

  function inkBox(ink, size) {
    let x0 = size, y0 = size, x1 = -1, y1 = -1, mass = 0, mx = 0, my = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const v = ink[y * size + x];
        if (v < FAINT) continue;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
        mass += v; mx += v * (x + 0.5); my += v * (y + 0.5);
      }
    }
    if (x1 < 0) return null;
    return { x0, y0, x1: x1 + 1, y1: y1 + 1, cx: mx / mass, cy: my / mass };
  }

  // MNIST's own recipe: fit the ink in a 20 x 20 box without stretching it,
  // then put its center of mass in the middle of the 28 x 28 frame.
  function tidy(ink, size) {
    const box = inkBox(ink, size);
    if (!box) return { image: new Float32Array(PIXELS), box: null };
    const zoom = Math.max(box.x1 - box.x0, box.y1 - box.y0) / INK_BOX, reach = SIDE / 2 * zoom;
    const cx = Math.min(Math.max(box.cx, box.x1 - reach), box.x0 + reach);
    const cy = Math.min(Math.max(box.cy, box.y1 - reach), box.y0 + reach);
    return { image: frame28(ink, size, cx, cy, zoom), box };
  }

  const trainBtn = document.getElementById('dg-train');
  if (!trainBtn) return;
  const $ = id => document.getElementById(id);
  const INK = [27, 31, 39], BLUE = [37, 99, 235], RED = [220, 38, 38], ORANGE = '#ea580c';
  const PAD = 280, PEN = 22, FIT = 160, SLICE = 800, REACH = 6, SPAN = 2 * REACH + 1;
  const GALLERY = 20, WRONG_SHOWN = 36, PAIRS = 6, GAP = 2, GRID = 8, SHEET = GRID * (SIDE + GAP) + GAP, FAINT_TILE = 0.3;
  const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';
  const SAMPLES = [
    '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',
  ].map(text => Uint8Array.from(atob(text), ch => ch.charCodeAt(0)));

  const pct = x => (x * 100).toFixed(1) + '%';
  const whole = x => Math.round(x * 100) + '%';
  // A message, not setTimeout: timers in a background tab tick once a second,
  // which would stretch a five-second training run to nearly two minutes.
  const channel = new MessageChannel();
  const pause = () => new Promise(resolve => {
    channel.port1.onmessage = resolve;
    channel.port2.postMessage(0);
  });
  const work = makeWork();
  let data = null, net = null, busy = false, history = [], testGuess = null, testSure = null;

  const picture = n => data.pixels.subarray(n * PIXELS, (n + 1) * PIXELS);

  function bitmap(canvas, image, scale) {
    const ctx = canvas.getContext('2d'), shot = ctx.createImageData(SIDE, SIDE);
    for (let i = 0; i < PIXELS; i++) {
      shot.data[4 * i] = INK[0]; shot.data[4 * i + 1] = INK[1]; shot.data[4 * i + 2] = INK[2];
      shot.data[4 * i + 3] = Math.min(1, image[i] * scale) * 255;
    }
    ctx.putImageData(shot, 0, 0);
  }

  function weightPixel(rgba, at, v) {
    const tint = v > 0 ? BLUE : RED, strength = Math.min(1, Math.abs(v));
    for (let c = 0; c < 3; c++) rgba[at + c] = 255 + (tint[c] - 255) * strength;
    rgba[at + 3] = 255;
  }

  function weightBitmap(canvas, weightAt, full) {
    const ctx = canvas.getContext('2d'), shot = ctx.createImageData(SIDE, SIDE);
    if (!full) for (let i = 0; i < PIXELS; i++) full = Math.max(full || 1e-9, Math.abs(weightAt(i)));
    for (let i = 0; i < PIXELS; i++) weightPixel(shot.data, 4 * i, weightAt(i) / full);
    ctx.putImageData(shot, 0, 0);
  }

  function tile(image, scale, caption) {
    const fig = document.createElement('figure'), canvas = document.createElement('canvas'), cap = document.createElement('figcaption');
    fig.className = 'dg-tile';
    canvas.className = 'dg-bitmap';
    canvas.width = canvas.height = SIDE;
    if (image) bitmap(canvas, image, scale);
    cap.innerHTML = caption;
    fig.append(canvas, cap);
    return fig;
  }

  const look = { digit: 5, at: 9 * SIDE + 12, rows: SIDE, timer: 0 };
  const lookView = Book.hidpi($('dg-px')), magView = Book.hidpi($('dg-mag')), stripView = Book.hidpi($('dg-strip'));

  function drawPicture() {
    const image = SAMPLES[look.digit], { ctx, w } = lookView, cell = w / SIDE;
    const col = look.at % SIDE, row = (look.at - col) / SIDE;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, w);
    for (let i = 0; i < PIXELS; i++) {
      if (!image[i]) continue;
      ctx.fillStyle = `rgba(${INK}, ${image[i] / 255})`;
      ctx.fillRect((i % SIDE) * cell, Math.floor(i / SIDE) * cell, cell, cell);
    }
    ctx.strokeStyle = 'rgba(27, 31, 39, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 1; k < SIDE; k++) {
      ctx.moveTo(k * cell, 0); ctx.lineTo(k * cell, w);
      ctx.moveTo(0, k * cell); ctx.lineTo(w, k * cell);
    }
    ctx.stroke();
    if (look.rows < SIDE) {
      ctx.fillStyle = 'rgba(234, 88, 12, 0.3)';
      ctx.fillRect(0, look.rows * cell, w, cell);
      return;
    }
    ctx.strokeStyle = ORANGE;
    ctx.lineWidth = 1.5;
    ctx.strokeRect((col - 3) * cell, (row - 3) * cell, 7 * cell, 7 * cell);
    ctx.lineWidth = 2.5;
    ctx.strokeRect(col * cell, row * cell, cell, cell);
  }

  function drawMagnifier() {
    const image = SAMPLES[look.digit], { ctx, w } = magView, cell = w / 7;
    const col = look.at % SIDE, row = (look.at - col) / SIDE;
    ctx.clearRect(0, 0, w, w);
    ctx.font = `600 12px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const x = col + dx, y = row + dy, left = (dx + 3) * cell, top = (dy + 3) * cell;
        const inside = x >= 0 && y >= 0 && x < SIDE && y < SIDE, v = inside ? image[y * SIDE + x] : 0;
        ctx.fillStyle = inside ? '#fff' : '#e5e7eb';
        ctx.fillRect(left, top, cell, cell);
        if (!inside) continue;
        ctx.fillStyle = `rgba(${INK}, ${v / 255})`;
        ctx.fillRect(left, top, cell, cell);
        ctx.fillStyle = v > 120 ? '#fff' : v ? '#1b1f27' : '#9ca3af';
        ctx.fillText(v, left + cell / 2, top + cell / 2 + 1);
      }
    }
    ctx.strokeStyle = 'rgba(27, 31, 39, 0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let k = 1; k < 7; k++) {
      ctx.moveTo(k * cell, 0); ctx.lineTo(k * cell, w);
      ctx.moveTo(0, k * cell); ctx.lineTo(w, k * cell);
    }
    ctx.stroke();
    ctx.strokeStyle = ORANGE;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(3 * cell, 3 * cell, cell, cell);
  }

  function drawStrip() {
    const image = SAMPLES[look.digit], { ctx, w, h } = stripView, each = w / PIXELS, top = 20, tall = 26;
    ctx.clearRect(0, 0, w, h);
    for (let r = 0; r < SIDE; r++) {
      ctx.fillStyle = r % 2 ? '#e9ecf1' : '#fff';
      ctx.fillRect(r * SIDE * each, top, SIDE * each, tall);
    }
    for (let i = 0; i < look.rows * SIDE; i++) {
      if (!image[i]) continue;
      ctx.fillStyle = `rgba(${INK}, ${image[i] / 255})`;
      ctx.fillRect(i * each, top, each + 0.25, tall);
    }
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, top + 0.5, w - 1, tall - 1);
    ctx.font = `11px ${MONO}`;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'left';
    ctx.fillText('row 1 starts here', 0, top + tall + 10);
    ctx.textAlign = 'right';
    ctx.fillText('row 28 ends here', w, top + tall + 10);
    if (look.rows < SIDE) return;
    const x = (look.at + 0.5) * each;
    ctx.strokeStyle = ORANGE;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, top - 5); ctx.lineTo(x, top + tall + 4);
    ctx.stroke();
    ctx.fillStyle = ORANGE;
    ctx.textAlign = 'center';
    ctx.font = `600 11px ${MONO}`;
    ctx.fillText(`input ${look.at + 1}`, Math.min(w - 36, Math.max(36, x)), 8);
  }

  function drawLook() {
    const col = look.at % SIDE, row = (look.at - col) / SIDE, v = SAMPLES[look.digit][look.at];
    drawPicture();
    drawMagnifier();
    drawStrip();
    $('dg-px-row').textContent = row + 1;
    $('dg-px-col').textContent = col + 1;
    $('dg-px-ink').textContent = v;
    $('dg-px-fed').textContent = (v / 255).toFixed(2);
    $('dg-px-n').textContent = look.at + 1;
  }

  function unroll() {
    clearInterval(look.timer);
    look.rows = 0;
    drawLook();
    look.timer = setInterval(() => {
      look.rows++;
      drawLook();
      if (look.rows >= SIDE) clearInterval(look.timer);
    }, 55);
  }

  function lookAtPicture(event) {
    const p = Book.pointer($('dg-px'), event), cell = lookView.w / SIDE;
    const col = Math.min(SIDE - 1, Math.max(0, Math.floor(p.x / cell))), row = Math.min(SIDE - 1, Math.max(0, Math.floor(p.y / cell)));
    look.at = row * SIDE + col;
    drawLook();
  }

  function lookAtStrip(event) {
    const p = Book.pointer($('dg-strip'), event);
    look.at = Math.min(PIXELS - 1, Math.max(0, Math.floor(p.x / stripView.w * PIXELS)));
    drawLook();
  }

  SAMPLES.forEach((_, digit) => {
    const btn = document.createElement('button');
    btn.className = 'secondary dg-pick' + (digit === look.digit ? ' on' : '');
    btn.textContent = digit;
    btn.addEventListener('click', () => {
      look.digit = digit;
      $('dg-picks').querySelectorAll('.dg-pick').forEach(b => b.classList.toggle('on', b === btn));
      drawLook();
    });
    $('dg-picks').append(btn);
  });
  for (const type of ['pointermove', 'pointerdown']) {
    $('dg-px').addEventListener(type, lookAtPicture);
    $('dg-strip').addEventListener(type, lookAtStrip);
  }
  $('dg-unroll').addEventListener('click', unroll);

  async function download(res, onProgress) {
    if (!res.body) return res.arrayBuffer();
    const reader = res.body.getReader(), chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(got);
    }
    const bytes = new Uint8Array(got);
    let at = 0;
    for (const chunk of chunks) { bytes.set(chunk, at); at += chunk.length; }
    return bytes.buffer;
  }

  async function loadData() {
    const res = await fetch('py/digits.bin');
    if (!res.ok) throw new Error('could not download digits.bin');
    data = parseDigits(await download(res, got => { trainBtn.textContent = `… downloading digits (${(got / 1e6).toFixed(1)} MB)`; }));
    testGuess = new Uint8Array(data.nTest);
    testSure = new Float32Array(data.nTest);
    mix.shuffle = Uint16Array.from({ length: PIXELS }, (_, i) => i);
    shuffle(mix.shuffle);
    buildGallery();
    buildPairs();
  }

  function shuffle(list) {
    for (let i = list.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1)), keep = list[i];
      list[i] = list[j];
      list[j] = keep;
    }
  }

  let galleryCaps = [];
  function buildGallery() {
    $('dg-gallery').textContent = '';
    galleryCaps = [];
    for (let k = 0; k < GALLERY; k++) {
      const fig = tile(picture(data.nTrain + k), 1 / 255, '?');
      galleryCaps.push(fig.lastChild);
      $('dg-gallery').append(fig);
    }
  }

  function mark(guess, truth) {
    return guess === truth ? `<b class="dg-ok">${guess} ✓</b>` : `<b class="dg-no">${guess} ✗</b>`;
  }

  function paintGallery() {
    for (let k = 0; k < GALLERY; k++) {
      const n = data.nTrain + k;
      forward(net, data.where, data.value, data.start[n], data.start[n + 1], work.h, work.p);
      galleryCaps[k].innerHTML = mark(best(work.p), data.labels[n]);
    }
  }

  function showWrong() {
    const box = $('dg-wrong');
    let count = 0;
    box.textContent = '';
    for (let k = 0; k < data.nTest; k++) {
      const truth = data.labels[data.nTrain + k];
      if (testGuess[k] === truth) continue;
      if (count++ < WRONG_SHOWN) {
        box.append(tile(picture(data.nTrain + k), 1 / 255,
          `said <b class="dg-no">${testGuess[k]}</b>, ${whole(testSure[k])} sure<br>label: <b>${truth}</b>`));
      }
    }
    $('dg-wrong-note').textContent = `It misread ${count} of the ${data.nTest.toLocaleString()} test digits. ` +
      `Here are the first ${Math.min(count, WRONG_SHOWN)}: what the network said, how sure it was, and what the label says the digit really is.`;
  }

  function drawCurve() {
    Book.lineChart($('dg-curve'), {
      series: [
        { label: 'training digits', color: '#2563eb', data: history.map(pt => pt.train) },
        { label: 'unseen test digits', color: '#ea580c', data: history.map(pt => pt.test) },
      ],
      xMax: EPOCHS, yMin: 0.8, yMax: 1, yFormat: whole, xLabel: 'epochs of training',
    });
  }

  async function runTraining(model, set, guess, sure, onSlice, onEpoch) {
    const order = Int32Array.from({ length: set.nTrain }, (_, i) => i);
    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
      shuffle(order);
      for (let at = 0; at < set.nTrain; at += SLICE) {
        trainSlice(model, set, order, at, Math.min(at + SLICE, set.nTrain), work);
        onSlice();
        await pause();
      }
      onEpoch(epoch, score(model, set, 0, set.nTrain, work), score(model, set, set.nTrain, set.nTrain + set.nTest, work, guess, sure));
    }
  }

  function setBusy(on, label) {
    busy = on;
    document.querySelectorAll('[data-dg-train], #dg-mix').forEach(btn => { btn.disabled = on; });
    trainBtn.textContent = label;
  }

  async function train() {
    if (busy) return;
    setBusy(true, data ? '… training' : '… downloading digits');
    $('dg-error').hidden = true;
    try {
      if (!data) await loadData();
    } catch (err) {
      data = null;
      setBusy(false, 'load 10,000 digits & train');
      $('dg-error').hidden = false;
      $('dg-error').textContent = 'Could not load digits.bin (' + err.message + '). If you opened this page as a file, ' +
        'run "make serve" in the book folder and open http://localhost:8000';
      return;
    }
    trainBtn.textContent = '… training';
    net = makeNet(Math.random);
    history = [];
    $('dg-epoch').textContent = 0;
    $('dg-train-acc').textContent = $('dg-test-acc').textContent = $('dg-time').textContent = '…';
    $('dg-before').textContent = pct(score(net, data, data.nTrain, data.nTrain + data.nTest, work));
    $('dg-wrong').textContent = '';
    drawCurve();
    paintGallery();
    paintNeurons(net);
    const began = performance.now();
    await runTraining(net, data, testGuess, testSure, () => {
      paintGallery();
      paintNeurons(net);
    }, (epoch, trainAcc, testAcc) => {
      history.push({ train: trainAcc, test: testAcc });
      $('dg-epoch').textContent = epoch;
      $('dg-train-acc').textContent = pct(trainAcc);
      $('dg-test-acc').textContent = pct(testAcc);
      $('dg-time').textContent = ((performance.now() - began) / 1000).toFixed(1) + ' s';
      drawCurve();
    });
    document.querySelectorAll('.dg-need').forEach(note => { note.hidden = true; });
    showWrong();
    if (picked < 0) picked = strongestNeurons(net, 1)[0];
    paintNeurons(net);
    showNeuron();
    showReading();
    $('dg-mix-normal').textContent = pct(history[history.length - 1].test);
    drawMixCurve();
    if (nudge.mine && pad.strokes.length) useDrawing(); else nextTestDigit();
    setBusy(false, 'train again from scratch');
  }

  let picked = -1;
  const sheet = $('dg-neurons');

  function strongestNeurons(model, count) {
    const size = new Float64Array(HIDDEN);
    for (let i = 0; i < model.w1.length; i++) size[i % HIDDEN] += Math.abs(model.w1[i]);
    return Array.from(size.keys()).sort((a, b) => size[b] - size[a]).slice(0, count);
  }

  // A neuron whose weights are all tiny is drawn faint, not stretched to full
  // color, so one that has barely learned looks different from one that has.
  function tileScales(model) {
    const peak = new Float64Array(HIDDEN);
    let biggest = 1e-9;
    for (let i = 0; i < model.w1.length; i++) {
      const size = Math.abs(model.w1[i]);
      if (size > peak[i % HIDDEN]) peak[i % HIDDEN] = size;
      if (size > biggest) biggest = size;
    }
    return { faint: Array.from(peak, p => p < FAINT_TILE * biggest), full: peak.map(p => Math.max(p, FAINT_TILE * biggest)) };
  }

  function paintNeurons(model) {
    const ctx = sheet.getContext('2d'), shot = ctx.createImageData(SHEET, SHEET), w1 = model.w1, { full } = tileScales(model);
    for (let j = 0; j < HIDDEN; j++) {
      const left = GAP + (j % GRID) * (SIDE + GAP), top = GAP + Math.floor(j / GRID) * (SIDE + GAP);
      for (let i = 0; i < PIXELS; i++) {
        weightPixel(shot.data, 4 * ((top + Math.floor(i / SIDE)) * SHEET + left + i % SIDE), w1[i * HIDDEN + j] / full[j]);
      }
    }
    ctx.putImageData(shot, 0, 0);
    if (picked < 0) return;
    ctx.strokeStyle = '#1b1f27';
    ctx.lineWidth = GAP;
    ctx.strokeRect((picked % GRID) * (SIDE + GAP) + 1, Math.floor(picked / GRID) * (SIDE + GAP) + 1, SIDE + GAP, SIDE + GAP);
  }

  function drawVotes(votes, peak) {
    const { ctx, w, h } = Book.hidpi($('dg-votes')), each = w / DIGITS, mid = (h - 16) / 2;
    ctx.clearRect(0, 0, w, h);
    ctx.font = `600 12px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    votes.forEach((v, digit) => {
      const tall = Math.abs(v) / peak * (mid - 2);
      ctx.fillStyle = v > 0 ? '#2563eb' : '#dc2626';
      ctx.fillRect(digit * each + 5, v > 0 ? mid - tall : mid, each - 10, tall);
      ctx.fillStyle = '#374151';
      ctx.fillText(digit, (digit + 0.5) * each, h - 7);
    });
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid + 0.5); ctx.lineTo(w, mid + 0.5);
    ctx.stroke();
  }

  function showNeuron() {
    if (!net || picked < 0) return;
    const { faint, full } = tileScales(net);
    $('dg-neuron-title').textContent = `Neuron ${picked + 1} of ${HIDDEN}`;
    weightBitmap($('dg-neuron-big'), i => net.w1[i * HIDDEN + picked], full[picked]);
    const fans = [];
    for (let n = data.nTrain; n < data.nTrain + data.nTest; n++) {
      forward(net, data.where, data.value, data.start[n], data.start[n + 1], work.h, work.p);
      if (work.h[picked] > 0) fans.push([work.h[picked], n]);
    }
    fans.sort((a, b) => b[0] - a[0]);
    $('dg-neuron-top').textContent = '';
    for (const [, n] of fans.slice(0, 8)) $('dg-neuron-top').append(tile(picture(n), 1 / 255, ''));
    const votes = Array.from(net.w2.subarray(picked * DIGITS, (picked + 1) * DIGITS));
    drawVotes(votes, net.w2.reduce((most, v) => Math.max(most, Math.abs(v)), 1e-9));
    if (!fans.length) {
      $('dg-neuron-fans').textContent = 'This neuron does not fire for a single test digit. A neuron that never fires is called a dead neuron: ' +
        'ReLU sends out 0 every time, so no blame flows back to it and it stops learning.';
      $('dg-neuron-says').textContent = 'Its votes never count, because a neuron that sends out 0 has nothing to vote with.';
      return;
    }
    $('dg-neuron-fans').textContent = `It fires for ${fans.length.toLocaleString()} of the ${data.nTest.toLocaleString()} test digits. ` + (faint[picked]
      ? 'But its weights have hardly moved from their random start, so it only fires weakly. These excite it the most:'
      : 'These excite it the most. Can you see what they share with the blue parts of its tile?');
    $('dg-neuron-says').innerHTML = `When it fires, it votes <b style="color: var(--accent);">for</b> ${votes.indexOf(Math.max(...votes))} hardest ` +
      `and <b style="color: var(--bad);">against</b> ${votes.indexOf(Math.min(...votes))} hardest. Blue bars are votes for a digit, red bars are votes against.`;
  }

  sheet.addEventListener('click', event => {
    if (!net) return;
    const rect = sheet.getBoundingClientRect(), unit = rect.width / SHEET;
    const col = Math.min(GRID - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / unit - 1) / (SIDE + GAP))));
    const row = Math.min(GRID - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / unit - 1) / (SIDE + GAP))));
    picked = row * GRID + col;
    paintNeurons(net);
    showNeuron();
  });

  const pad = { strokes: [], current: null, tidy: true, image: new Float32Array(PIXELS), box: null, queued: false };
  const padEl = $('dg-pad'), padView = Book.hidpi(padEl);
  const scratch = document.createElement('canvas');
  scratch.width = scratch.height = PAD;
  const scratchCtx = scratch.getContext('2d', { willReadFrequently: true });

  function trace(ctx, place) {
    ctx.lineWidth = PEN;
    ctx.lineCap = ctx.lineJoin = 'round';
    for (const stroke of pad.strokes) {
      const [x0, y0] = place(stroke[0]);
      ctx.beginPath();
      ctx.arc(x0, y0, PEN / 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      for (let k = 1; k < stroke.length; k++) ctx.lineTo(...place(stroke[k]));
      ctx.stroke();
    }
  }

  function inkOf(place) {
    scratchCtx.clearRect(0, 0, PAD, PAD);
    scratchCtx.strokeStyle = scratchCtx.fillStyle = '#000';
    trace(scratchCtx, place);
    const rgba = scratchCtx.getImageData(0, 0, PAD, PAD).data, ink = new Float32Array(PAD * PAD);
    for (let i = 0; i < ink.length; i++) ink[i] = rgba[4 * i + 3] / 255;
    return ink;
  }

  function drawPad() {
    const { ctx, w } = padView;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, w);
    if (!pad.tidy) {
      ctx.strokeStyle = 'rgba(27, 31, 39, 0.09)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 1; k < SIDE; k++) {
        const at = k * w / SIDE;
        ctx.moveTo(at, 0); ctx.lineTo(at, w);
        ctx.moveTo(0, at); ctx.lineTo(w, at);
      }
      ctx.stroke();
    }
    if (!pad.strokes.length) {
      ctx.fillStyle = '#9ca3af';
      ctx.font = '15px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('draw a digit here', w / 2, w / 2);
      return;
    }
    ctx.strokeStyle = ctx.fillStyle = '#1b1f27';
    trace(ctx, p => [p.x, p.y]);
    if (!pad.tidy || !pad.box) return;
    const { x0, y0, x1, y1, cx, cy } = pad.box;
    ctx.strokeStyle = ORANGE;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'butt';
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();
  }

  // With the tidy-up on, the strokes are first redrawn at one standard size
  // so the pen is as thick, next to the digit, as the pens in MNIST were.
  function readPad() {
    pad.queued = false;
    pad.box = null;
    if (!pad.strokes.length) {
      pad.image = new Float32Array(PIXELS);
    } else if (pad.tidy) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const stroke of pad.strokes) {
        for (const p of stroke) {
          x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
          y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
        }
      }
      const grow = FIT / Math.max(x1 - x0, y1 - y0, FIT / 8), mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const tidied = tidy(inkOf(p => [PAD / 2 + (p.x - mx) * grow, PAD / 2 + (p.y - my) * grow]), PAD);
      const back = (v, mid) => mid + (v - PAD / 2) / grow;
      pad.image = tidied.image;
      if (tidied.box) {
        const b = tidied.box;
        pad.box = { x0: back(b.x0, mx), x1: back(b.x1, mx), cx: back(b.cx, mx), y0: back(b.y0, my), y1: back(b.y1, my), cy: back(b.cy, my) };
      }
    } else {
      pad.image = frame28(inkOf(p => [p.x, p.y]), PAD, PAD / 2, PAD / 2, PAD / SIDE);
    }
    drawPad();
    bitmap($('dg-seen'), pad.image, 1);
    showReading();
  }

  function queueRead() {
    if (pad.queued) return;
    pad.queued = true;
    requestAnimationFrame(readPad);
  }

  const bars = [];
  for (let digit = 0; digit < DIGITS; digit++) {
    const row = document.createElement('div');
    row.className = 'dg-bar';
    row.innerHTML = `<span class="dg-bar-digit">${digit}</span><div class="dg-bar-track"><div class="dg-bar-fill"></div></div><span class="dg-bar-pct">–</span>`;
    $('dg-bars').append(row);
    bars.push(row);
  }

  function showReading() {
    const ready = net && pad.strokes.length, p = ready ? read(net, pad.image, work) : null, top = ready ? best(p) : -1;
    bars.forEach((row, digit) => {
      row.classList.toggle('top', digit === top);
      row.children[1].firstChild.style.width = ready ? (p[digit] * 100) + '%' : '0';
      row.children[2].textContent = ready ? whole(p[digit]) : '–';
    });
    $('dg-guess').textContent = ready ? top : '?';
    $('dg-sure').textContent = ready ? whole(p[top]) + ' sure' : net ? 'draw a digit' : 'train the network first';
  }

  padEl.addEventListener('pointerdown', event => {
    event.preventDefault();
    try { padEl.setPointerCapture(event.pointerId); } catch (err) { /* the pointer is already gone */ }
    pad.current = [Book.pointer(padEl, event)];
    pad.strokes.push(pad.current);
    queueRead();
  });
  padEl.addEventListener('pointermove', event => {
    if (!pad.current) return;
    pad.current.push(Book.pointer(padEl, event));
    queueRead();
  });
  for (const type of ['pointerup', 'pointercancel']) padEl.addEventListener(type, () => { pad.current = null; });
  $('dg-clear').addEventListener('click', () => {
    pad.strokes = [];
    pad.current = null;
    readPad();
  });
  $('dg-tidy').addEventListener('click', event => {
    const choice = event.target.closest('[data-tidy]');
    if (!choice) return;
    pad.tidy = choice.dataset.tidy === 'on';
    $('dg-tidy').querySelectorAll('.toggle').forEach(t => t.classList.toggle('on', t === choice));
    readPad();
  });

  const nudge = { base: null, truth: 0, dx: 0, dy: 0, mine: false, map: [] };
  const mapView = Book.hidpi($('dg-nudge-map'));

  function nudgeFrom(base, truth, mine) {
    Object.assign(nudge, { base, truth, mine, dx: 0, dy: 0, map: [] });
    for (let dy = -REACH; dy <= REACH; dy++) {
      for (let dx = -REACH; dx <= REACH; dx++) {
        const p = read(net, shifted(base, dx, dy), work), top = best(p);
        nudge.map.push({ top, sure: p[top] });
      }
    }
    drawNudge();
  }

  function nextTestDigit() {
    for (let tries = 0; tries < 500; tries++) {
      const n = data.nTrain + Math.floor(Math.random() * data.nTest);
      const image = Float32Array.from(picture(n), v => v / 255), p = read(net, image, work), top = best(p);
      if (top === data.labels[n] && p[top] > 0.9) return nudgeFrom(image, top, false);
    }
  }

  function useDrawing() {
    if (!pad.strokes.length) return Book.toast('Draw a digit in the box further up first');
    nudgeFrom(pad.image.slice(), best(read(net, pad.image, work)), true);
  }

  function drawNudge() {
    if (!nudge.base) return;
    const { dx, dy, truth } = nudge, now = nudge.map[(dy + REACH) * SPAN + dx + REACH], right = now.top === truth;
    bitmap($('dg-nudge-view'), shifted(nudge.base, dx, dy), 1);
    $('dg-nudge-guess').textContent = now.top;
    $('dg-nudge-guess').classList.toggle('dg-no', !right);
    $('dg-nudge-sure').textContent = whole(now.sure) + ' sure';
    const step = (count, way) => `${count} pixel${count > 1 ? 's' : ''} ${way}`, moves = [];
    if (dx) moves.push(step(Math.abs(dx), dx > 0 ? 'right' : 'left'));
    if (dy) moves.push(step(Math.abs(dy), dy > 0 ? 'down' : 'up'));
    const moved = moves.length ? `Moved ${moves.join(' and ')}.` : 'Not moved yet.';
    let story = `The label says ${truth}. ${right ? (moves.length ? 'The network still agrees.' : 'The network agrees.') : `Now the network says ${now.top}.`}`;
    if (nudge.mine) {
      story = moves.length
        ? `Unmoved, the network read your drawing as ${truth}. ${right ? 'It still does.' : `Now it says ${now.top}.`}`
        : `The network reads your drawing as ${truth}.`;
    }
    $('dg-nudge-text').textContent = `${moved} ${story}`;

    const { ctx, w } = mapView, cell = w / SPAN;
    ctx.clearRect(0, 0, w, w);
    nudge.map.forEach((spot, k) => {
      ctx.fillStyle = `rgba(${spot.top === truth ? '22, 163, 74' : '220, 38, 38'}, ${0.25 + 0.75 * spot.sure})`;
      ctx.fillRect((k % SPAN) * cell + 0.5, Math.floor(k / SPAN) * cell + 0.5, cell - 1, cell - 1);
    });
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(w / 2, w / 2, 2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1b1f27';
    ctx.lineWidth = 2.5;
    ctx.strokeRect((dx + REACH) * cell, (dy + REACH) * cell, cell, cell);
  }

  function nudgeTo(dx, dy) {
    if (!nudge.base) return;
    nudge.dx = Math.min(REACH, Math.max(-REACH, dx));
    nudge.dy = Math.min(REACH, Math.max(-REACH, dy));
    drawNudge();
  }

  document.querySelectorAll('[data-nudge]').forEach(btn => {
    const [dx, dy] = btn.dataset.nudge.split(',').map(Number);
    btn.addEventListener('click', () => nudgeTo(nudge.dx + dx, nudge.dy + dy));
  });
  $('dg-nudge-map').addEventListener('click', event => {
    const p = Book.pointer($('dg-nudge-map'), event), cell = mapView.w / SPAN;
    nudgeTo(Math.floor(p.x / cell) - REACH, Math.floor(p.y / cell) - REACH);
  });
  $('dg-nudge-home').addEventListener('click', () => nudgeTo(0, 0));
  $('dg-nudge-next').addEventListener('click', () => { if (net && !busy) nextTestDigit(); });
  $('dg-nudge-mine').addEventListener('click', () => { if (net && !busy) useDrawing(); });

  const mix = { shuffle: null, net: null, tests: [], caps: [] };

  function jumbled(image) {
    const out = new Uint8Array(PIXELS);
    for (let i = 0; i < PIXELS; i++) out[mix.shuffle[i]] = image[i];
    return out;
  }

  function buildPairs() {
    $('dg-mix-pairs').textContent = '';
    mix.caps = [];
    for (let k = 0; k < PAIRS; k++) {
      const image = picture(data.nTrain + GALLERY + k), fig = tile(image, 1 / 255, '↓ scrambled'), below = tile(jumbled(image), 1 / 255, '&nbsp;');
      fig.append(...below.childNodes);
      mix.caps.push(fig.lastChild);
      $('dg-mix-pairs').append(fig);
    }
  }

  function drawMixCurve() {
    Book.lineChart($('dg-mix-curve'), {
      series: [
        { label: 'normal pictures', color: '#2563eb', data: history.map(pt => pt.test) },
        { label: 'scrambled pictures', color: '#ea580c', data: mix.tests },
      ],
      xMax: EPOCHS, yMin: 0.8, yMax: 1, yFormat: whole, xLabel: 'epochs of training',
    });
  }

  function showMixNeurons() {
    const box = $('dg-mix-neurons'), w1 = mix.net.w1;
    box.textContent = '';
    for (const j of strongestNeurons(mix.net, 8)) {
      const fig = tile(null, 1, 'as stored'), below = tile(null, 1, 'put back');
      weightBitmap(fig.firstChild, i => w1[i * HIDDEN + j]);
      weightBitmap(below.firstChild, i => w1[mix.shuffle[i] * HIDDEN + j]);
      fig.append(...below.childNodes);
      box.append(fig);
    }
    $('dg-mix-note').textContent = 'Eight of the new network\'s neurons, each shown twice. "As stored" is the tile exactly as this network holds it. ' +
      '"Put back" is the same weights with every pixel returned to where it came from.';
  }

  async function trainMixed() {
    if (busy || !net) return;
    setBusy(true, trainBtn.textContent);
    $('dg-mix').textContent = '… training on scrambled pictures';
    mix.net = makeNet(Math.random);
    mix.tests = [];
    const guess = new Uint8Array(data.nTest), sure = new Float32Array(data.nTest);
    await runTraining(mix.net, scrambled(data, mix.shuffle), guess, sure, () => {}, (epoch, trainAcc, testAcc) => {
      mix.tests.push(testAcc);
      $('dg-mix-epoch').textContent = epoch;
      $('dg-mix-acc').textContent = pct(testAcc);
      drawMixCurve();
    });
    mix.caps.forEach((cap, k) => { cap.innerHTML = 'it says ' + mark(guess[GALLERY + k], data.labels[data.nTrain + GALLERY + k]); });
    showMixNeurons();
    $('dg-mix').textContent = 'train on scrambled pictures again';
    setBusy(false, trainBtn.textContent);
  }

  function stopped(err) {
    setBusy(false, data ? 'train again from scratch' : 'load 10,000 digits & train');
    $('dg-error').hidden = false;
    $('dg-error').textContent = 'Training stopped: ' + err.message;
  }

  document.querySelectorAll('[data-dg-train]').forEach(btn => btn.addEventListener('click', () => train().catch(stopped)));
  $('dg-mix').addEventListener('click', () => trainMixed().catch(stopped));
  $('dg-epochs').textContent = $('dg-mix-epochs').textContent = EPOCHS;
  drawLook();
  drawPad();
  paintNeurons(makeNet(Math.random));
  drawCurve();
  drawMixCurve();
  bitmap($('dg-seen'), pad.image, 1);
  drawVotes(new Array(DIGITS).fill(0), 1);
})();
