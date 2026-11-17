(function overfittingWidget() {
  const field = document.getElementById('of-field');
  if (!field) return;
  const $ = id => document.getElementById(id);

  const SHAPES = { small: [4], medium: [8, 8], big: [32, 32] };
  const POOL = 400, TEST = 300, BLOCK = 20, WRONG_PER_BLOCK = 3;
  const WAVE = 0.3, LR = 0.2, MOMENTUM = 0.9;
  const EPOCHS = 1000, STEP = 5, POINTS = EPOCHS / STEP, TICK_MS = 30;
  const HOME_SEED = 14;
  const VIEW = 1.08, GRID = 64;
  const ORANGE = [234, 88, 12], BLUE = [37, 99, 235];
  const TRAIN_COLOR = '#475569', TEST_COLOR = '#0d9488';
  const CHART = { left: 46, right: 12, top: 14, bottom: 34 }, Y_MIN = 0.6;

  const { ctx, w: SIZE } = Book.hidpi(field);
  const paint = document.createElement('canvas');
  paint.width = paint.height = GRID;
  const paintCtx = paint.getContext('2d');
  const pixels = paintCtx.createImageData(GRID, GRID);
  const pct = v => Math.round(v * 100) + '%';

  let seed = HOME_SEED, pool, test, n, net;
  let epoch, history, best, rewound, running = false, runId = 0, showTest = false;

  const trueColor = (x, y) => (y > WAVE * Math.sin(Math.PI * x) ? 1 : 0);

  function makeDots(count, rnd, wrongPerBlock) {
    const xs = new Float64Array(2 * count), ys = new Uint8Array(count), wrong = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      xs[2 * i] = rnd() * 2 - 1;
      xs[2 * i + 1] = rnd() * 2 - 1;
      ys[i] = trueColor(xs[2 * i], xs[2 * i + 1]);
    }
    // Exactly 3 wrong labels in every block of 20, so every slider stop is exactly 15% wrong.
    for (let start = 0; start < count && wrongPerBlock; start += BLOCK) {
      const idx = Array.from({ length: BLOCK }, (_, k) => start + k);
      for (let k = 0; k < wrongPerBlock; k++) {
        const j = k + Math.floor(rnd() * (BLOCK - k));
        [idx[k], idx[j]] = [idx[j], idx[k]];
        wrong[idx[k]] = 1;
        ys[idx[k]] ^= 1;
      }
    }
    return { xs, ys, wrong };
  }

  function makeNet(hidden, rnd) {
    const sizes = [2, ...hidden, 1];
    const W = [], B = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const scale = Math.sqrt(1 / sizes[l]);
      W.push(Float64Array.from({ length: sizes[l] * sizes[l + 1] }, () => (rnd() * 2 - 1) * scale));
      B.push(new Float64Array(sizes[l + 1]));
    }
    const like = arrays => arrays.map(a => new Float64Array(a.length));
    return {
      sizes, W, B,
      acts: sizes.map(s => new Float64Array(s)),
      deltas: sizes.map(s => new Float64Array(s)),
      gW: like(W), gB: like(B), vW: like(W), vB: like(B),
    };
  }

  function forward(x, y) {
    const { sizes, W, B, acts } = net;
    const last = W.length - 1;
    acts[0][0] = x;
    acts[0][1] = y;
    for (let l = 0; l <= last; l++) {
      const inp = acts[l], out = acts[l + 1], w = W[l], b = B[l], nin = sizes[l];
      for (let j = 0; j < out.length; j++) {
        let z = b[j];
        const row = j * nin;
        for (let i = 0; i < nin; i++) z += w[row + i] * inp[i];
        out[j] = l === last ? sigmoid(z) : Math.tanh(z);
      }
    }
    return acts[last + 1][0];
  }

  function addGradient(x, y, target) {
    const { sizes, W, acts, deltas, gW, gB } = net;
    const last = W.length - 1;
    // Sigmoid output with cross-entropy loss: the error signal at the output is simply (guess - truth).
    deltas[last + 1][0] = forward(x, y) - target;
    for (let l = last; l >= 0; l--) {
      const inp = acts[l], d = deltas[l + 1], w = W[l], nin = sizes[l];
      const gw = gW[l], gb = gB[l], back = deltas[l];
      if (l > 0) back.fill(0);
      for (let j = 0; j < d.length; j++) {
        const dj = d[j], row = j * nin;
        gb[j] += dj;
        for (let i = 0; i < nin; i++) {
          gw[row + i] += dj * inp[i];
          if (l > 0) back[i] += dj * w[row + i];
        }
      }
      if (l > 0) for (let i = 0; i < nin; i++) back[i] *= 1 - inp[i] * inp[i];
    }
  }

  function roll(weights, grads, velocity) {
    for (let k = 0; k < weights.length; k++) {
      velocity[k] = MOMENTUM * velocity[k] - LR * grads[k] / n;
      weights[k] += velocity[k];
    }
  }

  function trainEpoch() {
    for (const g of net.gW) g.fill(0);
    for (const g of net.gB) g.fill(0);
    for (let i = 0; i < n; i++) addGradient(pool.xs[2 * i], pool.xs[2 * i + 1], pool.ys[i]);
    for (let l = 0; l < net.W.length; l++) {
      roll(net.W[l], net.gW[l], net.vW[l]);
      roll(net.B[l], net.gB[l], net.vB[l]);
    }
  }

  function accuracy(dots, count) {
    let right = 0;
    for (let i = 0; i < count; i++) {
      if ((forward(dots.xs[2 * i], dots.xs[2 * i + 1]) > 0.5) === (dots.ys[i] === 1)) right++;
    }
    return right / count;
  }

  const copyAll = arrays => arrays.map(a => a.slice());
  const snapshot = () => ({ W: copyAll(net.W), B: copyAll(net.B), vW: copyAll(net.vW), vB: copyAll(net.vB) });
  function restore(snap) {
    for (const key of ['W', 'B', 'vW', 'vB']) snap[key].forEach((a, l) => net[key][l].set(a));
  }

  function record() {
    const point = { train: accuracy(pool, n), test: accuracy(test, TEST) };
    history.push(point);
    if (!best || point.test > best.test) {
      best = { index: history.length - 1, epoch, train: point.train, test: point.test, weights: snapshot() };
    }
  }

  const toPx = v => ((v + VIEW) / (2 * VIEW)) * SIZE;
  const toPy = v => SIZE - toPx(v);

  function drawField() {
    const data = pixels.data;
    for (let gy = 0; gy < GRID; gy++) {
      const y = VIEW - ((gy + 0.5) / GRID) * 2 * VIEW;
      for (let gx = 0; gx < GRID; gx++) {
        const p = forward(((gx + 0.5) / GRID) * 2 * VIEW - VIEW, y);
        const rgb = p > 0.5 ? ORANGE : BLUE, k = 4 * (gy * GRID + gx);
        data[k] = rgb[0];
        data[k + 1] = rgb[1];
        data[k + 2] = rgb[2];
        data[k + 3] = 255 * (0.08 + 0.4 * Math.abs(p - 0.5) * 2);
      }
    }
    paintCtx.putImageData(pixels, 0, 0);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(paint, 0, 0, SIZE, SIZE);
  }

  function drawTrueRule() {
    ctx.beginPath();
    for (let s = 0; s <= 60; s++) {
      const x = -VIEW + (s / 60) * 2 * VIEW;
      const px = toPx(x), py = toPy(WAVE * Math.sin(Math.PI * x));
      if (s) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = 'rgba(17, 24, 39, 0.6)';
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawTrainingDots() {
    const r = n <= 100 ? 4.4 : n <= 200 ? 3.7 : 3.1;
    ctx.globalAlpha = showTest ? 0.3 : 1;
    for (let i = 0; i < n; i++) {
      const px = toPx(pool.xs[2 * i]), py = toPy(pool.xs[2 * i + 1]);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fillStyle = pool.ys[i] ? '#ea580c' : '#2563eb';
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
      if (pool.wrong[i]) {
        ctx.beginPath();
        ctx.arc(px, py, r + 2.4, 0, 2 * Math.PI);
        ctx.lineWidth = 1.7;
        ctx.strokeStyle = '#111827';
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawTestDots() {
    for (let i = 0; i < TEST; i++) {
      const x = test.xs[2 * i], y = test.xs[2 * i + 1];
      const px = toPx(x), py = toPy(y);
      const missed = (forward(x, y) > 0.5) !== (test.ys[i] === 1);
      const r = missed ? 5 : 3.2;
      ctx.beginPath();
      ctx.moveTo(px, py - r);
      ctx.lineTo(px + r, py);
      ctx.lineTo(px, py + r);
      ctx.lineTo(px - r, py);
      ctx.closePath();
      ctx.fillStyle = test.ys[i] ? '#ea580c' : '#2563eb';
      ctx.fill();
      ctx.lineWidth = missed ? 2 : 0.8;
      ctx.strokeStyle = missed ? '#111827' : '#fff';
      ctx.stroke();
    }
  }

  function drawChart() {
    const canvas = $('of-chart');
    Book.lineChart(canvas, {
      series: [
        { label: 'training dots', color: TRAIN_COLOR, data: history.map(p => p.train) },
        { label: 'test dots', color: TEST_COLOR, data: history.map(p => p.test) },
      ],
      xMax: POINTS, yMin: Y_MIN, yMax: 1, yFormat: pct, xLabel: 'epochs of training (0 to ' + EPOCHS.toLocaleString() + ')',
    });
    if (!best || history.length < 2) return;
    // Book.lineChart has no marker option, so the best-moment line is drawn on top using its plot margins.
    const g = canvas.getContext('2d');
    const w = +canvas.dataset.w, h = +canvas.dataset.h;
    const x = CHART.left + (best.index / (POINTS - 1)) * (w - CHART.left - CHART.right);
    const yOf = v => CHART.top + (1 - (Math.max(Y_MIN, v) - Y_MIN) / (1 - Y_MIN)) * (h - CHART.top - CHART.bottom);
    g.save();
    g.strokeStyle = TEST_COLOR;
    g.lineWidth = 1.5;
    g.setLineDash([4, 4]);
    g.beginPath();
    g.moveTo(x, CHART.top + 16);
    g.lineTo(x, h - CHART.bottom);
    g.stroke();
    g.setLineDash([]);
    if (rewound) {
      for (const [v, color] of [[best.train, TRAIN_COLOR], [best.test, TEST_COLOR]]) {
        g.beginPath();
        g.arc(x, yOf(v), 5.5, 0, 2 * Math.PI);
        g.fillStyle = '#fff';
        g.fill();
        g.lineWidth = 2.5;
        g.strokeStyle = color;
        g.stroke();
      }
    }
    const label = 'best test: epoch ' + best.epoch;
    g.font = '600 12px ' + getComputedStyle(canvas).fontFamily;
    g.textBaseline = 'middle';
    g.textAlign = 'left';
    const width = g.measureText(label).width;
    const labelX = Math.min(x + 7, w - CHART.right - width - 4);
    const labelY = h - CHART.bottom - 12;
    g.fillStyle = 'rgba(255, 255, 255, 0.9)';
    g.fillRect(labelX - 4, labelY - 8, width + 8, 16);
    g.fillStyle = TEST_COLOR;
    g.fillText(label, labelX, labelY);
    g.restore();
  }

  function describe(now) {
    if (!history.length) {
      return 'A fresh network with random weights. It has seen nothing yet, so the map behind the dots is a blur. Press train.';
    }
    const drop = Math.round((best.test - now.test) * 100);
    if (rewound) {
      return 'Rewound to epoch ' + best.epoch + ', when the test score was at its best. Back then the network scored ' +
        pct(now.train) + ' on the training dots and ' + pct(now.test) + ' on the test.';
    }
    if (epoch >= EPOCHS) {
      const scores = 'Finished. ' + pct(now.train) + ' on the training dots, ' + pct(now.test) + ' on the test';
      return drop >= 3
        ? scores + '. The test score peaked at ' + pct(best.test) + ' back at epoch ' + best.epoch +
          ' and has dropped ' + drop + ' points since.'
        : scores + ', which is ' + (drop > 0 ? 'within ' + drop + (drop === 1 ? ' point' : ' points') + ' of' : 'level with') +
          ' its best. No memorizing to speak of.';
    }
    if (drop >= 3) {
      return now.train > best.train
        ? 'Memorizing. The training score is still climbing, but the test score is down ' + drop + ' points from its best.'
        : 'The test score has slipped ' + drop + ' points from its best, back at epoch ' + best.epoch + '.';
    }
    return 'Learning the rule. Both scores move together.';
  }

  function drawMap() {
    drawField();
    drawTrueRule();
    drawTrainingDots();
    if (showTest) drawTestDots();
  }

  function render(skipMap) {
    if (!skipMap) drawMap();
    drawChart();
    const now = history.length ? history[history.length - 1] : null;
    const shown = rewound ? best : now;
    $('of-epoch').textContent = epoch.toLocaleString();
    $('of-train-acc').textContent = shown ? pct(shown.train) : '?';
    $('of-test-acc').textContent = shown ? pct(shown.test) : '?';
    $('of-best').textContent = best ? pct(best.test) + ' at epoch ' + best.epoch : '?';
    $('of-note').textContent = describe(shown);
    $('of-train').textContent = running ? 'pause' : epoch >= EPOCHS ? 'train again' : epoch > 0 ? 'keep training' : 'train';
    $('of-train').classList.toggle('stop', running);
    $('of-rewind').disabled = !best || rewound || epoch <= best.epoch;
  }

  function reset() {
    runId++;
    running = false;
    n = +$('of-n').value;
    const shape = SHAPES[document.querySelector('#of-size .toggle.on').dataset.size];
    net = makeNet(shape, Book.rng(seed * 7919 + shape.length * 100 + shape[0]));
    epoch = 0;
    history = [];
    best = null;
    rewound = false;
    $('of-n-v').textContent = n + ' dots, ' + (n * WRONG_PER_BLOCK / BLOCK) + ' mislabeled';
    $('of-params').textContent = net.W.reduce((sum, w, l) => sum + w.length + net.B[l].length, 0).toLocaleString();
    render();
  }

  function newDots() {
    pool = makeDots(POOL, Book.rng(seed), WRONG_PER_BLOCK);
    test = makeDots(TEST, Book.rng(seed + 5000), 0);
    reset();
  }

  async function train() {
    if (running) {
      runId++;
      running = false;
      render();
      return;
    }
    if (epoch >= EPOCHS) reset();
    if (rewound) {
      history.length = best.index + 1;
      rewound = false;
    }
    const run = ++runId;
    running = true;
    let mapCost = 0, mapDone = 0;
    while (epoch < EPOCHS) {
      const started = performance.now();
      for (let k = 0; k < STEP; k++) trainEpoch();
      epoch += STEP;
      record();
      // On a slow machine painting the map costs more than the training, so skip frames rather than stall.
      const skipMap = performance.now() - mapDone < 1.5 * mapCost;
      const before = performance.now();
      render(skipMap);
      if (!skipMap) {
        mapDone = performance.now();
        mapCost = mapDone - before;
      }
      await new Promise(resolve => setTimeout(resolve, Math.max(0, TICK_MS - (performance.now() - started))));
      if (run !== runId) return;
    }
    running = false;
    render();
  }

  function rewind() {
    runId++;
    running = false;
    restore(best.weights);
    epoch = best.epoch;
    rewound = true;
    render();
  }

  $('of-train').addEventListener('click', train);
  $('of-rewind').addEventListener('click', rewind);
  $('of-reset').addEventListener('click', reset);
  $('of-new').addEventListener('click', () => {
    seed = 1 + Math.floor(Math.random() * 100000);
    newDots();
  });
  $('of-n').addEventListener('input', reset);
  $('of-size').addEventListener('click', e => {
    const picked = e.target.closest('.toggle');
    if (!picked) return;
    $('of-size').querySelectorAll('.toggle').forEach(t => t.classList.toggle('on', t === picked));
    reset();
  });
  $('of-show-test').addEventListener('click', () => {
    showTest = !showTest;
    $('of-show-test').classList.toggle('on', showTest);
    render();
  });

  newDots();
})();
