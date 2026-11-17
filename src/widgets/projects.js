(function symbolTeacher() {
  const SIDE = 28, PIXELS = SIDE * SIDE, PAD = 280, FINE = 4, PEN = 11;
  const HIDDEN = 32, EPOCHS = 60, BATCH = 8, LR = 0.1;
  const COPIES = 8, SLIDE = 4, NUDGE = 3, HOLD_EVERY = 4;

  // Every pixel is cut into FINE x FINE smaller squares, and its ink is the share
  // of those that lie within PEN of a stroke. That share is what makes the gray edges.
  function raster(strokes) {
    const grid = SIDE * FINE, cell = PAD / grid, covered = new Uint8Array(grid * grid);
    const mark = (ax, ay, bx, by) => {
      const x0 = Math.max(0, Math.floor((Math.min(ax, bx) - PEN) / cell)), x1 = Math.min(grid - 1, Math.floor((Math.max(ax, bx) + PEN) / cell));
      const y0 = Math.max(0, Math.floor((Math.min(ay, by) - PEN) / cell)), y1 = Math.min(grid - 1, Math.floor((Math.max(ay, by) + PEN) / cell));
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
      for (let gy = y0; gy <= y1; gy++) {
        const py = (gy + 0.5) * cell;
        for (let gx = x0; gx <= x1; gx++) {
          const px = (gx + 0.5) * cell;
          const t = len2 ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
          const ex = px - ax - t * dx, ey = py - ay - t * dy;
          if (ex * ex + ey * ey <= PEN * PEN) covered[gy * grid + gx] = 1;
        }
      }
    };
    for (const stroke of strokes) {
      if (stroke.length === 1) mark(stroke[0].x, stroke[0].y, stroke[0].x, stroke[0].y);
      for (let k = 1; k < stroke.length; k++) mark(stroke[k - 1].x, stroke[k - 1].y, stroke[k].x, stroke[k].y);
    }
    const image = new Float32Array(PIXELS);
    for (let y = 0; y < SIDE; y++) {
      for (let x = 0; x < SIDE; x++) {
        let sum = 0;
        for (let fy = 0; fy < FINE; fy++) for (let fx = 0; fx < FINE; fx++) sum += covered[(y * FINE + fy) * grid + x * FINE + fx];
        image[y * SIDE + x] = sum / (FINE * FINE);
      }
    }
    return image;
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

  function makeNet(kinds, random) {
    const gauss = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    const w1 = new Float64Array(PIXELS * HIDDEN), w2 = new Float64Array(HIDDEN * kinds);
    const spread1 = Math.sqrt(2 / PIXELS), spread2 = Math.sqrt(2 / HIDDEN);
    for (let i = 0; i < w1.length; i++) w1[i] = gauss() * spread1;
    for (let i = 0; i < w2.length; i++) w2[i] = gauss() * spread2;
    return {
      kinds, w1, b1: new Float64Array(HIDDEN), w2, b2: new Float64Array(kinds),
      g1: new Float64Array(w1.length), gb1: new Float64Array(HIDDEN), g2: new Float64Array(w2.length), gb2: new Float64Array(kinds),
      h: new Float64Array(HIDDEN), p: new Float64Array(kinds), dh: new Float64Array(HIDDEN),
    };
  }

  function forward(net, image) {
    const { kinds, w1, b1, w2, b2, h, p } = net;
    for (let j = 0; j < HIDDEN; j++) h[j] = b1[j];
    for (let i = 0; i < PIXELS; i++) {
      const v = image[i];
      if (!v) continue;
      const row = i * HIDDEN;
      for (let j = 0; j < HIDDEN; j++) h[j] += v * w1[row + j];
    }
    for (let c = 0; c < kinds; c++) p[c] = b2[c];
    for (let j = 0; j < HIDDEN; j++) {
      if (h[j] <= 0) { h[j] = 0; continue; }
      const row = j * kinds, a = h[j];
      for (let c = 0; c < kinds; c++) p[c] += a * w2[row + c];
    }
    let top = p[0], sum = 0;
    for (let c = 1; c < kinds; c++) if (p[c] > top) top = p[c];
    for (let c = 0; c < kinds; c++) { p[c] = Math.exp(p[c] - top); sum += p[c]; }
    for (let c = 0; c < kinds; c++) p[c] /= sum;
    return p;
  }

  function batchGradient(net, examples, from, to) {
    const { kinds, w2, g1, gb1, g2, gb2, h, p, dh } = net;
    const share = 1 / (to - from);
    let loss = 0;
    g1.fill(0); gb1.fill(0); g2.fill(0); gb2.fill(0);
    for (let n = from; n < to; n++) {
      const { image, label } = examples[n];
      forward(net, image);
      loss -= Math.log(p[label] + 1e-300);
      p[label] -= 1;
      for (let c = 0; c < kinds; c++) { p[c] *= share; gb2[c] += p[c]; }
      for (let j = 0; j < HIDDEN; j++) {
        if (h[j] <= 0) { dh[j] = 0; continue; }
        const row = j * kinds, act = h[j];
        let blame = 0;
        for (let c = 0; c < kinds; c++) { g2[row + c] += act * p[c]; blame += p[c] * w2[row + c]; }
        dh[j] = blame;
        gb1[j] += blame;
      }
      for (let i = 0; i < PIXELS; i++) {
        const v = image[i];
        if (!v) continue;
        const row = i * HIDDEN;
        for (let j = 0; j < HIDDEN; j++) g1[row + j] += v * dh[j];
      }
    }
    return loss * share;
  }

  function step(net) {
    const { w1, b1, w2, b2, g1, gb1, g2, gb2 } = net;
    for (let i = 0; i < w1.length; i++) w1[i] -= LR * g1[i];
    for (let i = 0; i < w2.length; i++) w2[i] -= LR * g2[i];
    for (let j = 0; j < b1.length; j++) b1[j] -= LR * gb1[j];
    for (let c = 0; c < b2.length; c++) b2[c] -= LR * gb2[c];
  }

  function trainEpoch(net, examples, random) {
    for (let i = examples.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [examples[i], examples[j]] = [examples[j], examples[i]];
    }
    for (let at = 0; at < examples.length; at += BATCH) {
      batchGradient(net, examples, at, Math.min(at + BATCH, examples.length));
      step(net);
    }
  }

  function best(p) {
    let top = 0;
    for (let c = 1; c < p.length; c++) if (p[c] > p[top]) top = c;
    return top;
  }

  function countRight(net, examples) {
    let right = 0;
    for (const { image, label } of examples) if (best(forward(net, image)) === label) right++;
    return right;
  }

  function withSlidCopies(examples, random) {
    const out = examples.slice(), span = 2 * SLIDE + 1;
    for (const { image, label } of examples) {
      for (let c = 0; c < COPIES; c++) {
        let dx = 0, dy = 0;
        while (!dx && !dy) {
          dx = Math.floor(random() * span) - SLIDE;
          dy = Math.floor(random() * span) - SLIDE;
        }
        out.push({ image: shifted(image, dx, dy), label });
      }
    }
    return out;
  }

  function nudged(examples) {
    const out = [];
    for (const { image, label } of examples) {
      for (const [dx, dy] of [[NUDGE, 0], [-NUDGE, 0], [0, NUDGE], [0, -NUDGE]]) out.push({ image: shifted(image, dx, dy), label });
    }
    return out;
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = {
      SIDE, PIXELS, PAD, PEN, HIDDEN, EPOCHS, BATCH, LR, COPIES, SLIDE, NUDGE, HOLD_EVERY,
      raster, shifted, makeNet, forward, batchGradient, step, trainEpoch, best, countRight, withSlidCopies, nudged,
    };
  }
  if (typeof document === 'undefined') return;
  const padEl = document.getElementById('sy-pad');
  if (!padEl) return;

  const $ = id => document.getElementById(id);
  const COLORS = ['#db2777', '#2563eb', '#0d9488', '#ea580c'];
  const STARTERS = ['star', 'bolt', 'spiral', 'my initials'];
  const INK = '#1d2238', MIN_SYMBOLS = 2, MAX_SYMBOLS = 4, MAX_EACH = 40, MIN_EACH = HOLD_EVERY, GOAL_EACH = 12, STORE = 'projects.symbols';
  const MODES = ['mine', 'copies'];
  const whole = x => Math.round(x * 100) + '%';
  const escapeHtml = text => text.replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

  // A message, not a timer: timers in a background tab fire once a second,
  // which would stretch a one-second training run to a minute.
  const channel = new MessageChannel();
  const pause = () => new Promise(resolve => {
    if (document.hidden) { channel.port1.onmessage = resolve; channel.port2.postMessage(0); } else setTimeout(resolve, 16);
  });

  function toBytes(image) {
    let text = '';
    for (let i = 0; i < PIXELS; i++) text += String.fromCharCode(Math.round(image[i] * 255));
    return btoa(text);
  }

  function fromBytes(text) {
    const raw = atob(text);
    if (raw.length !== PIXELS) throw new Error('not a 28 x 28 drawing');
    const image = new Float32Array(PIXELS);
    for (let i = 0; i < PIXELS; i++) image[i] = raw.charCodeAt(i) / 255;
    return image;
  }

  function loadState() {
    const fresh = { symbols: STARTERS.slice(0, MIN_SYMBOLS).map(name => ({ name, drawings: [] })), copies: false };
    const saved = Book.store.get(STORE, null);
    if (!saved || !Array.isArray(saved.symbols)) return fresh;
    try {
      const symbols = saved.symbols.slice(0, MAX_SYMBOLS).map(s => ({
        name: String(s.name || '').slice(0, 14),
        drawings: (Array.isArray(s.drawings) ? s.drawings : []).slice(0, MAX_EACH).map(fromBytes),
      }));
      return symbols.length >= MIN_SYMBOLS ? { symbols, copies: !!saved.copies } : fresh;
    } catch (err) {
      return fresh;
    }
  }

  const state = loadState();
  let active = 0, busy = false, trained = null, stale = false, results = {};

  function save() {
    Book.store.set(STORE, {
      symbols: state.symbols.map(s => ({ name: s.name, drawings: s.drawings.map(toBytes) })),
      copies: state.copies,
    });
  }

  const nameOf = k => state.symbols[k].name.trim() || 'symbol ' + (k + 1);
  const isHeld = n => n % HOLD_EVERY === HOLD_EVERY - 1;

  function paintBitmap(canvas, image, rgb) {
    const ctx = canvas.getContext('2d'), shot = ctx.createImageData(SIDE, SIDE);
    for (let i = 0; i < PIXELS; i++) {
      shot.data[4 * i] = rgb[0]; shot.data[4 * i + 1] = rgb[1]; shot.data[4 * i + 2] = rgb[2];
      shot.data[4 * i + 3] = image[i] * 255;
    }
    ctx.putImageData(shot, 0, 0);
  }

  const rgbOf = hex => [1, 3, 5].map(at => parseInt(hex.slice(at, at + 2), 16));

  function bitmapCanvas(image, hex) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = SIDE;
    paintBitmap(canvas, image, rgbOf(hex));
    return canvas;
  }

  function makePad(canvas, getColor, hint, onChange) {
    const view = Book.hidpi(canvas), pad = { strokes: [], current: null };
    pad.draw = () => {
      const { ctx, w } = view;
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, w, w);
      if (!pad.strokes.length) {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '15px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(hint(), w / 2, w / 2);
        return;
      }
      ctx.strokeStyle = ctx.fillStyle = getColor();
      ctx.lineWidth = 2 * PEN;
      ctx.lineCap = ctx.lineJoin = 'round';
      for (const stroke of pad.strokes) {
        ctx.beginPath();
        ctx.arc(stroke[0].x, stroke[0].y, PEN, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(stroke[0].x, stroke[0].y);
        for (let k = 1; k < stroke.length; k++) ctx.lineTo(stroke[k].x, stroke[k].y);
        ctx.stroke();
      }
    };
    let queued = false;
    const changed = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => { queued = false; pad.draw(); onChange(); });
    };
    pad.set = strokes => { pad.strokes = strokes; pad.current = null; pad.draw(); onChange(); };
    canvas.addEventListener('pointerdown', event => {
      event.preventDefault();
      try { canvas.setPointerCapture(event.pointerId); } catch (err) { /* the pointer is already gone */ }
      pad.current = [Book.pointer(canvas, event)];
      pad.strokes.push(pad.current);
      changed();
    });
    canvas.addEventListener('pointermove', event => {
      if (!pad.current) return;
      pad.current.push(Book.pointer(canvas, event));
      changed();
    });
    for (const type of ['pointerup', 'pointercancel']) canvas.addEventListener(type, () => { pad.current = null; });
    pad.draw();
    return pad;
  }

  const collect = makePad(padEl, () => COLORS[active], () => `draw one ${nameOf(active)} here`, () => {
    $('sy-keep').disabled = !collect.strokes.length;
  });

  function renderSymbols() {
    const box = $('sy-symbols');
    box.innerHTML = '';
    state.symbols.forEach((symbol, k) => {
      const row = document.createElement('div');
      row.className = 'sy-symbol' + (k === active ? ' on' : '');
      row.style.setProperty('--sy', COLORS[k]);
      row.dataset.symbol = k;
      const n = symbol.drawings.length, held = symbol.drawings.filter((d, i) => isHeld(i)).length;
      row.innerHTML = `
        <div class="sy-symbol-head">
          <span class="sy-dot"></span>
          <input class="sy-name" type="text" maxlength="14" value="${escapeHtml(symbol.name)}" placeholder="name it" aria-label="name of symbol ${k + 1}" autocomplete="off" spellcheck="false">
          <span class="sy-count"><strong>${n}</strong> of ${GOAL_EACH}${n > GOAL_EACH ? ' and counting' : ''}${held ? ` · ${held} held back` : ''}</span>
          ${state.symbols.length > MIN_SYMBOLS ? '<button type="button" class="secondary sy-remove">remove</button>' : ''}
        </div>
        <div class="sy-thumbs"></div>`;
      const thumbs = row.querySelector('.sy-thumbs');
      if (!n) thumbs.innerHTML = `<span class="sy-empty">${k === active ? 'No drawings yet. Draw one on the pad and keep it.' : 'Click here, then draw on the pad.'}</span>`;
      symbol.drawings.forEach((image, i) => {
        const thumb = document.createElement('button');
        thumb.type = 'button';
        thumb.className = 'sy-thumb' + (isHeld(i) ? ' held' : '');
        thumb.dataset.drawing = i;
        thumb.title = (isHeld(i) ? 'held back for the test. ' : '') + 'click to throw this drawing away';
        thumb.setAttribute('aria-label', `${nameOf(k)} drawing ${i + 1}${isHeld(i) ? ', held back for the test' : ''}. Click to throw it away.`);
        thumb.append(bitmapCanvas(image, INK));
        thumbs.append(thumb);
      });
      box.append(row);
    });
    $('sy-add').hidden = state.symbols.length >= MAX_SYMBOLS;
    $('sy-keep').textContent = `keep it as a “${nameOf(active)}”`;
    $('sy-pad-wrap').style.setProperty('--sy', COLORS[active]);
    const ready = state.symbols.filter(s => s.drawings.length >= MIN_EACH).length;
    $('sy-train').disabled = busy || ready < MIN_SYMBOLS;
    $('sy-ready').textContent = ready >= MIN_SYMBOLS
      ? ''
      : `To train, at least two symbols need ${MIN_EACH} drawings each, so that each has one to hold back. A dozen each is better.`;
    $('sy-ready').hidden = ready >= MIN_SYMBOLS;
    collect.draw();
  }

  function drawingsChanged() {
    results = {};
    if (trained) stale = true;
    save();
    renderSymbols();
    showScores();
  }

  $('sy-symbols').addEventListener('click', event => {
    const row = event.target.closest('.sy-symbol');
    if (!row || busy) return;
    const k = +row.dataset.symbol;
    if (event.target.closest('.sy-remove')) {
      state.symbols.splice(k, 1);
      active = Math.min(active, state.symbols.length - 1);
      trained = null;
      stale = false;
      drawingsChanged();
      showReading();
      return;
    }
    const thumb = event.target.closest('.sy-thumb');
    if (thumb) {
      state.symbols[k].drawings.splice(+thumb.dataset.drawing, 1);
      active = k;
      drawingsChanged();
      return;
    }
    if (event.target.closest('.sy-name') || k === active) return;
    active = k;
    renderSymbols();
  });
  $('sy-symbols').addEventListener('focusin', event => {
    const row = event.target.closest('.sy-symbol');
    if (!row || !event.target.closest('.sy-name') || +row.dataset.symbol === active) return;
    active = +row.dataset.symbol;
    document.querySelectorAll('#sy-symbols .sy-symbol').forEach(el => el.classList.toggle('on', el === row));
    $('sy-keep').textContent = `keep it as a “${nameOf(active)}”`;
    $('sy-pad-wrap').style.setProperty('--sy', COLORS[active]);
    collect.draw();
  });
  $('sy-symbols').addEventListener('input', event => {
    const row = event.target.closest('.sy-symbol');
    if (!row || !event.target.closest('.sy-name')) return;
    state.symbols[+row.dataset.symbol].name = event.target.value;
    save();
    $('sy-keep').textContent = `keep it as a “${nameOf(active)}”`;
    collect.draw();
    showScores();
    showReading();
  });
  $('sy-add').addEventListener('click', () => {
    if (busy || state.symbols.length >= MAX_SYMBOLS) return;
    const taken = new Set(state.symbols.map(s => s.name));
    state.symbols.push({ name: STARTERS.find(name => !taken.has(name)) || '', drawings: [] });
    active = state.symbols.length - 1;
    trained = null;
    stale = false;
    drawingsChanged();
    showReading();
  });
  $('sy-keep').addEventListener('click', () => {
    if (busy || !collect.strokes.length) return;
    const symbol = state.symbols[active];
    if (symbol.drawings.length >= MAX_EACH) {
      Book.toast(`${MAX_EACH} is plenty for one symbol`);
      return;
    }
    symbol.drawings.push(fromBytes(toBytes(raster(collect.strokes))));
    collect.set([]);
    drawingsChanged();
  });
  $('sy-wipe').addEventListener('click', () => collect.set([]));

  const clearBtn = $('sy-clear');
  let clearTimer = 0;
  clearBtn.addEventListener('click', () => {
    if (busy) return;
    if (!clearBtn.classList.contains('stop')) {
      clearBtn.classList.add('stop');
      clearBtn.textContent = 'really throw away every drawing?';
      clearTimer = setTimeout(() => { clearBtn.classList.remove('stop'); clearBtn.textContent = 'throw away all my drawings'; }, 4000);
      return;
    }
    clearTimeout(clearTimer);
    clearBtn.classList.remove('stop');
    clearBtn.textContent = 'throw away all my drawings';
    state.symbols.forEach(symbol => { symbol.drawings = []; });
    trained = null;
    stale = false;
    drawingsChanged();
    showReading();
    Book.toast('All drawings thrown away');
  });

  $('sy-mode').addEventListener('click', event => {
    const choice = event.target.closest('[data-copies]');
    if (!choice || busy) return;
    state.copies = choice.dataset.copies === '1';
    save();
    showMode();
  });

  function showMode() {
    $('sy-mode').querySelectorAll('.toggle').forEach(t => {
      const on = (t.dataset.copies === '1') === state.copies;
      t.classList.toggle('on', on);
      t.setAttribute('aria-pressed', on);
    });
  }

  function gather() {
    const kept = state.symbols.map((s, k) => k).filter(k => state.symbols[k].drawings.length >= MIN_EACH);
    const train = [], held = [];
    kept.forEach((k, label) => {
      state.symbols[k].drawings.forEach((image, i) => (isHeld(i) ? held : train).push({ image, label, symbol: k, n: i }));
    });
    return { kept, train, held };
  }

  let history = { train: [], held: [] };

  function drawChart() {
    Book.lineChart($('sy-chart'), {
      series: [
        { label: 'training drawings', color: '#2563eb', data: history.train },
        { label: 'held-back drawings', color: '#ea580c', data: history.held },
      ],
      xMax: EPOCHS + 1, yMin: 0, yMax: 1, yFormat: whole, xLabel: 'epoch',
    });
  }

  function cell(score) {
    return score ? `<strong>${score[0]}</strong> of ${score[1]}` : '<span class="sy-none">not run yet</span>';
  }

  function showScores() {
    const rows = { mine: 'only my drawings', copies: 'with slid copies' };
    $('sy-scores').innerHTML = MODES.map(mode => {
      const r = results[mode];
      return `<tr class="${r && r.latest ? 'latest' : ''}"><th scope="row">${rows[mode]}</th><td>${r ? r.pictures : '<span class="sy-none">–</span>'}</td>` +
        `<td>${cell(r && r.train)}</td><td>${cell(r && r.held)}</td><td>${cell(r && r.slid)}</td></tr>`;
    }).join('');
    const note = $('sy-stale');
    note.hidden = !(trained && stale);
  }

  function showWrong(sets) {
    const box = $('sy-wrong'), tiles = [];
    for (const ex of sets.held) {
      const p = forward(trained.net, ex.image), guess = best(p);
      if (guess === ex.label) continue;
      const fig = document.createElement('figure');
      fig.className = 'sy-tile';
      fig.append(bitmapCanvas(ex.image, INK));
      const cap = document.createElement('figcaption');
      cap.innerHTML = `a <b style="color:${COLORS[ex.symbol]}">${escapeHtml(nameOf(ex.symbol))}</b>, read as <b style="color:${COLORS[trained.kept[guess]]}">${escapeHtml(nameOf(trained.kept[guess]))}</b> (${whole(p[guess])} sure)`;
      fig.append(cap);
      tiles.push(fig);
    }
    box.innerHTML = '';
    if (tiles.length) box.append(...tiles);
    $('sy-wrong-note').textContent = tiles.length
      ? 'Held-back drawings it got wrong:'
      : 'It read every held-back drawing correctly. Before you celebrate, look at the last column.';
  }

  async function train() {
    if (busy) return;
    const sets = gather();
    if (sets.kept.length < MIN_SYMBOLS) return;
    busy = true;
    const btn = $('sy-train');
    btn.disabled = true;
    btn.textContent = 'training…';
    const mode = state.copies ? 'copies' : 'mine';
    const examples = state.copies ? withSlidCopies(sets.train, Math.random) : sets.train.slice();
    const net = makeNet(sets.kept.length, Math.random), slid = nudged(sets.held);
    trained = { net, kept: sets.kept };
    stale = false;
    history = { train: [countRight(net, sets.train) / sets.train.length], held: [countRight(net, sets.held) / sets.held.length] };
    $('sy-pictures').textContent = examples.length;
    $('sy-weights').textContent = (net.w1.length + net.b1.length + net.w2.length + net.b2.length).toLocaleString('en-US');
    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
      trainEpoch(net, examples, Math.random);
      history.train.push(countRight(net, sets.train) / sets.train.length);
      history.held.push(countRight(net, sets.held) / sets.held.length);
      $('sy-epoch').textContent = epoch;
      drawChart();
      await pause();
    }
    for (const m of MODES) if (results[m]) results[m].latest = false;
    results[mode] = {
      latest: true, pictures: examples.length,
      train: [countRight(net, sets.train), sets.train.length],
      held: [countRight(net, sets.held), sets.held.length],
      slid: [countRight(net, slid), slid.length],
    };
    busy = false;
    btn.textContent = 'train again from new random weights';
    renderSymbols();
    showScores();
    showWrong(sets);
    showReading();
  }
  $('sy-train').addEventListener('click', () => train());

  const tryPad = makePad($('sy-try'), () => INK, () => (trained ? 'draw one of your symbols' : 'train the network first'), () => showReading());
  const seen = $('sy-seen');

  function showReading() {
    const image = raster(tryPad.strokes), ready = trained && tryPad.strokes.length;
    paintBitmap(seen, image, rgbOf(INK));
    const bars = $('sy-bars');
    if (!trained) {
      bars.innerHTML = '';
      $('sy-guess').textContent = '?';
      $('sy-guess').style.color = '';
      $('sy-sure').textContent = 'train the network first';
      tryPad.draw();
      return;
    }
    const p = ready ? forward(trained.net, image) : null, top = ready ? best(p) : -1;
    bars.innerHTML = trained.kept.map((k, c) => `
      <div class="sy-bar${c === top ? ' top' : ''}" style="--sy:${COLORS[k]}">
        <span class="sy-bar-name">${escapeHtml(nameOf(k))}</span>
        <div class="sy-bar-track"><div class="sy-bar-fill" style="width:${ready ? p[c] * 100 : 0}%"></div></div>
        <span class="sy-bar-pct">${ready ? whole(p[c]) : '–'}</span>
      </div>`).join('');
    $('sy-guess').textContent = ready ? nameOf(trained.kept[top]) : '?';
    $('sy-guess').style.color = ready ? COLORS[trained.kept[top]] : '';
    $('sy-sure').textContent = ready ? whole(p[top]) + ' sure' : 'draw one of your symbols';
  }

  function reshape(move) {
    if (!tryPad.strokes.length) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const stroke of tryPad.strokes) {
      for (const p of stroke) {
        x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
        y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
      }
    }
    const mid = { x: (x0 + x1) / 2, y: (y0 + y1) / 2 };
    tryPad.set(tryPad.strokes.map(stroke => stroke.map(p => move(p, mid))));
  }
  const pixel = PAD / SIDE;
  $('sy-try-wipe').addEventListener('click', () => tryPad.set([]));
  $('sy-try-left').addEventListener('click', () => reshape(p => ({ x: p.x - NUDGE * pixel, y: p.y })));
  $('sy-try-right').addEventListener('click', () => reshape(p => ({ x: p.x + NUDGE * pixel, y: p.y })));
  $('sy-try-shrink').addEventListener('click', () => reshape((p, mid) => ({ x: mid.x + (p.x - mid.x) * 0.8, y: mid.y + (p.y - mid.y) * 0.8 })));

  $('sy-copies-n').textContent = COPIES;
  $('sy-slide-n').textContent = SLIDE;
  document.querySelectorAll('.sy-nudge-n').forEach(el => { el.textContent = NUDGE; });
  showMode();
  renderSymbols();
  showScores();
  drawChart();
  showReading();
})();
