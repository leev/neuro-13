(function playgroundWidget() {
  const KINDS = ['blobs', 'circle', 'xor', 'moons', 'spiral', 'draw'];
  const ACTS = ['tanh', 'relu', 'sigmoid'];
  const RULES = ['plain', 'adam'];
  const LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];
  const BATCHES = [1, 2, 4, 8, 16, 32, 0];
  const DECAYS = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1];
  const DOTS = [20, 40, 60, 100, 150, 200, 300, 400];
  const MAX_LAYERS = 4, MAX_NEURONS = 16, MAX_NOISE = 30, TEST_N = 200, DRAW_MAX = 480, SCALE = 2, RANGE = 1.2 * SCALE;
  const DEFAULTS = { kind: 'circle', noise: 0, dots: 200, hidden: [4], act: 'tanh', rule: 'plain', lr: 0.1, batch: 16, decay: 0, seed: 1, dseed: 1 };

  // Own exp, tanh, log and cosine from + - * / only, which every browser rounds identically. Math.tanh and
  // friends differ in the last digit between browsers, and training magnifies that into different results.
  const LN2_HI = 0.6931471803691238, LN2_LO = 1.9082149292705877e-10, HALF_PI = Math.PI / 2;
  const E2 = 1 / 2, E3 = 1 / 6, E4 = 1 / 24, E5 = 1 / 120, E6 = 1 / 720, E7 = 1 / 5040, E8 = 1 / 40320;
  const E9 = 1 / 362880, E10 = 1 / 3628800, E11 = 1 / 39916800, E12 = 1 / 479001600;
  const POW2 = new Float64Array(1090), tmp = new Float64Array(MAX_NEURONS), one = new Float64Array(1), terms = new Float64Array(DRAW_MAX);
  POW2[64] = 1;
  for (let i = 65; i < POW2.length; i++) POW2[i] = POW2[i - 1] * 2;
  for (let i = 63; i >= 0; i--) POW2[i] = POW2[i + 1] / 2;

  function expm1All(buf, n) {
    for (let i = 0; i < n; i++) {
      const x = buf[i];
      if (!(x < 709)) {
        buf[i] = x !== x ? x : Infinity;
      } else if (x < -40) {
        buf[i] = -1;
      } else {
        const k = Math.round(x * 1.4426950408889634), r = x - k * LN2_HI - k * LN2_LO;
        const p = r * (1 + r * (E2 + r * (E3 + r * (E4 + r * (E5 + r * (E6 + r * (E7 + r * (E8 + r * (E9 + r * (E10 + r * (E11 + r * E12)))))))))));
        buf[i] = k === 0 ? p : POW2[k + 64] * (1 + p) - 1;
      }
    }
  }

  function squashAll(buf, n) {
    for (let i = 0; i < n; i++) {
      const a = buf[i] < 0 ? -buf[i] : buf[i];
      tmp[i] = a > 20 ? 40 : 2 * a;
    }
    expm1All(tmp, n);
    for (let i = 0; i < n; i++) {
      const y = tmp[i] / (tmp[i] + 2);
      buf[i] = buf[i] < 0 ? -y : y;
    }
  }

  function sigmoidAll(buf, n) {
    for (let i = 0; i < n; i++) tmp[i] = -buf[i];
    expm1All(tmp, n);
    for (let i = 0; i < n; i++) buf[i] = 1 / (2 + tmp[i]);
  }

  function lnAll(buf, n) {
    for (let i = 0; i < n; i++) {
      let v = buf[i], e = 0, sum = 0;
      if (!(v > 0) || v === Infinity) {
        buf[i] = v === 0 ? -Infinity : v > 0 ? v : NaN;
        continue;
      }
      while (v < 0.7071067811865476) { v *= 2; e--; }
      while (v > 1.4142135623730951) { v /= 2; e++; }
      const s = (v - 1) / (v + 1), q = s * s;
      for (let m = 21; m >= 1; m -= 2) sum = 1 / m + q * sum;
      buf[i] = e * LN2_HI + (e * LN2_LO + 2 * s * sum);
    }
  }

  function cosine(x) {
    const k = Math.round(x / HALF_PI), r = x - k * 1.5707963267341256 - k * 6.077100506506192e-11, q = r * r, turn = ((k % 4) + 4) % 4;
    let s = 1, c = 1;
    for (let m = 18; m >= 2; m -= 2) s = 1 - q * s / (m * (m + 1));
    for (let m = 17; m >= 1; m -= 2) c = 1 - q * c / (m * (m + 1));
    return turn === 0 ? c : turn === 1 ? -r * s : turn === 2 ? -c : r * s;
  }
  const sine = x => cosine(x - HALF_PI);

  function gaussian(rnd) {
    one[0] = 1 - rnd();
    lnAll(one, 1);
    return Math.sqrt(-2 * one[0]) * cosine(2 * Math.PI * rnd());
  }

  function makeSet(capacity) {
    return { n: 0, x: new Float64Array(2 * capacity), y: new Uint8Array(capacity), flip: new Uint8Array(capacity) };
  }

  function fillDots(set, kind, n, wrong, seed) {
    const rnd = Book.rng(seed), half = n / 2;
    set.n = n;
    for (let i = 0; i < n; i++) {
      const c = i % 2, u = rnd();
      let px = 0, py = 0, label = c, spread = 0;
      if (kind === 'blobs') {
        px = c ? 0.45 : -0.45;
        py = px;
        spread = 0.16;
      } else if (kind === 'circle') {
        const r = c ? 0.42 * Math.sqrt(u) : 0.68 + 0.27 * u, t = 2 * Math.PI * rnd();
        px = r * cosine(t);
        py = r * sine(t);
      } else if (kind === 'xor') {
        px = (0.08 + 0.87 * u) * (rnd() < 0.5 ? -1 : 1);
        py = (0.08 + 0.87 * rnd()) * (rnd() < 0.5 ? -1 : 1);
        label = px * py > 0 ? 1 : 0;
      } else if (kind === 'moons') {
        const t = Math.PI * u;
        px = 0.64 * (c ? 0.5 - cosine(t) : cosine(t) - 0.5);
        py = 0.64 * (c ? 0.25 - sine(t) : sine(t) - 0.25);
        spread = 0.05;
      } else {
        const along = (Math.floor(i / 2) + u) / half;
        const r = 0.06 + 0.9 * along, t = 1.75 * along * 2 * Math.PI + c * Math.PI;
        px = r * sine(t);
        py = r * cosine(t);
        spread = 0.012;
      }
      set.x[2 * i] = SCALE * (px + spread * gaussian(rnd));
      set.x[2 * i + 1] = SCALE * (py + spread * gaussian(rnd));
      set.y[i] = label;
      set.flip[i] = 0;
    }
    const pick = Book.rng(seed ^ 0x5bd1e995), rank = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(pick() * (i + 1)), t = rank[i];
      rank[i] = rank[j];
      rank[j] = t;
    }
    for (let k = 0; k < Math.round(n * wrong / 100); k++) {
      set.flip[rank[k]] = 1;
      set.y[rank[k]] ^= 1;
    }
  }

  const drawnCoord = q => -RANGE + (q + 0.5) / 128 * 2 * RANGE;
  const drawnByte = v => Math.max(0, Math.min(127, Math.floor((v + RANGE) / (2 * RANGE) * 128)));

  // A dot you draw is stored as two bytes (7 bits of x plus the color bit, then 7 bits of y), so a
  // shared link rebuilds exactly the dots its author saw. Every fourth dot is held back as a test dot.
  function fillDrawn(train, test, bytes, count) {
    train.n = test.n = 0;
    for (let i = 0; i < count; i++) {
      const set = i % 4 === 3 ? test : train, k = set.n++;
      set.x[2 * k] = drawnCoord(bytes[2 * i] & 127);
      set.x[2 * k + 1] = drawnCoord(bytes[2 * i + 1] & 127);
      set.y[k] = bytes[2 * i] >> 7;
      set.flip[k] = 0;
    }
  }

  function loadSets(train, test, s, bytes, count) {
    if (s.kind === 'draw') {
      fillDrawn(train, test, bytes, count);
    } else {
      fillDots(train, s.kind, s.dots, s.noise, s.dseed);
      fillDots(test, s.kind, TEST_N, 0, s.dseed + 7919);
    }
  }

  function createNet(hidden, act, seed) {
    const sizes = [2].concat(hidden, [1]);
    const L = sizes.length - 1, actId = ACTS.indexOf(act);
    const dice = new Int32Array([seed]);
    const roll = () => {
      const a = dice[0] = (dice[0] + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const W = [], B = [], gW = [], gB = [], mW = [], mB = [], vW = [], vB = [], A = [new Float64Array(2)], D = [new Float64Array(2)];
    const fade = new Float64Array([1, 1]);
    for (let l = 0; l < L; l++) {
      const nIn = sizes[l], nOut = sizes[l + 1];
      const reach = Math.sqrt((actId === 1 && l < L - 1 ? 6 : 3) / nIn);
      W.push(Float64Array.from({ length: nIn * nOut }, () => (roll() * 2 - 1) * reach));
      B.push(new Float64Array(nOut));
      gW.push(new Float64Array(nIn * nOut));
      gB.push(new Float64Array(nOut));
      mW.push(new Float64Array(nIn * nOut));
      mB.push(new Float64Array(nOut));
      vW.push(new Float64Array(nIn * nOut));
      vB.push(new Float64Array(nOut));
      A.push(new Float64Array(nOut));
      D.push(new Float64Array(nOut));
    }
    const net = { sizes, L, W, B, gW, gB, A, right: 0, loss: 0.5 };

    net.forward = function (xs, at) {
      A[0][0] = xs[at];
      A[0][1] = xs[at + 1];
      for (let l = 0; l < L; l++) {
        const a = A[l], out = A[l + 1], w = W[l], b = B[l], nIn = sizes[l], nOut = sizes[l + 1];
        for (let j = 0, k = 0; j < nOut; j++) {
          let z = b[j];
          for (let i = 0; i < nIn; i++, k++) z += w[k] * a[i];
          out[j] = z;
        }
        if (l === L - 1) break;
        if (actId === 0) squashAll(out, nOut);
        else if (actId === 2) sigmoidAll(out, nOut);
        else for (let j = 0; j < nOut; j++) if (!(out[j] > 0)) out[j] = 0;
      }
    };

    net.backward = function (y) {
      one[0] = A[L][0];
      sigmoidAll(one, 1);
      D[L][0] = one[0] - y;
      for (let l = L - 1; l >= 0; l--) {
        const a = A[l], d = D[l + 1], w = W[l], gw = gW[l], gb = gB[l], nIn = sizes[l], nOut = sizes[l + 1];
        for (let j = 0, k = 0; j < nOut; j++) {
          const dj = d[j];
          gb[j] += dj;
          for (let i = 0; i < nIn; i++, k++) gw[k] += dj * a[i];
        }
        if (l === 0) break;
        const before = D[l];
        for (let i = 0; i < nIn; i++) {
          let s = 0;
          for (let j = 0; j < nOut; j++) s += d[j] * w[j * nIn + i];
          const ai = a[i];
          before[i] = s * (actId === 0 ? 1 - ai * ai : actId === 1 ? (ai > 0 ? 1 : 0) : ai * (1 - ai));
        }
      }
    };

    // Adam as in Kingma and Ba (2015), with the weight decay applied to the weight itself (the AdamW way),
    // so "discourage big weights" means the same shrink-by-rate-times-decay under both learning rules.
    net.applyGrads = function (count, lr, decay, adam) {
      const keep = 1 - lr * decay, step = lr / count;
      if (adam) {
        fade[0] *= 0.9;
        fade[1] *= 0.999;
      }
      const fix1 = 1 - fade[0], fix2 = 1 - fade[1];
      for (let l = 0; l < L; l++) {
        const w = W[l], gw = gW[l], b = B[l], gb = gB[l], mw = mW[l], vw = vW[l], mb = mB[l], vb = vB[l];
        if (!adam) {
          for (let k = 0; k < w.length; k++) { w[k] = w[k] * keep - step * gw[k]; gw[k] = 0; }
          for (let j = 0; j < b.length; j++) { b[j] -= step * gb[j]; gb[j] = 0; }
          continue;
        }
        for (let k = 0; k < w.length; k++) {
          const g = gw[k] / count, m = mw[k] = 0.9 * mw[k] + 0.1 * g, v = vw[k] = 0.999 * vw[k] + 0.001 * g * g;
          w[k] = w[k] * keep - lr * (m / fix1) / (Math.sqrt(v / fix2) + 1e-8);
          gw[k] = 0;
        }
        for (let j = 0; j < b.length; j++) {
          const g = gb[j] / count, m = mb[j] = 0.9 * mb[j] + 0.1 * g, v = vb[j] = 0.999 * vb[j] + 0.001 * g * g;
          b[j] -= lr * (m / fix1) / (Math.sqrt(v / fix2) + 1e-8);
          gb[j] = 0;
        }
      }
    };

    net.trainEpoch = function (set, order, lr, batch, decay, adam) {
      const n = set.n, size = batch > 0 ? Math.min(batch, n) : n;
      // roll() is written out in place because a call that returns a fraction allocates, and training must not.
      let a = dice[0];
      for (let i = n - 1; i > 0; i--) {
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        const j = Math.floor(((t ^ (t >>> 14)) >>> 0) / 4294967296 * (i + 1)), swap = order[i];
        order[i] = order[j];
        order[j] = swap;
      }
      dice[0] = a;
      let pending = 0;
      for (let i = 0; i < n; i++) {
        const p = order[i];
        net.forward(set.x, 2 * p);
        net.backward(set.y[p]);
        if (++pending === size || i === n - 1) {
          net.applyGrads(pending, lr, decay, adam);
          pending = 0;
        }
      }
    };

    net.evaluate = function (set) {
      const out = A[L], n = set.n;
      let right = 0, loss = 0;
      for (let p = 0; p < n; p++) {
        net.forward(set.x, 2 * p);
        const z = out[0], y = set.y[p];
        if ((z > 0) === (y === 1)) right++;
        loss += (z > 0 ? z : 0) - y * z;
        terms[p] = z < 0 ? z : -z;
      }
      expm1All(terms, n);
      for (let p = 0; p < n; p++) terms[p] += 2;
      lnAll(terms, n);
      for (let p = 0; p < n; p++) loss += terms[p];
      net.right = right;
      net.loss = n ? loss / n : 0.5;
    };

    net.count = function () {
      let total = 0;
      for (let l = 0; l < L; l++) total += W[l].length + B[l].length;
      return total;
    };

    // Written as "not smaller than" so that NaN counts as exploded too.
    net.exploded = function () {
      for (let l = 0; l < L; l++) {
        const w = W[l], b = B[l];
        for (let k = 0; k < w.length; k++) if (!(Math.abs(w[k]) < 1e9)) return true;
        for (let j = 0; j < b.length; j++) if (!(Math.abs(b[j]) < 1e9)) return true;
      }
      return false;
    };

    return net;
  }

  const nearest = (list, value) => list.reduce((best, v) => Math.abs(v - value) < Math.abs(best - value) ? v : best);

  function encodeSetup(s, bytes, count) {
    const parts = ['lab', 'data=' + s.kind];
    if (s.kind === 'draw') {
      let bin = '';
      for (let i = 0; i < 2 * count; i++) bin += String.fromCharCode(bytes[i]);
      parts.push('pts=' + btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''));
    } else {
      parts.push('noise=' + s.noise, 'dots=' + s.dots, 'dseed=' + s.dseed);
    }
    parts.push('layers=' + (s.hidden.length ? s.hidden.join('-') : 'none'), 'act=' + s.act, 'rule=' + s.rule, 'lr=' + s.lr,
      'batch=' + (s.batch || 'all'), 'decay=' + s.decay, 'seed=' + s.seed);
    return '#' + parts.join('&');
  }

  function decodeSetup(hash, bytes) {
    if (!/^#lab(&|$)/.test(hash)) return null;
    const q = new URLSearchParams(hash.slice(1));
    const num = (key, fallback) => {
      const v = parseFloat(q.get(key));
      return Number.isFinite(v) ? v : fallback;
    };
    const s = Object.assign({}, DEFAULTS, { count: 0 });
    if (KINDS.includes(q.get('data'))) s.kind = q.get('data');
    if (ACTS.includes(q.get('act'))) s.act = q.get('act');
    if (RULES.includes(q.get('rule'))) s.rule = q.get('rule');
    s.noise = Math.max(0, Math.min(MAX_NOISE, Math.round(num('noise', s.noise))));
    s.dots = nearest(DOTS, num('dots', s.dots));
    s.lr = nearest(LRS, num('lr', s.lr));
    s.batch = q.get('batch') === 'all' ? 0 : nearest(BATCHES.slice(0, -1), num('batch', s.batch));
    s.decay = nearest(DECAYS, num('decay', s.decay));
    s.seed = Math.abs(Math.round(num('seed', s.seed))) % 1e9;
    s.dseed = Math.abs(Math.round(num('dseed', s.dseed))) % 1e9;
    if (q.has('layers')) {
      s.hidden = q.get('layers').split('-').map(v => parseInt(v, 10)).filter(Number.isFinite)
        .slice(0, MAX_LAYERS).map(v => Math.max(1, Math.min(MAX_NEURONS, v)));
    }
    if (s.kind === 'draw') {
      let bin = '';
      try { bin = atob((q.get('pts') || '').replace(/-/g, '+').replace(/_/g, '/')); } catch (err) { bin = ''; }
      s.count = Math.min(DRAW_MAX, bin.length >> 1);
      for (let i = 0; i < 2 * s.count; i++) bytes[i] = bin.charCodeAt(i) & 255;
    }
    return s;
  }

  const lab = document.getElementById('pg-lab');
  if (!lab) return;
  const $ = id => document.getElementById(id);
  const VIEW = 380, FIELD_MAX = 190, TILE = 20, ATLAS_COLS = 8, HISTORY_MAX = 480, SLOTS = MAX_LAYERS + 2;
  const ORANGE = '#ea580c', BLUE = '#2563eb', INK = '#1d2238';
  const fontOf = name => getComputedStyle(lab).getPropertyValue(name).trim() || 'system-ui, sans-serif';
  const MONO = '600 11px ' + fontOf('--font-mono'), HINT = '600 15px ' + fontOf('--font-body');

  const state = Object.assign({}, DEFAULTS, { hidden: DEFAULTS.hidden.slice() });
  const train = makeSet(DRAW_MAX), test = makeSet(TEST_N), order = new Int32Array(DRAW_MAX), drawn = new Uint8Array(2 * DRAW_MAX);
  const hist = { trainAcc: [], testAcc: [], trainLoss: [], testLoss: [] };
  const colX = new Float64Array(SLOTS), colSize = new Float64Array(SLOTS), nodeY = new Float64Array(SLOTS * MAX_NEURONS), probe = new Float64Array(2);
  let net = null, drawnCount = 0, epoch = 0, stride = 1, playing = false, broken = false, resumeOnShow = false;
  let speed = 1, tick = 0, rafId = 0, statsAt = 0, fieldRes = FIELD_MAX, netW = 0, netH = 0, nctx = null;
  let pinned = -1, hovered = -1, showTest = false, brush = 1, chartMode = 'acc', stroking = false, lastX = 0, lastY = 0;
  let netNote = '', viewNote = '';

  const view = $('pg-view'), netCanvas = $('pg-netcanvas'), chart = $('pg-chart'), alertBox = $('pg-alert');
  const vctx = Book.hidpi(view).ctx;

  function offscreen(w, h) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    return { canvas, ctx, img: ctx.createImageData(w, h) };
  }
  const field = offscreen(FIELD_MAX, FIELD_MAX), zoom = offscreen(FIELD_MAX, FIELD_MAX);
  const atlas = offscreen(ATLAS_COLS * TILE, Math.ceil((3 + MAX_LAYERS * MAX_NEURONS) / ATLAS_COLS) * TILE);

  function makeLut(orange, blue, floor) {
    const lut = new Uint8Array(768);
    for (let i = 0; i < 256; i++) {
      const v = i / 127.5 - 1, tint = v > 0 ? orange : blue, strength = floor + (1 - floor) * Math.abs(v);
      for (let c = 0; c < 3; c++) lut[3 * i + c] = Math.round(255 + (tint[c] - 255) * strength);
    }
    return lut;
  }
  const fieldLut = makeLut([253, 186, 116], [147, 197, 253], 0.16), mapLut = makeLut([251, 146, 60], [96, 165, 250], 0);

  const layerOf = id => Math.floor(id / MAX_NEURONS);
  const alive = id => id >= 0 && layerOf(id) <= net.L && id % MAX_NEURONS < net.sizes[layerOf(id)];

  // Whole-number arguments only: a fraction passed into a function costs an allocation, and this runs per pixel.
  function shade(data, at, lut, l, j) {
    let v = net.A[l][j];
    if (l === 0) {
      v /= RANGE;
    } else if (l === net.L) {
      one[0] = v / 2;
      squashAll(one, 1);
      v = one[0];
    }
    const i = 3 * (v >= 1 ? 255 : v <= -1 ? 0 : ((v + 1) * 127.5) | 0);
    data[at] = lut[i];
    data[at + 1] = lut[i + 1];
    data[at + 2] = lut[i + 2];
    data[at + 3] = 255;
  }

  function computeFields() {
    const G = fieldRes, edge = Math.min(FIELD_MAX, G + 1), main = field.img.data, side = zoom.img.data;
    const second = hovered >= 0 && hovered !== pinned, shown = pinned < 0 ? net.L * MAX_NEURONS : pinned;
    const lut = pinned < 0 ? fieldLut : mapLut, l = layerOf(shown), j = shown % MAX_NEURONS;
    const hl = second ? layerOf(hovered) : 0, hj = second ? hovered % MAX_NEURONS : 0;
    for (let gy = 0; gy < edge; gy++) {
      probe[1] = RANGE - (gy + 0.5) / G * 2 * RANGE;
      for (let gx = 0, at = 4 * gy * FIELD_MAX; gx < edge; gx++, at += 4) {
        probe[0] = -RANGE + (gx + 0.5) / G * 2 * RANGE;
        net.forward(probe, 0);
        shade(main, at, lut, l, j);
        if (second) shade(side, at, mapLut, hl, hj);
      }
    }
    field.ctx.putImageData(field.img, 0, 0, 0, 0, edge, edge);
    if (second) zoom.ctx.putImageData(zoom.img, 0, 0, 0, 0, edge, edge);
  }

  function computeTiles() {
    const data = atlas.img.data, row = 4 * atlas.canvas.width;
    for (let gy = 0; gy < TILE; gy++) {
      probe[1] = RANGE - (gy + 0.5) / TILE * 2 * RANGE;
      for (let gx = 0; gx < TILE; gx++) {
        probe[0] = -RANGE + (gx + 0.5) / TILE * 2 * RANGE;
        net.forward(probe, 0);
        for (let l = 0, t = 0; l <= net.L; l++) {
          for (let j = 0; j < net.sizes[l]; j++, t++) {
            shade(data, (Math.floor(t / ATLAS_COLS) * TILE + gy) * row + 4 * ((t % ATLAS_COLS) * TILE + gx), mapLut, l, j);
          }
        }
      }
    }
    atlas.ctx.putImageData(atlas.img, 0, 0);
  }

  const toPx = v => (v + RANGE) / (2 * RANGE) * VIEW;

  function trace(ctx, set, label, flipped, r, scale, ox, oy) {
    ctx.beginPath();
    for (let p = 0; p < set.n; p++) {
      if (flipped ? !set.flip[p] : set.y[p] !== label) continue;
      const x = ox + toPx(set.x[2 * p]) * scale, y = oy + (VIEW - toPx(set.x[2 * p + 1])) * scale;
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, 6.2832);
    }
  }

  function drawView() {
    const ctx = vctx, mid = VIEW / 2 + 0.5;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(field.canvas, 0, 0, fieldRes, fieldRes, 0, 0, VIEW, VIEW);
    ctx.strokeStyle = 'rgba(27, 31, 39, 0.14)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mid, 0);
    ctx.lineTo(mid, VIEW);
    ctx.moveTo(0, mid);
    ctx.lineTo(VIEW, mid);
    ctx.stroke();
    ctx.fillStyle = 'rgba(27, 31, 39, 0.45)';
    ctx.font = MONO;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    ctx.fillText('−2', toPx(-2), mid + 3);
    ctx.fillText('2', toPx(2), mid + 3);
    ctx.textAlign = 'left';
    ctx.fillText('2', mid + 4, VIEW - toPx(2) - 5);
    ctx.fillText('−2', mid + 4, VIEW - toPx(-2) - 5);
    ctx.fillText('x', VIEW - 12, mid + 3);
    ctx.fillText('y', mid + 4, 3);

    if (net.L === 1 && pinned < 0) {
      const wx = net.W[0][0], wy = net.W[0][1], b = net.B[0][0], steep = Math.abs(wy) < Math.abs(wx);
      const at = v => steep ? -(wy * v + b) / wx : -(wx * v + b) / wy;
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      if (steep) {
        ctx.moveTo(toPx(at(-RANGE)), VIEW);
        ctx.lineTo(toPx(at(RANGE)), 0);
      } else {
        ctx.moveTo(0, VIEW - toPx(at(-RANGE)));
        ctx.lineTo(VIEW, VIEW - toPx(at(RANGE)));
      }
      if (wx || wy) ctx.stroke();
      ctx.setLineDash([]);
    }

    for (let label = 0; label < 2; label++) {
      trace(ctx, train, label, false, 3.6, 1, 0, 0);
      ctx.fillStyle = label ? ORANGE : BLUE;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    trace(ctx, train, 0, true, 6, 1, 0, 0);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (showTest) {
      for (let label = 0; label < 2; label++) {
        trace(ctx, test, label, false, 3, 1, 0, 0);
        ctx.fillStyle = '#fff';
        ctx.fill();
        ctx.strokeStyle = label ? ORANGE : BLUE;
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
    }
    if (state.kind === 'draw' && !drawnCount) {
      ctx.fillStyle = 'rgba(27, 31, 39, 0.6)';
      ctx.font = HINT;
      ctx.textAlign = 'center';
      ctx.fillText('Click or drag here to drop dots', VIEW / 2, VIEW / 2 - 40);
    }
  }

  function layoutDiagram() {
    const cols = net.L + 1, slot = (netW - 12) / cols, top = 8, bottom = netH - 24;
    for (let l = 0; l < cols; l++) {
      const n = net.sizes[l], room = (bottom - top) / n;
      const size = Math.round(Math.max(8, Math.min(52, room - 3, slot * 0.58)));
      const gap = Math.min(14, room - size), first = top + (bottom - top - n * size - (n - 1) * gap) / 2;
      colSize[l] = size;
      colX[l] = Math.round(6 + slot * (l + 0.5) - size / 2);
      for (let j = 0; j < n; j++) nodeY[l * MAX_NEURONS + j] = Math.round(first + j * (size + gap));
    }
    const box = $('pg-cols');
    box.textContent = '';
    box.classList.toggle('tight', slot < 86);
    for (let l = 1; l < net.L; l++) {
      const pill = document.createElement('div'), n = net.sizes[l];
      pill.className = 'pg-col';
      pill.style.left = Math.round(colX[l] + colSize[l] / 2) + 'px';
      pill.innerHTML =
        `<button type="button" data-layer="${l - 1}" data-by="-1" aria-label="remove a neuron from hidden layer ${l}"${n <= 1 ? ' disabled' : ''}>−</button>` +
        `<span title="neurons in hidden layer ${l}">${n}</span>` +
        `<button type="button" data-layer="${l - 1}" data-by="1" aria-label="add a neuron to hidden layer ${l}"${n >= MAX_NEURONS ? ' disabled' : ''}>+</button>`;
      box.appendChild(pill);
    }
  }

  function nodeAt(px, py) {
    for (let l = 0; l <= net.L; l++) {
      const size = colSize[l];
      if (px < colX[l] - 3 || px > colX[l] + size + 3) continue;
      for (let j = 0; j < net.sizes[l]; j++) {
        const y = nodeY[l * MAX_NEURONS + j];
        if (py >= y - 1.5 && py <= y + size + 1.5) return l * MAX_NEURONS + j;
      }
    }
    return -1;
  }

  function drawDiagram() {
    const ctx = nctx, focus = hovered >= 0 ? hovered : pinned;
    ctx.clearRect(0, 0, netW, netH);
    ctx.lineCap = 'round';
    for (let l = 0; l < net.L; l++) {
      const w = net.W[l], nIn = net.sizes[l], nOut = net.sizes[l + 1];
      const x0 = colX[l] + colSize[l], x1 = colX[l + 1], bend = (x0 + x1) / 2, widest = Math.min(4.2, 48 / Math.max(nIn, nOut));
      for (let j = 0, k = 0; j < nOut; j++) {
        const y1 = nodeY[(l + 1) * MAX_NEURONS + j] + colSize[l + 1] / 2;
        for (let i = 0; i < nIn; i++, k++) {
          const v = w[k], size = v < 0 ? -v : v;
          if (!(size > 0.005)) continue;
          const lit = focus < 0 || focus === l * MAX_NEURONS + i || focus === (l + 1) * MAX_NEURONS + j;
          const y0 = nodeY[l * MAX_NEURONS + i] + colSize[l] / 2, strength = Math.tanh(size / 3);
          ctx.globalAlpha = lit ? 0.1 + 0.7 * strength : 0.04;
          ctx.lineWidth = 0.4 + widest * strength;
          ctx.strokeStyle = v < 0 ? BLUE : ORANGE;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.bezierCurveTo(bend, y0, bend, y1, x1, y1);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.imageSmoothingEnabled = true;
    for (let l = 0, t = 0; l <= net.L; l++) {
      const size = colSize[l], x = colX[l];
      for (let j = 0; j < net.sizes[l]; j++, t++) {
        const id = l * MAX_NEURONS + j, y = nodeY[id], on = id === pinned || id === hovered;
        ctx.drawImage(atlas.canvas, (t % ATLAS_COLS) * TILE + 0.5, Math.floor(t / ATLAS_COLS) * TILE + 0.5, TILE - 1, TILE - 1, x, y, size, size);
        ctx.lineWidth = on ? 2.5 : 1;
        ctx.strokeStyle = on ? INK : '#94a3b8';
        ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
      }
    }
    ctx.font = MONO;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('in', colX[0] + colSize[0] / 2, nodeY[1] + colSize[0] + 5);
    ctx.fillText('out', colX[net.L] + colSize[net.L] / 2, nodeY[net.L * MAX_NEURONS] + colSize[net.L] + 5);
    ctx.fillStyle = INK;
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.strokeText('x', colX[0] + colSize[0] / 2, nodeY[0] + colSize[0] / 2 - 6);
    ctx.fillText('x', colX[0] + colSize[0] / 2, nodeY[0] + colSize[0] / 2 - 6);
    ctx.strokeText('y', colX[0] + colSize[0] / 2, nodeY[1] + colSize[0] / 2 - 6);
    ctx.fillText('y', colX[0] + colSize[0] / 2, nodeY[1] + colSize[0] / 2 - 6);

    if (hovered < 0) return;
    const l = layerOf(hovered), size = colSize[l], big = Math.min(136, netH - 12, netW * 0.45);
    let px = colX[l] + size + 10;
    if (px + big > netW - 4) px = Math.max(4, colX[l] - 10 - big);
    const py = Math.max(4, Math.min(netH - big - 4, nodeY[hovered] + size / 2 - big / 2));
    ctx.shadowColor = 'rgba(27, 31, 39, 0.32)';
    ctx.shadowBlur = 18;
    ctx.fillStyle = '#fff';
    ctx.fillRect(px - 3, py - 3, big + 6, big + 6);
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.drawImage(hovered === pinned ? field.canvas : zoom.canvas, 0, 0, fieldRes, fieldRes, px, py, big, big);
    for (let label = 0; label < 2; label++) {
      trace(ctx, train, label, false, 1.5, big / VIEW, px, py);
      ctx.fillStyle = label ? ORANGE : BLUE;
      ctx.fill();
    }
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = INK;
    ctx.strokeRect(px - 0.5, py - 0.5, big + 1, big + 1);
  }

  const pct = v => Math.round(100 * v) + '%';
  const plain = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2);
  const signed = v => (v < 0 ? '− ' : '+ ') + Math.abs(v).toFixed(2);

  function nodeName(id) {
    const l = layerOf(id), j = id % MAX_NEURONS;
    if (l === 0) return j ? 'Input y: how far up the dot is.' : 'Input x: how far right the dot is.';
    const bias = ' Its bias is ' + plain(net.B[l - 1][j]) + '.';
    return (l === net.L ? 'The output neuron.' : `Hidden layer ${l}, neuron ${j + 1}.`) + bias;
  }

  function describe() {
    const id = hovered >= 0 ? hovered : pinned;
    let note = 'Point at a neuron to enlarge its map. Click one to pin it to the big map.';
    if (id >= 0) note = nodeName(id);
    else if (net.L === 1) note = `No hidden layers: this is the single neuron from Part 1. Its dashed line is where ${plain(net.W[0][0])} × x ${signed(net.W[0][1])} × y ${signed(net.B[0][0])} = 0.`;
    if (note !== netNote) $('pg-netnote').textContent = netNote = note;
    note = pinned < 0
      ? 'The background is the network’s answer at every point: orange where it says orange, blue where it says blue, pale where it is unsure.'
      : `Pinned: ${nodeName(pinned)} The background is now this one neuron’s output. Click it again to let go.`;
    if (note !== viewNote) $('pg-viewnote').textContent = viewNote = note;
  }

  function drawChart() {
    const loss = chartMode === 'loss';
    let top = 1;
    if (loss) {
      for (let i = 0; i < hist.trainLoss.length; i++) top = Math.max(top, hist.trainLoss[i], hist.testLoss[i]);
      top = Math.min(8, top * 1.05);
    }
    Book.lineChart(chart, {
      series: [
        { color: '#1d2238', data: !train.n ? [] : loss ? hist.trainLoss : hist.trainAcc },
        { color: '#db2777', data: !test.n ? [] : loss ? hist.testLoss : hist.testAcc },
      ],
      yMin: 0, yMax: top, yFormat: loss ? undefined : pct,
      xLabel: epoch ? 'epochs 0 to ' + epoch.toLocaleString('en-US') : 'epochs',
    });
  }

  function showStats() {
    $('pg-epoch').textContent = epoch.toLocaleString('en-US');
    net.evaluate(train);
    $('pg-train-acc').textContent = train.n && !broken ? pct(net.right / train.n) : '–';
    $('pg-train-loss').textContent = train.n && !broken ? net.loss.toFixed(3) : '–';
    net.evaluate(test);
    $('pg-test-acc').textContent = test.n && !broken ? pct(net.right / test.n) : '–';
    $('pg-test-loss').textContent = test.n && !broken ? net.loss.toFixed(3) : '–';
    drawChart();
    describe();
  }

  function render() {
    if (!net) return;
    fieldRes = Math.round(Math.max(40, Math.min(playing ? 96 : FIELD_MAX, Math.sqrt((playing ? 1.4e6 : 1.2e7) / net.count()))));
    computeFields();
    computeTiles();
    drawView();
    drawDiagram();
    const now = performance.now();
    if (playing && now - statsAt < 140) return;
    statsAt = now;
    showStats();
  }

  function record() {
    net.evaluate(train);
    hist.trainAcc.push(train.n ? net.right / train.n : 0.5);
    hist.trainLoss.push(net.loss);
    net.evaluate(test);
    hist.testAcc.push(test.n ? net.right / test.n : 0.5);
    hist.testLoss.push(net.loss);
    if (hist.trainAcc.length <= HISTORY_MAX) return;
    for (const key in hist) {
      const line = hist[key];
      for (let i = 0; 2 * i < line.length; i++) line[i] = line[2 * i];
      line.length = Math.ceil(line.length / 2);
    }
    stride *= 2;
  }

  function trainOne() {
    net.trainEpoch(train, order, state.lr, state.batch, state.decay, state.rule === 'adam');
    epoch++;
    if (net.exploded()) {
      broken = true;
      setPlaying(false);
      alertBox.textContent = 'Boom. The weights exploded: some are now bigger than a billion, or too big for the computer to store at all. The learning rate is far too high for this network. Lower it, then press reset.';
      alertBox.hidden = false;
    } else if (epoch % stride === 0) {
      record();
    }
  }

  function frame() {
    rafId = 0;
    if (!playing) return;
    rafId = requestAnimationFrame(frame);
    if (speed === 0 && ++tick % 8) return;
    const start = performance.now(), want = speed === 2 ? 12 : 1;
    let count = 0;
    do {
      trainOne();
      count++;
    } while (playing && count < want && performance.now() - start < 12);
    if (playing) render();
  }

  function setPlaying(on) {
    if (on && broken) return;
    if (on && train.n < 2) {
      Book.toast('Drop some dots on the map first.');
      return;
    }
    playing = on;
    $('pg-play').textContent = on ? '❚❚ pause' : '▶ play';
    if (on && !rafId) rafId = requestAnimationFrame(frame);
    if (on) return;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    render();
  }

  function resetOrder() {
    for (let i = 0; i < order.length; i++) order[i] = i;
  }

  function restart(newDots) {
    if (newDots) loadSets(train, test, state, drawn, drawnCount);
    net = createNet(state.hidden, state.act, state.seed);
    resetOrder();
    epoch = 0;
    stride = 1;
    broken = false;
    hovered = -1;
    if (!alive(pinned)) pinned = -1;
    for (const key in hist) hist[key].length = 0;
    alertBox.hidden = true;
    record();
    fitDiagram();
    layoutDiagram();
    $('pg-count').textContent = net.count().toLocaleString('en-US');
    statsAt = 0;
    if (playing && train.n < 2) setPlaying(false);
    else render();
  }

  function mark(group, key, value) {
    $(group).querySelectorAll('button').forEach(b => {
      const on = b.dataset[key] === String(value);
      b.classList.toggle('on', on);
      b.setAttribute('aria-pressed', on);
    });
  }

  function setSlider(id, value, text) {
    $(id).value = value;
    $(id).setAttribute('aria-valuetext', text);
    $(id + '-v').textContent = text;
  }

  function syncControls() {
    const drawing = state.kind === 'draw';
    mark('pg-kinds', 'kind', state.kind);
    mark('pg-act', 'act', state.act);
    mark('pg-rule', 'rule', state.rule);
    mark('pg-speed', 'speed', speed);
    mark('pg-brush', 'brush', brush);
    mark('pg-chartmode', 'mode', chartMode);
    $('pg-draw-tools').hidden = !drawing;
    $('pg-gen-tools').hidden = drawing;
    view.classList.toggle('pg-drawable', drawing);
    setSlider('pg-noise', state.noise, state.noise + '%');
    setSlider('pg-dots', DOTS.indexOf(state.dots), state.dots);
    setSlider('pg-lr', LRS.indexOf(state.lr), state.lr);
    setSlider('pg-batch', BATCHES.indexOf(state.batch), state.batch || 'all');
    setSlider('pg-decay', DECAYS.indexOf(state.decay), state.decay || 'off');
    $('pg-layer-count').textContent = state.hidden.length;
    $('pg-layer-less').disabled = !state.hidden.length;
    $('pg-layer-more').disabled = state.hidden.length >= MAX_LAYERS;
    $('pg-showtest').checked = showTest;
    $('pg-drawn').textContent = drawnCount + ' of ' + DRAW_MAX + ' dots';
  }

  function applySetup(s) {
    setPlaying(false);
    for (const key in DEFAULTS) state[key] = key === 'hidden' ? s.hidden.slice() : s[key];
    if (s.kind === 'draw') {
      drawnCount = s.count;
      showTest = true;
    }
    pinned = -1;
    syncControls();
    restart(true);
  }

  function dropDot(e) {
    const p = Book.pointer(view, e);
    if (Math.hypot(p.x - lastX, p.y - lastY) < 11) return;
    if (drawnCount >= DRAW_MAX) {
      stroking = false;
      Book.toast('That is all the dots the lab can hold. Press clear to start again.');
      return;
    }
    lastX = p.x;
    lastY = p.y;
    drawn[2 * drawnCount] = drawnByte(p.x / VIEW * 2 * RANGE - RANGE) | brush << 7;
    drawn[2 * drawnCount + 1] = drawnByte(RANGE - p.y / VIEW * 2 * RANGE);
    drawnCount++;
    redrawn();
  }

  function redrawn() {
    loadSets(train, test, state, drawn, drawnCount);
    resetOrder();
    if (!epoch) {
      for (const key in hist) hist[key].length = 0;
      record();
    }
    $('pg-drawn').textContent = drawnCount + ' of ' + DRAW_MAX + ' dots';
    statsAt = 0;
    if (playing && train.n < 2) setPlaying(false);
    else if (!playing) render();
  }

  function fitDiagram() {
    const w = Math.max(240, Math.floor($('pg-diagram').clientWidth));
    const stacked = getComputedStyle(lab.querySelector('.pg-grid')).gridTemplateColumns.split(' ').length < 2;
    const h = stacked ? Math.min(w < 400 ? 320 : 380, 44 + 48 * Math.max(4, ...state.hidden)) : 380;
    chart.dataset.w = Math.max(240, Math.floor(chart.parentNode.clientWidth));
    chart.dataset.h = 170;
    if (w === netW && h === netH) return false;
    netW = w;
    netH = h;
    netCanvas.dataset.w = w;
    netCanvas.dataset.h = h;
    nctx = Book.hidpi(netCanvas).ctx;
    return true;
  }

  function thumbnails() {
    const tiny = makeSet(80);
    $('pg-kinds').querySelectorAll('button').forEach(b => {
      const { ctx, w } = Book.hidpi(b.querySelector('canvas'));
      if (b.dataset.kind === 'draw') {
        tiny.n = 13;
        for (let i = 0; i < 13; i++) {
          const eye = i >= 9, t = Math.PI * (1.15 + 0.7 * i / 8);
          tiny.x[2 * i] = eye ? (i % 2 ? 0.75 : -0.75) : 1.45 * Math.cos(t);
          tiny.x[2 * i + 1] = eye ? (i > 10 ? 1.05 : 0.7) : 0.35 + 1.45 * Math.sin(t);
          tiny.y[i] = eye ? 0 : 1;
        }
      } else {
        fillDots(tiny, b.dataset.kind, 80, 0, 5);
      }
      for (let label = 0; label < 2; label++) {
        trace(ctx, tiny, label, false, 1.7, w / VIEW, 0, 0);
        ctx.fillStyle = label ? ORANGE : BLUE;
        ctx.fill();
      }
    });
  }

  $('pg-play').addEventListener('click', () => setPlaying(!playing));
  $('pg-step').addEventListener('click', () => {
    if (broken) return;
    if (playing) setPlaying(false);
    if (train.n < 2) return Book.toast('Drop some dots on the map first.');
    trainOne();
    statsAt = 0;
    render();
  });
  $('pg-reset').addEventListener('click', () => {
    setPlaying(false);
    restart(false);
  });
  $('pg-newstart').addEventListener('click', () => {
    setPlaying(false);
    state.seed = 1 + Math.floor(Math.random() * 99999);
    restart(false);
    Book.toast('New random starting weights.');
  });
  $('pg-newdots').addEventListener('click', () => {
    state.dseed = 1 + Math.floor(Math.random() * 99999);
    restart(true);
  });
  $('pg-speed').addEventListener('click', e => {
    const b = e.target.closest('[data-speed]');
    if (!b) return;
    speed = +b.dataset.speed;
    mark('pg-speed', 'speed', speed);
  });
  $('pg-kinds').addEventListener('click', e => {
    const b = e.target.closest('[data-kind]');
    if (!b) return;
    state.kind = b.dataset.kind;
    if (state.kind === 'draw') showTest = true;
    syncControls();
    restart(true);
  });
  $('pg-act').addEventListener('click', e => {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    state.act = b.dataset.act;
    syncControls();
    restart(false);
  });
  $('pg-rule').addEventListener('click', e => {
    const b = e.target.closest('[data-rule]');
    if (!b || b.dataset.rule === state.rule) return;
    state.rule = b.dataset.rule;
    const adam = state.rule === 'adam';
    if (adam ? state.lr > 0.03 : state.lr < 0.03) {
      state.lr = adam ? 0.01 : 0.1;
      Book.toast(adam ? 'Adam takes bigger steps, so the learning rate is now 0.01.' : 'The plain rule takes smaller steps, so the learning rate is now 0.1.');
    }
    syncControls();
    restart(false);
  });
  $('pg-layer-more').addEventListener('click', () => {
    state.hidden.push(state.hidden.length ? state.hidden[state.hidden.length - 1] : 4);
    syncControls();
    restart(false);
  });
  $('pg-layer-less').addEventListener('click', () => {
    state.hidden.pop();
    syncControls();
    restart(false);
  });
  $('pg-cols').addEventListener('click', e => {
    const b = e.target.closest('[data-layer]');
    if (!b) return;
    const l = +b.dataset.layer, by = +b.dataset.by;
    state.hidden[l] = Math.max(1, Math.min(MAX_NEURONS, state.hidden[l] + by));
    restart(false);
    const again = $('pg-cols').querySelector(`[data-layer="${l}"][data-by="${by}"]`);
    if (again && !again.disabled) again.focus();
  });
  $('pg-noise').addEventListener('input', e => {
    state.noise = +e.target.value;
    syncControls();
    restart(true);
  });
  $('pg-dots').addEventListener('input', e => {
    state.dots = DOTS[+e.target.value];
    syncControls();
    restart(true);
  });
  $('pg-lr').addEventListener('input', e => {
    state.lr = LRS[+e.target.value];
    syncControls();
  });
  $('pg-batch').addEventListener('input', e => {
    state.batch = BATCHES[+e.target.value];
    syncControls();
  });
  $('pg-decay').addEventListener('input', e => {
    state.decay = DECAYS[+e.target.value];
    syncControls();
  });
  $('pg-showtest').addEventListener('change', e => {
    showTest = e.target.checked;
    if (!playing) render();
  });
  $('pg-brush').addEventListener('click', e => {
    const b = e.target.closest('[data-brush]');
    if (!b) return;
    brush = +b.dataset.brush;
    mark('pg-brush', 'brush', brush);
  });
  $('pg-undo').addEventListener('click', () => {
    if (!drawnCount) return;
    drawnCount--;
    redrawn();
  });
  $('pg-clear').addEventListener('click', () => {
    drawnCount = 0;
    setPlaying(false);
    redrawn();
  });
  $('pg-chartmode').addEventListener('click', e => {
    const b = e.target.closest('[data-mode]');
    if (!b) return;
    chartMode = b.dataset.mode;
    mark('pg-chartmode', 'mode', chartMode);
    drawChart();
  });

  view.addEventListener('pointerdown', e => {
    if (state.kind !== 'draw') return;
    e.preventDefault();
    stroking = true;
    lastX = lastY = -99;
    dropDot(e);
    try { view.setPointerCapture(e.pointerId); } catch (err) { /* a scripted pointer has nothing to capture */ }
  });
  view.addEventListener('pointermove', e => {
    if (stroking) dropDot(e);
  });
  ['pointerup', 'pointercancel'].forEach(type => view.addEventListener(type, () => { stroking = false; }));

  netCanvas.addEventListener('pointermove', e => {
    if (e.pointerType !== 'mouse') return;
    const p = Book.pointer(netCanvas, e), id = nodeAt(p.x, p.y);
    if (id === hovered) return;
    hovered = id;
    netCanvas.style.cursor = id < 0 ? '' : 'pointer';
    statsAt = 0;
    if (!playing) render();
  });
  netCanvas.addEventListener('pointerleave', () => {
    if (hovered < 0) return;
    hovered = -1;
    statsAt = 0;
    if (!playing) render();
  });
  netCanvas.addEventListener('click', e => {
    const p = Book.pointer(netCanvas, e), id = nodeAt(p.x, p.y);
    pinned = id === pinned ? -1 : id;
    statsAt = 0;
    if (!playing) render();
  });

  $('pg-share').addEventListener('click', () => {
    history.replaceState(null, '', encodeSetup(state, drawn, drawnCount));
    const link = $('pg-link'), url = location.href;
    const byHand = () => {
      link.hidden = false;
      link.value = url;
      link.select();
      Book.toast('Copy the link from the box.');
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(() => Book.toast('Link copied. Send it to a friend.'), byHand);
    } else {
      byHand();
    }
  });

  window.addEventListener('hashchange', () => {
    const s = decodeSetup(location.hash, drawn);
    if (s) applySetup(s);
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      resumeOnShow = playing;
      if (playing) setPlaying(false);
    } else if (resumeOnShow) {
      resumeOnShow = false;
      setPlaying(true);
    }
  });
  new ResizeObserver(() => {
    if (fitDiagram()) layoutDiagram();
    statsAt = 0;
    render();
  }).observe($('pg-diagram'));

  const ticks = Book.store.get('playground.missions', {});
  const boxes = [...document.querySelectorAll('input[data-mission]')];
  const tally = () => {
    const el = $('pg-tally');
    if (el) el.textContent = boxes.filter(b => b.checked).length + ' of ' + boxes.length;
  };
  boxes.forEach(box => {
    box.checked = !!ticks[box.dataset.mission];
    box.addEventListener('change', () => {
      ticks[box.dataset.mission] = box.checked;
      Book.store.set('playground.missions', ticks);
      tally();
    });
  });
  tally();
  document.querySelectorAll('[data-setup]').forEach(b => b.addEventListener('click', () => {
    const s = decodeSetup(b.dataset.setup, drawn);
    if (!s) return;
    applySetup(s);
    lab.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    Book.toast('The lab is set up. Make your guess, then press play.');
  }));

  const shared = decodeSetup(location.hash, drawn);
  thumbnails();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
  if (shared) {
    applySetup(shared);
  } else {
    syncControls();
    restart(true);
  }
})();
