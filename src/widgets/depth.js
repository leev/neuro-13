// ===== Interactive: depth playground — two tangled spirals =====
(function() {
  const canvas = document.getElementById('dp-canvas');
  if (!canvas) return;
  const { ctx } = Book.hidpi(canvas);
  const SIZE = 280, RANGE = 1.3, LR = 1.5;

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // two tangled spirals, deterministic so the lesson always reproduces
  const data = (function() {
    const rnd = mulberry32(7), pts = [], N = 120;
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < N; i++) {
        const r = (i + 1) / N;
        const t = 1.75 * (i / N) * 2 * Math.PI + c * Math.PI;
        pts.push({
          x: [r * Math.sin(t) + (rnd() - 0.5) * 0.12, r * Math.cos(t) + (rnd() - 0.5) * 0.12],
          y: c,
        });
      }
    }
    return pts;
  })();

  let net, epoch, dump, thumbs;

  function makeNet(sizes) {
    const rnd = mulberry32(42);
    const Ws = [], Bs = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fan = sizes[l];
      Ws.push(Array.from({length: fan}, () =>
        Array.from({length: sizes[l + 1]}, () => (rnd() * 2 - 1) * Math.sqrt(1 / fan))));
      Bs.push(Array(sizes[l + 1]).fill(0));
    }
    return { sizes, Ws, Bs };
  }

  function forward(x) {
    const acts = [x];
    for (let l = 0; l < net.Ws.length; l++) {
      const inp = acts[l], W = net.Ws[l], B = net.Bs[l], out = [];
      const last = l === net.Ws.length - 1;
      for (let j = 0; j < B.length; j++) {
        let z = B[j];
        for (let i = 0; i < inp.length; i++) z += inp[i] * W[i][j];
        out.push(last ? sigmoid(z) : Math.tanh(z));
      }
      acts.push(out);
    }
    return acts;
  }

  function trainEpochs(n) {
    for (let e = 0; e < n; e++) {
      const gW = net.Ws.map(W => W.map(r => r.map(() => 0)));
      const gB = net.Bs.map(B => B.map(() => 0));
      for (const {x, y} of data) {
        const acts = forward(x);
        let delta = [acts[acts.length - 1][0] - y];   // sigmoid + cross-entropy
        for (let l = net.Ws.length - 1; l >= 0; l--) {
          const inp = acts[l];
          for (let j = 0; j < delta.length; j++) {
            gB[l][j] += delta[j];
            for (let i = 0; i < inp.length; i++) gW[l][i][j] += inp[i] * delta[j];
          }
          if (l > 0) {
            const nd = [];
            for (let i = 0; i < inp.length; i++) {
              let s = 0;
              for (let j = 0; j < delta.length; j++) s += delta[j] * net.Ws[l][i][j];
              nd.push(s * (1 - inp[i] * inp[i]));     // tanh derivative
            }
            delta = nd;
          }
        }
      }
      const m = data.length;
      for (let l = 0; l < net.Ws.length; l++) {
        for (let j = 0; j < net.Bs[l].length; j++) {
          net.Bs[l][j] -= LR * gB[l][j] / m;
          for (let i = 0; i < net.Ws[l].length; i++) net.Ws[l][i][j] -= LR * gW[l][i][j] / m;
        }
      }
      epoch++;
    }
    render();
  }

  function accuracy() {
    let ok = 0;
    for (const {x, y} of data) {
      const o = forward(x)[net.sizes.length - 1][0];
      if ((o > 0.5) === (y === 1)) ok++;
    }
    return ok / data.length;
  }

  function paramCount() {
    let n = 0;
    for (let l = 0; l < net.sizes.length - 1; l++) n += net.sizes[l] * net.sizes[l + 1] + net.sizes[l + 1];
    return n;
  }

  function buildThumbs() {
    const wrap = document.getElementById('dp-thumbs');
    wrap.innerHTML = '';
    thumbs = [];
    for (let hl = 1; hl <= net.sizes.length - 2; hl++) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin:6px 0;';
      const lab = document.createElement('span');
      lab.style.cssText = 'font-size:12px;color:#6b7280;width:104px;flex:0 0 auto;';
      lab.textContent = 'hidden layer ' + hl;
      row.appendChild(lab);
      for (let j = 0; j < net.sizes[hl]; j++) {
        const c = document.createElement('canvas');
        c.width = 56; c.height = 56;
        c.style.cssText = 'border:1px solid #e5e7eb;border-radius:4px;';
        row.appendChild(c);
        thumbs.push({ hl, j, tctx: c.getContext('2d') });
      }
      wrap.appendChild(row);
    }
  }

  function renderThumbs() {
    const cell = 4, n = 56 / cell;
    const grid = [];
    for (let gy = 0; gy < n; gy++) {
      for (let gx = 0; gx < n; gx++) {
        const wx = ((gx + 0.5) / n) * 2 * RANGE - RANGE;
        const wy = ((gy + 0.5) / n) * 2 * RANGE - RANGE;
        grid.push(forward([wx, wy]));
      }
    }
    for (const t of thumbs) {
      t.tctx.clearRect(0, 0, 56, 56);
      for (let gy = 0; gy < n; gy++) {
        for (let gx = 0; gx < n; gx++) {
          const a = grid[gy * n + gx][t.hl][t.j];
          t.tctx.fillStyle = a >= 0
            ? 'rgba(37, 99, 235, ' + Math.min(1, a).toFixed(2) + ')'
            : 'rgba(220, 38, 38, ' + Math.min(1, -a).toFixed(2) + ')';
          t.tctx.fillRect(gx * cell, gy * cell, cell, cell);
        }
      }
    }
  }

  function makeDumpForArch() {
    const deep = net.sizes.length === 4;
    const head = {
      w1: 'weights: input (x, y) → hidden layer 1',
      b1: 'biases: hidden layer 1',
      w2: deep ? 'weights: hidden layer 1 → hidden layer 2' : 'weights: hidden layer 1 → output (one per neuron)',
      b2: deep ? 'biases: hidden layer 2' : 'bias: the output neuron',
    };
    if (deep) {
      head.w3 = 'weights: hidden layer 2 → output (one per neuron)';
      head.b3 = 'bias: the output neuron';
    }
    dump = makeModelDump('dp-model', {
      decimals: 2,
      headComments: head,
      rowComments: { w1: ['x', 'y'] },
    });
  }

  function renderDump() {
    const m = {};
    net.Ws.forEach((W, l) => {
      const lastLayer = l === net.Ws.length - 1;
      m['w' + (l + 1)] = lastLayer ? W.map(r => r[0]) : W;   // flatten the 1-wide output column
      m['b' + (l + 1)] = net.Bs[l];
    });
    dump.update(m);
  }

  function toWorld(px) { return (px / SIZE) * 2 * RANGE - RANGE; }
  function toPx(v) { return ((v + RANGE) / (2 * RANGE)) * SIZE; }

  function render() {
    ctx.clearRect(0, 0, SIZE, SIZE);
    const cell = 5;
    for (let gy = 0; gy < SIZE; gy += cell) {
      for (let gx = 0; gx < SIZE; gx += cell) {
        const o = forward([toWorld(gx + cell / 2), toWorld(gy + cell / 2)])[net.sizes.length - 1][0];
        const conf = Math.abs(o - 0.5) * 2;
        ctx.fillStyle = o > 0.5
          ? 'rgba(234, 88, 12, ' + (0.06 + conf * 0.30).toFixed(2) + ')'
          : 'rgba(37, 99, 235, ' + (0.06 + conf * 0.30).toFixed(2) + ')';
        ctx.fillRect(gx, gy, cell, cell);
      }
    }
    for (const {x, y} of data) {
      ctx.beginPath();
      ctx.arc(toPx(x[0]), toPx(x[1]), 3.2, 0, Math.PI * 2);
      ctx.fillStyle = y === 1 ? '#ea580c' : '#2563eb';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    renderThumbs();
    renderDump();
    document.getElementById('dp-epoch').textContent = epoch;
    document.getElementById('dp-acc').textContent = (accuracy() * 100).toFixed(0) + '%';
    document.getElementById('dp-params').textContent = paramCount();
  }

  function reset() {
    const archStr = document.querySelector('#dp-arch .toggle.on').dataset.arch;
    const arch = archStr.split(',').map(Number);
    net = makeNet([2].concat(arch, [1]));
    epoch = 0;
    buildThumbs();
    makeDumpForArch();
    render();
  }

  document.querySelectorAll('#dp-arch .toggle').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('#dp-arch .toggle').forEach(o => o.classList.remove('on'));
      t.classList.add('on');
      reset();
    });
  });
  document.getElementById('dp-train1').addEventListener('click', () => trainEpochs(500));
  document.getElementById('dp-train2').addEventListener('click', () => trainEpochs(5000));
  document.getElementById('dp-reset').addEventListener('click', reset);
  document.getElementById('dp-dump-toggle').addEventListener('click', () => {
    const el = document.getElementById('dp-model');
    el.hidden = !el.hidden;
    document.getElementById('dp-dump-toggle').textContent = el.hidden ? 'show the numbers' : 'hide the numbers';
  });
  reset();
})();

