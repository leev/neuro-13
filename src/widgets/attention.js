(function whoIsItFigure() {
  const row = document.getElementById('wi-words');
  if (!row) return;
  const toggles = document.getElementById('wi-ending');
  const note = document.getElementById('wi-note');
  const START = ['The', 'trophy', "didn't", 'fit', 'in', 'the', 'suitcase', 'because', 'it', 'was', 'too'];
  const ENDINGS = {
    big: { means: 'trophy', why: 'A thing that is too big is the thing that will not go in. So <strong>it</strong> is the <strong>trophy</strong>.' },
    small: { means: 'suitcase', why: 'A thing that is too small is the thing with no room inside. So <strong>it</strong> is the <strong>suitcase</strong>.' },
  };

  function render(ending) {
    const { means, why } = ENDINGS[ending];
    row.innerHTML = [...START, ending].map(word => {
      let role = '';
      if (word === 'it') role = ' wi-asker';
      else if (word === means) role = ' wi-strong';
      else if (word === 'trophy' || word === 'suitcase') role = ' wi-weak';
      else if (word === ending) role = ' wi-clue';
      return `<span class="wi-word${role}">${word}</span>`;
    }).join('');
    note.innerHTML = why;
    toggles.querySelectorAll('.toggle').forEach(t => {
      const on = t.dataset.end === ending;
      t.classList.toggle('on', on);
      t.setAttribute('aria-pressed', String(on));
    });
  }

  toggles.addEventListener('click', e => {
    const toggle = e.target.closest('[data-end]');
    if (toggle) render(toggle.dataset.end);
  });
  toggles.addEventListener('keydown', e => {
    const toggle = e.target.closest('[data-end]');
    if (!toggle || (e.key !== 'Enter' && e.key !== ' ')) return;
    e.preventDefault();
    render(toggle.dataset.end);
  });
  render('big');
})();

(function aimTheQueryWidget() {
  const canvas = document.getElementById('ah-canvas');
  if (!canvas) return;
  const $ = id => document.getElementById(id);
  const TOKENS = [
    { name: 'red', key: [2, 1], value: [220, 38, 38] },
    { name: 'yellow', key: [-1, 2], value: [250, 204, 21] },
    { name: 'blue', key: [-1, -2], value: [37, 99, 235] },
  ];
  const REACH = 3, INK = '#1f2937';
  const rgb = v => `rgb(${v.map(Math.round).join(', ')})`;
  const signed = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
  const bracket = v => (v < 0 ? `(${signed(v)})` : signed(v));
  const whole = v => (v < 0 ? `(−${-v})` : String(v));
  let query = [1, 1], dragging = false;

  $('ah-rows').innerHTML = TOKENS.map((t, i) =>
    `<div class="ah-row"><span class="ah-name"><i class="ah-swatch" style="background:${rgb(t.value)}"></i>${t.name}</span>` +
    `<span class="ah-math" id="ah-math-${i}"></span>` +
    `<span class="ah-weight"><span class="bar-track"><span class="bar" id="ah-bar-${i}" style="background:${rgb(t.value)}"></span></span>` +
    `<span class="pct" id="ah-pct-${i}"></span></span></div>`).join('');

  function arrow(ctx, x0, y0, x1, y1, color, width) {
    const length = Math.hypot(x1 - x0, y1 - y0);
    if (length < 1) return;
    const ux = (x1 - x0) / length, uy = (y1 - y0) / length, head = Math.min(13, length);
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1 - ux * head * 0.8, y1 - uy * head * 0.8);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - ux * head - uy * head * 0.5, y1 - uy * head + ux * head * 0.5);
    ctx.lineTo(x1 - ux * head + uy * head * 0.5, y1 - uy * head - ux * head * 0.5);
    ctx.closePath();
    ctx.fill();
  }

  function draw() {
    const { ctx, w, h } = Book.hidpi(canvas);
    const cx = w / 2, cy = h / 2, unit = w / 7.4;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    for (let i = -REACH; i <= REACH; i++) {
      ctx.strokeStyle = i === 0 ? '#9ca3af' : '#e5e7eb';
      ctx.beginPath();
      ctx.moveTo(Math.round(cx + i * unit) + 0.5, cy - REACH * unit);
      ctx.lineTo(Math.round(cx + i * unit) + 0.5, cy + REACH * unit);
      ctx.moveTo(cx - REACH * unit, Math.round(cy + i * unit) + 0.5);
      ctx.lineTo(cx + REACH * unit, Math.round(cy + i * unit) + 0.5);
      ctx.stroke();
    }
    ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'middle';
    for (const t of TOKENS) {
      const x = cx + t.key[0] * unit, y = cy - t.key[1] * unit;
      arrow(ctx, cx, cy, x, y, rgb(t.value), 4);
      ctx.fillStyle = INK;
      ctx.textAlign = t.key[0] >= 0 ? 'left' : 'right';
      ctx.fillText(`${t.name} key`, x + (t.key[0] >= 0 ? 8 : -8), y + (t.key[1] >= 0 ? -9 : 11));
    }
    const qx = cx + query[0] * unit, qy = cy - query[1] * unit;
    arrow(ctx, cx, cy, qx, qy, INK, 5);
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = INK;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(qx, qy, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }

  function render() {
    const scores = TOKENS.map(t => t.key[0] * query[0] + t.key[1] * query[1]);
    const top = Math.max(...scores);
    const raised = scores.map(s => Math.exp(s - top));
    const total = raised.reduce((a, b) => a + b, 0);
    const weights = raised.map(r => r / total);
    const blend = [0, 1, 2].map(c => TOKENS.reduce((sum, t, i) => sum + weights[i] * t.value[c], 0));

    TOKENS.forEach((t, i) => {
      $(`ah-math-${i}`).innerHTML = `${whole(t.key[0])}×${bracket(query[0])} + ${whole(t.key[1])}×${bracket(query[1])} = <strong>${signed(scores[i])}</strong>`;
      $(`ah-bar-${i}`).style.width = (weights[i] * 100) + '%';
      $(`ah-pct-${i}`).textContent = weights[i].toFixed(2);
    });
    $('ah-blend-swatch').style.background = rgb(blend);
    $('ah-blend-sum').textContent = TOKENS.map((t, i) => `${weights[i].toFixed(2)} × ${t.name}`).join(' + ');
    $('ah-blend-rgb').textContent = `= red ${Math.round(blend[0])}, green ${Math.round(blend[1])}, blue ${Math.round(blend[2])}`;
    $('ah-qx').value = query[0];
    $('ah-qy').value = query[1];
    $('ah-qx-v').textContent = signed(query[0]);
    $('ah-qy-v').textContent = signed(query[1]);
    draw();
  }

  function setQuery(x, y) {
    const snap = v => Math.max(-REACH, Math.min(REACH, Math.round(v * 10) / 10));
    query = [snap(x), snap(y)];
    render();
  }

  function aim(event) {
    const p = Book.pointer(canvas, event);
    const w = +canvas.dataset.w, h = +canvas.dataset.h, unit = w / 7.4;
    setQuery((p.x - w / 2) / unit, (h / 2 - p.y) / unit);
  }

  canvas.addEventListener('pointerdown', e => {
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    aim(e);
  });
  canvas.addEventListener('pointermove', e => { if (dragging) aim(e); });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointercancel', () => { dragging = false; });
  $('ah-qx').addEventListener('input', e => setQuery(+e.target.value, query[1]));
  $('ah-qy').addEventListener('input', e => setQuery(query[0], +e.target.value));
  $('ah-presets').addEventListener('click', e => {
    const btn = e.target.closest('[data-q]');
    if (!btn) return;
    const [x, y] = btn.dataset.q.split(',').map(Number);
    setQuery(x, y);
  });
  render();
})();

(function lockerRoomWidget() {
  const trainBtn = document.getElementById('lk-train');
  if (!trainBtn) return;
  const $ = id => document.getElementById(id);
  const OWNERS = ['sam', 'mia', 'leo', 'zoe', 'ava', 'max', 'eli', 'ivy'];
  const ITEMS = ['hat', 'ball', 'book', 'kite', 'drum', 'map', 'sock', 'frog'];
  const TRAIN_LOCKERS = 3, MAX_LOCKERS = 8, BATCH = 16, LR = 0.1, STEPS = 1500;
  const CHECK_EVERY = 10, TEST_SIZE = 500, STEPS_PER_SECOND = 200, TICK_BUDGET_MS = 30, DARKEST_SCORE = 8;
  const pct = x => Math.round(x * 100) + '%';

  function createLockerModel(rand, initScale) {
    const OWNER_COUNT = 8, ITEM_COUNT = 8, D = 16, ROOM = 8;
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    const randomGrid = size => Float64Array.from({ length: size }, () => gauss() * initScale);
    const P = {
      EO: randomGrid(OWNER_COUNT * D), EI: randomGrid(ITEM_COUNT * D),
      WQ: randomGrid(D * D), WK: randomGrid(D * D), WV: randomGrid(D * D),
      WO: randomGrid(D * ITEM_COUNT), b: new Float64Array(ITEM_COUNT),
    };
    const G = {};
    for (const name in P) G[name] = new Float64Array(P[name].length);

    const x = new Float64Array(ROOM * D), keys = new Float64Array(ROOM * D), values = new Float64Array(ROOM * D);
    const query = new Float64Array(D), blend = new Float64Array(D);
    const scores = new Float64Array(ROOM), attention = new Float64Array(ROOM), probs = new Float64Array(ITEM_COUNT);
    const dBlend = new Float64Array(D), dQuery = new Float64Array(D), dScores = new Float64Array(ROOM);

    function softmax(v, n) {
      let max = -Infinity, sum = 0;
      for (let i = 0; i < n; i++) if (v[i] > max) max = v[i];
      for (let i = 0; i < n; i++) { v[i] = Math.exp(v[i] - max); sum += v[i]; }
      for (let i = 0; i < n; i++) v[i] /= sum;
    }

    function forward(ex) {
      const n = ex.owners.length, asked = ex.asked * D;
      for (let i = 0; i < n; i++) {
        const o = ex.owners[i] * D, t = ex.items[i] * D;
        for (let d = 0; d < D; d++) x[i * D + d] = P.EO[o + d] + P.EI[t + d];
      }
      for (let h = 0; h < D; h++) {
        let sum = 0;
        for (let d = 0; d < D; d++) sum += P.EO[asked + d] * P.WQ[d * D + h];
        query[h] = sum;
      }
      for (let i = 0; i < n; i++) {
        let score = 0;
        for (let h = 0; h < D; h++) {
          let key = 0, value = 0;
          for (let d = 0; d < D; d++) {
            key += x[i * D + d] * P.WK[d * D + h];
            value += x[i * D + d] * P.WV[d * D + h];
          }
          keys[i * D + h] = key;
          values[i * D + h] = value;
          score += query[h] * key;
        }
        scores[i] = attention[i] = score;
      }
      softmax(attention, n);
      for (let e = 0; e < D; e++) {
        let sum = 0;
        for (let i = 0; i < n; i++) sum += attention[i] * values[i * D + e];
        blend[e] = sum;
      }
      for (let m = 0; m < ITEM_COUNT; m++) {
        let z = P.b[m];
        for (let e = 0; e < D; e++) z += blend[e] * P.WO[e * ITEM_COUNT + m];
        probs[m] = z;
      }
      softmax(probs, ITEM_COUNT);
      return { n, scores, attention, probs };
    }

    function addGradients(ex) {
      const { n } = forward(ex);
      const asked = ex.asked * D;

      dBlend.fill(0);
      for (let m = 0; m < ITEM_COUNT; m++) {
        const dz = probs[m] - (m === ex.answer ? 1 : 0);
        G.b[m] += dz;
        for (let e = 0; e < D; e++) {
          G.WO[e * ITEM_COUNT + m] += blend[e] * dz;
          dBlend[e] += dz * P.WO[e * ITEM_COUNT + m];
        }
      }

      let expected = 0;
      for (let i = 0; i < n; i++) {
        let dAttention = 0;
        for (let e = 0; e < D; e++) dAttention += dBlend[e] * values[i * D + e];
        dScores[i] = dAttention;
        expected += attention[i] * dAttention;
      }
      for (let i = 0; i < n; i++) dScores[i] = attention[i] * (dScores[i] - expected);

      for (let h = 0; h < D; h++) {
        let sum = 0;
        for (let i = 0; i < n; i++) sum += dScores[i] * keys[i * D + h];
        dQuery[h] = sum;
      }
      for (let d = 0; d < D; d++) {
        let dAsked = 0;
        for (let h = 0; h < D; h++) {
          G.WQ[d * D + h] += P.EO[asked + d] * dQuery[h];
          dAsked += dQuery[h] * P.WQ[d * D + h];
        }
        G.EO[asked + d] += dAsked;
      }

      for (let i = 0; i < n; i++) {
        const o = ex.owners[i] * D, t = ex.items[i] * D;
        for (let d = 0; d < D; d++) {
          const xd = x[i * D + d];
          let dx = 0;
          for (let h = 0; h < D; h++) {
            const dKey = dScores[i] * query[h], dValue = attention[i] * dBlend[h];
            G.WK[d * D + h] += xd * dKey;
            G.WV[d * D + h] += xd * dValue;
            dx += dKey * P.WK[d * D + h] + dValue * P.WV[d * D + h];
          }
          G.EO[o + d] += dx;
          G.EI[t + d] += dx;
        }
      }
    }

    function train(batch, lr) {
      for (const name in G) G[name].fill(0);
      for (const ex of batch) addGradients(ex);
      const step = lr / batch.length;
      for (const name in P) {
        const p = P[name], g = G[name];
        for (let j = 0; j < p.length; j++) p[j] -= step * g[j];
      }
    }

    function matchScore(asked, table, row) {
      let score = 0;
      for (let h = 0; h < D; h++) {
        let q = 0, k = 0;
        for (let d = 0; d < D; d++) {
          q += P.EO[asked * D + d] * P.WQ[d * D + h];
          k += P[table][row * D + d] * P.WK[d * D + h];
        }
        score += q * k;
      }
      return score;
    }

    const weightCount = Object.values(P).reduce((sum, grid) => sum + grid.length, 0);
    return { P, G, forward, addGradients, train, matchScore, weightCount };
  }

  let model, testSet, heldOut, history, step, example, running = false, activeRun = 0;
  let beams = [], labels = [], lockers = [];

  function shuffled(count) {
    const order = Array.from({ length: count }, (_, i) => i);
    for (let i = count - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    return order;
  }

  function randomExample(count) {
    const owners = shuffled(OWNERS.length).slice(0, count), items = shuffled(ITEMS.length).slice(0, count);
    const pick = Math.floor(Math.random() * count);
    return { owners, items, asked: owners[pick], answer: items[pick] };
  }

  const signature = ex => ex.owners.map((o, i) => o * ITEMS.length + ex.items[i]).sort((a, b) => a - b).join(',') + '?' + ex.asked;

  function trainingExample() {
    for (;;) {
      const ex = randomExample(TRAIN_LOCKERS);
      if (!heldOut.has(signature(ex))) return ex;
    }
  }

  function argmax(v, n) {
    let best = 0;
    for (let i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
  }

  function checkTestSet() {
    let right = 0, focus = 0;
    for (const ex of testSet) {
      const { attention, probs } = model.forward(ex);
      if (argmax(probs, ITEMS.length) === ex.answer) right++;
      focus += attention[ex.owners.indexOf(ex.asked)];
    }
    return { accuracy: right / testSet.length, focus: focus / testSet.length };
  }

  function buildRoom() {
    const options = (names, chosen) => names.map((name, id) => `<option value="${id}"${id === chosen ? ' selected' : ''}>${name}</option>`).join('');
    $('lk-room').innerHTML = example.owners.map((owner, i) =>
      `<div class="lk-locker"><div class="lk-beam"><div class="lk-beam-fill"></div></div><div class="lk-pct"></div>` +
      `<div class="lk-door"><select class="lk-owner" data-locker="${i}" aria-label="who owns locker ${i + 1}">${options(OWNERS, owner)}</select>` +
      `<select class="lk-item" data-locker="${i}" aria-label="what is in locker ${i + 1}">${options(ITEMS, example.items[i])}</select></div></div>`).join('');
    lockers = [...$('lk-room').querySelectorAll('.lk-locker')];
    beams = lockers.map(el => el.querySelector('.lk-beam-fill'));
    labels = lockers.map(el => el.querySelector('.lk-pct'));
    buildQuestion();
    $('lk-add').disabled = example.owners.length >= MAX_LOCKERS;
    $('lk-remove').disabled = example.owners.length <= 2;
  }

  function buildQuestion() {
    $('lk-asked').innerHTML = OWNERS.map((name, id) =>
      `<option value="${id}"${id === example.asked ? ' selected' : ''}>${name}${example.owners.includes(id) ? '' : ' (no locker here)'}</option>`).join('');
  }

  function renderExample() {
    const { attention, probs } = model.forward(example);
    const right = example.owners.indexOf(example.asked);
    lockers.forEach((el, i) => {
      el.classList.toggle('right', i === right);
      beams[i].style.height = (attention[i] * 100) + '%';
      labels[i].textContent = pct(attention[i]);
    });

    const ranked = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    const guess = ranked[0], sure = pct(probs[guess]);
    const verdict = $('lk-verdict');
    if (right < 0) {
      verdict.className = 'lk-verdict';
      verdict.innerHTML = `Nobody here is ${OWNERS[example.asked]}, so there is no right answer. The model says <strong>${ITEMS[guess]}</strong> anyway (${sure} sure).`;
    } else {
      const ok = guess === example.items[right];
      verdict.className = 'lk-verdict ' + (ok ? 'ok' : 'miss');
      verdict.innerHTML = `Right answer: <strong>${ITEMS[example.items[right]]}</strong>. The model says <strong>${ITEMS[guess]}</strong> (${sure} sure) ${ok ? '✓' : '✗'}`;
    }
    $('lk-guesses').innerHTML = ranked.slice(0, 3).map(m =>
      `<div class="bar-container"><span class="label">${ITEMS[m]}</span><div class="bar-track"><div class="bar" style="width:${probs[m] * 100}%;"></div></div>` +
      `<span class="pct">${probs[m].toFixed(2)}</span></div>`).join('');
  }

  function buildGrid(id, columns) {
    const head = `<span></span>` + columns.map(name => `<span class="lk-grid-head">${name}</span>`).join('');
    $(id).innerHTML = head + OWNERS.map((owner, a) =>
      `<span class="lk-grid-side">${owner}?</span>` + columns.map((name, c) =>
        `<span class="lk-cell" data-a="${a}" data-c="${c}"></span>`).join('')).join('');
  }

  function renderGrids() {
    for (const [id, table, names, what] of [['lk-grid-owner', 'EO', OWNERS, 'the name'], ['lk-grid-item', 'EI', ITEMS, 'the item']]) {
      $(id).querySelectorAll('.lk-cell').forEach(cell => {
        const a = +cell.dataset.a, c = +cell.dataset.c;
        const score = model.matchScore(a, table, c);
        const strength = Math.min(1, Math.abs(score) / DARKEST_SCORE) ** 3;
        cell.style.background = score >= 0 ? `rgba(234, 88, 12, ${strength})` : `rgba(71, 85, 105, ${strength * 0.6})`;
        cell.title = `question "${OWNERS[a]}?" against ${what} "${names[c]}": score ${score.toFixed(1)}`;
      });
    }
  }

  function renderProgress() {
    const now = history[history.length - 1];
    $('lk-seen').textContent = (step * BATCH).toLocaleString();
    $('lk-acc').textContent = pct(now.accuracy);
    $('lk-focus').textContent = pct(now.focus);
    Book.lineChart($('lk-chart'), {
      series: [
        { label: 'answers right', color: '#2563eb', data: history.map(h => h.accuracy) },
        { label: 'attention on the right locker', color: '#ea580c', data: history.map(h => h.focus) },
      ],
      xMax: STEPS / CHECK_EVERY + 1, yMin: 0, yMax: 1, yFormat: pct,
      xLabel: `training examples seen (0 to ${(STEPS * BATCH).toLocaleString()})`,
    });
    renderGrids();
  }

  function syncButtons() {
    trainBtn.textContent = running ? 'pause' : step >= STEPS ? 'train again from scratch' : step > 0 ? 'keep training' : 'train';
    trainBtn.classList.toggle('stop', running);
  }

  function stop() {
    activeRun++;
    running = false;
    syncButtons();
  }

  function startOver() {
    stop();
    model = createLockerModel(Math.random, 0.1);
    heldOut = new Set();
    testSet = [];
    while (testSet.length < TEST_SIZE) {
      const ex = randomExample(TRAIN_LOCKERS);
      if (heldOut.has(signature(ex))) continue;
      heldOut.add(signature(ex));
      testSet.push(ex);
    }
    step = 0;
    history = [checkTestSet()];
    $('lk-weights').textContent = model.weightCount.toLocaleString();
    renderProgress();
    renderExample();
    syncButtons();
  }

  async function train() {
    if (step >= STEPS) startOver();
    const run = ++activeRun;
    running = true;
    syncButtons();
    let last = performance.now(), owed = 0;
    while (step < STEPS) {
      await new Promise(resolve => setTimeout(resolve, 16));
      if (run !== activeRun) return;
      const now = performance.now();
      owed = Math.min(owed + (now - last) / 1000 * STEPS_PER_SECOND, STEPS_PER_SECOND);
      last = now;
      let checked = false;
      while (owed >= 1 && step < STEPS && performance.now() - now < TICK_BUDGET_MS) {
        owed--;
        model.train(Array.from({ length: BATCH }, trainingExample), LR);
        step++;
        if (step % CHECK_EVERY === 0) {
          history.push(checkTestSet());
          checked = true;
        }
      }
      if (checked) renderProgress();
      renderExample();
    }
    stop();
  }

  function setExample(next) {
    example = next;
    buildRoom();
    renderExample();
  }

  trainBtn.addEventListener('click', () => { if (running) stop(); else train(); });
  $('lk-reset').addEventListener('click', startOver);
  $('lk-new').addEventListener('click', () => setExample(randomExample(example.owners.length)));
  $('lk-add').addEventListener('click', () => {
    const free = shuffled(OWNERS.length).find(id => !example.owners.includes(id));
    setExample({
      owners: [...example.owners, free],
      items: [...example.items, Math.floor(Math.random() * ITEMS.length)],
      asked: example.asked,
    });
  });
  $('lk-remove').addEventListener('click', () => {
    const owners = example.owners.slice(0, -1);
    setExample({ owners, items: example.items.slice(0, -1), asked: owners.includes(example.asked) ? example.asked : owners[0] });
  });
  $('lk-shuffle').addEventListener('click', () => {
    let order;
    do order = shuffled(example.owners.length); while (order.every((from, to) => from === to));
    setExample({ owners: order.map(i => example.owners[i]), items: order.map(i => example.items[i]), asked: example.asked });
  });
  $('lk-room').addEventListener('change', e => {
    const select = e.target.closest('select');
    if (!select) return;
    const i = +select.dataset.locker, picked = +select.value;
    if (select.classList.contains('lk-item')) {
      example.items[i] = picked;
    } else {
      const clash = example.owners.indexOf(picked);
      if (clash >= 0 && clash !== i) {
        example.owners[clash] = example.owners[i];
        lockers[clash].querySelector('.lk-owner').value = example.owners[i];
      }
      example.owners[i] = picked;
      buildQuestion();
    }
    renderExample();
  });
  $('lk-asked').addEventListener('change', e => {
    example.asked = +e.target.value;
    renderExample();
  });

  buildGrid('lk-grid-owner', OWNERS);
  buildGrid('lk-grid-item', ITEMS);
  example = { owners: [0, 1, 2], items: [0, 1, 2], asked: 1 };
  buildRoom();
  startOver();
})();
