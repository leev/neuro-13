(function decisionModelWidget() {
  const trainBtn = document.getElementById('dm-train');
  if (!trainBtn) return;
  const $ = id => document.getElementById(id);
  const TOPICS = ['world', 'sports', 'business', 'scitech'];
  const COLORS = { world: '#2563eb', sports: '#16a34a', business: '#ea580c', scitech: '#9333ea' };
  const K = TOPICS.length, D = 16, N_TRAIN = 8000, EPOCHS = 15, BATCH = 64, LR = 1.0;
  const BANDS = [[0.25, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1.01]];

  let docs = null, labels = null, vocab = null, idToWord = [], V = 0;
  let E, W, b, history = [], testConf = null, testRight = null, ready = false, question = 'choice';

  const words = text => text.toLowerCase().match(/[a-z]+/g) || [];
  const pct = x => Math.round(x * 100) + '%';

  function gauss() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  async function loadData() {
    const res = await fetch('py/headlines.csv');
    if (!res.ok) throw new Error('could not download headlines.csv');
    const rows = [];
    for (const line of (await res.text()).split(/\r?\n/).slice(1)) {
      const m = line.match(/^"([a-z]+)","(.*)"$/);
      if (m) rows.push([TOPICS.indexOf(m[1]), words(m[2])]);
    }
    if (rows.length <= N_TRAIN) throw new Error('headlines.csv is incomplete');
    const counts = new Map();
    for (const [, ws] of rows.slice(0, N_TRAIN)) for (const w of ws) counts.set(w, (counts.get(w) || 0) + 1);
    const known = [...counts].filter(([, c]) => c >= 3).map(([w]) => w).sort();
    vocab = new Map(known.map((w, i) => [w, i + 1]));
    V = known.length + 1;
    idToWord = ['', ...known];
    labels = rows.map(r => r[0]);
    docs = rows.map(r => r[1].filter(w => vocab.has(w)).map(w => vocab.get(w)));
  }

  function initModel() {
    E = new Float64Array(V * D);
    for (let i = D; i < E.length; i++) E[i] = gauss() * 0.1;
    W = new Float64Array(D * K).map(() => gauss() * 0.1);
    b = new Float64Array(K);
    history = [];
  }

  function embed(ids, h) {
    h.fill(0);
    for (const id of ids) for (let d = 0; d < D; d++) h[d] += E[id * D + d];
    const n = Math.max(ids.length, 1);
    for (let d = 0; d < D; d++) h[d] /= n;
  }

  function predict(ids, h, p) {
    embed(ids, h);
    let max = -Infinity, sum = 0;
    for (let k = 0; k < K; k++) {
      let z = b[k];
      for (let d = 0; d < D; d++) z += h[d] * W[d * K + k];
      p[k] = z;
      if (z > max) max = z;
    }
    for (let k = 0; k < K; k++) { p[k] = Math.exp(p[k] - max); sum += p[k]; }
    for (let k = 0; k < K; k++) p[k] /= sum;
  }

  function argmax(p) {
    let best = 0;
    for (let k = 1; k < K; k++) if (p[k] > p[best]) best = k;
    return best;
  }

  function accuracy(lo, hi) {
    const h = new Float64Array(D), p = new Float64Array(K);
    let right = 0;
    for (let i = lo; i < hi; i++) {
      predict(docs[i], h, p);
      if (argmax(p) === labels[i]) right++;
    }
    return right / (hi - lo);
  }

  function trainEpoch() {
    const order = Array.from({ length: N_TRAIN }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const h = new Float64Array(D), p = new Float64Array(K);
    const gW = new Float64Array(D * K), gb = new Float64Array(K);
    for (let start = 0; start < N_TRAIN; start += BATCH) {
      const batch = order.slice(start, start + BATCH);
      gW.fill(0); gb.fill(0);
      const wordGrads = [];
      for (const i of batch) {
        predict(docs[i], h, p);
        p[labels[i]] -= 1;
        const dh = new Float64Array(D);
        for (let k = 0; k < K; k++) {
          const g = p[k] / batch.length;
          gb[k] += g;
          for (let d = 0; d < D; d++) {
            gW[d * K + k] += h[d] * g;
            dh[d] += g * W[d * K + k];
          }
        }
        wordGrads.push(dh);
      }
      for (let j = 0; j < W.length; j++) W[j] -= LR * gW[j];
      for (let k = 0; k < K; k++) b[k] -= LR * gb[k];
      batch.forEach((i, n) => {
        const ids = docs[i], dh = wordGrads[n];
        for (const id of ids) for (let d = 0; d < D; d++) E[id * D + d] -= LR * dh[d] / ids.length;
      });
    }
  }

  function drawCurve() {
    Book.lineChart($('dm-curve'), {
      series: [
        { label: 'training headlines', color: '#2563eb', data: history.map(pt => pt.train) },
        { label: 'unseen test headlines', color: '#ea580c', data: history.map(pt => pt.test) },
      ],
      xMax: EPOCHS, yMin: 0, yMax: 1, yFormat: pct, xLabel: 'epochs of training',
    });
  }

  function scoreTestSet() {
    const h = new Float64Array(D), p = new Float64Array(K);
    testConf = []; testRight = [];
    for (let i = N_TRAIN; i < docs.length; i++) {
      predict(docs[i], h, p);
      const best = argmax(p);
      testConf.push(p[best]);
      testRight.push(best === labels[i]);
    }
  }

  function renderCalibration() {
    $('dm-calib').innerHTML = BANDS.map(([lo, hi]) => {
      let n = 0, right = 0;
      testConf.forEach((c, i) => { if (c >= lo && c < hi) { n++; if (testRight[i]) right++; } });
      const acc = n ? right / n : 0;
      return `<div class="bar-container"><span class="label" style="width: 150px;">says ${pct(lo)}–${pct(Math.min(hi, 1))} sure</span>` +
        `<div class="bar-track"><div class="bar" style="width:${acc * 100}%;"></div></div>` +
        `<span class="pct" style="width: 120px;">right ${n ? pct(acc) : '—'} of ${n}</span></div>`;
    }).join('');
  }

  function renderThreshold() {
    const t = +$('dm-thresh').value / 100;
    $('dm-thresh-v').textContent = pct(t);
    if (!ready) return;
    let n = 0, right = 0;
    testConf.forEach((c, i) => { if (c >= t) { n++; if (testRight[i]) right++; } });
    $('dm-auto').textContent = `${n} of ${testConf.length}`;
    $('dm-auto-acc').textContent = n ? pct(right / n) : '—';
    $('dm-human').textContent = testConf.length - n;
  }

  function renderAnswer() {
    if (!ready) return;
    const ws = words($('dm-input').value);
    const ids = ws.filter(w => vocab.has(w)).map(w => vocab.get(w));
    $('dm-pills').innerHTML = ws.map(w => vocab.has(w)
      ? `<div class="token-pill">${w}<span class="token-id">id ${vocab.get(w)}</span></div>`
      : `<div class="token-pill" style="background: var(--code-bg); border-color: var(--border); color: var(--muted); text-decoration: line-through;">${w}<span class="token-id" style="text-decoration: none;">not in dictionary</span></div>`
    ).join('');

    const h = new Float64Array(D), p = new Float64Array(K);
    if (ids.length) predict(ids, h, p); else p.fill(1 / K);
    const best = argmax(p);
    $('dm-json').textContent = question === 'choice'
      ? `{"choice": "${TOPICS[best]}", "confidence": ${p[best].toFixed(2)}}`
      : `{"${question}": ${p[TOPICS.indexOf(question)].toFixed(2)}}`;
    if (!ids.length) $('dm-json').textContent += '\n\n(no words it knows, so we give every topic an equal 25%)';
    $('dm-bars').innerHTML = TOPICS.map((t, k) =>
      `<div class="bar-container"><span class="label">${t}</span><div class="bar-track">` +
      `<div class="bar" style="width:${p[k] * 100}%; background:${COLORS[t]};"></div></div>` +
      `<span class="pct">${p[k].toFixed(2)}</span></div>`).join('');
  }

  function renderNeighbors() {
    if (!ready) return;
    const word = (words($('dm-word').value)[0] || '');
    const box = $('dm-neighbors');
    if (!vocab.has(word)) {
      box.innerHTML = `<span style="font-size: 13px; color: var(--muted);">${word ? `"${word}" is not in the dictionary` : 'type a word'}</span>`;
      return;
    }
    const norm = id => {
      let s = 0;
      for (let d = 0; d < D; d++) s += E[id * D + d] * E[id * D + d];
      return Math.sqrt(s) || 1;
    };
    const me = vocab.get(word), myNorm = norm(me), scored = [];
    for (let id = 1; id < V; id++) {
      if (id === me) continue;
      let dot = 0;
      for (let d = 0; d < D; d++) dot += E[me * D + d] * E[id * D + d];
      scored.push([dot / (myNorm * norm(id)), id]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    box.innerHTML = scored.slice(0, 10).map(([sim, id]) =>
      `<div class="token-pill">${idToWord[id]}<span class="token-id">${sim.toFixed(2)} alike</span></div>`).join('');
  }

  async function train() {
    trainBtn.disabled = true;
    ready = false;
    $('dm-input').disabled = true;
    $('dm-thresh').disabled = true;
    $('dm-word').disabled = true;
    try {
      if (!docs) {
        trainBtn.textContent = '… downloading headlines';
        await loadData();
      }
    } catch (err) {
      trainBtn.disabled = false;
      trainBtn.textContent = 'load 10,000 headlines & train';
      $('dm-json').textContent = 'Could not load headlines.csv. If you opened this page as a file, run\n' +
        '"make serve" in the book folder and open http://localhost:8000';
      return;
    }
    initModel();
    $('dm-vocab').textContent = (V - 1).toLocaleString();
    $('dm-params').textContent = (E.length + W.length + b.length).toLocaleString();
    trainBtn.textContent = '… training';
    try {
      for (let epoch = 1; epoch <= EPOCHS; epoch++) {
        await new Promise(resolve => setTimeout(resolve, 0));
        trainEpoch();
        const point = { train: accuracy(0, N_TRAIN), test: accuracy(N_TRAIN, docs.length) };
        history.push(point);
        $('dm-epoch').textContent = epoch;
        $('dm-train-acc').textContent = pct(point.train);
        $('dm-test-acc').textContent = pct(point.test);
        drawCurve();
      }
      scoreTestSet();
    } catch (err) {
      trainBtn.disabled = false;
      trainBtn.textContent = 'train again from scratch';
      $('dm-json').textContent = 'Training stopped: ' + err.message;
      return;
    }
    ready = true;
    $('dm-input').disabled = false;
    $('dm-thresh').disabled = false;
    $('dm-word').disabled = false;
    trainBtn.disabled = false;
    trainBtn.textContent = 'train again from scratch';
    renderCalibration();
    renderThreshold();
    renderAnswer();
    renderNeighbors();
  }

  trainBtn.addEventListener('click', train);
  $('dm-word').addEventListener('input', renderNeighbors);
  $('dm-input').addEventListener('input', renderAnswer);
  $('dm-thresh').addEventListener('input', renderThreshold);
  $('dm-examples').addEventListener('click', e => {
    const btn = e.target.closest('[data-text]');
    if (!btn) return;
    $('dm-input').value = btn.getAttribute('data-text');
    renderAnswer();
  });
  $('dm-question').addEventListener('click', e => {
    const tog = e.target.closest('[data-q]');
    if (!tog) return;
    question = tog.getAttribute('data-q');
    $('dm-question').querySelectorAll('.toggle').forEach(t => t.classList.toggle('on', t === tog));
    renderAnswer();
  });
  drawCurve();
})();

