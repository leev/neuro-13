makeSingleTrainer({
  data: [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],1]],
  labels: ['piano','homework'],
  predIds: 't-predictions',
  statIds: { epoch: 't-epoch', w1: 't-w1', w2: 't-w2', b: 't-b' },
  btnIds: { step: 't-step', fast: 't-fast', reset: 't-reset' },
  modelId: 't-model',
});


(function nudgeWidget() {
  const data = [
    { x: [0,0], y: 0 },
    { x: [1,0], y: 1 },
    { x: [0,1], y: 1 },
    { x: [1,1], y: 1 },
  ];
  const START = { w1: -0.5, w2: 0.2, b: -0.1 };
  const STEP = 0.35;
  const dump = makeModelDump('nd-model', {
    rowComments: {
      w1: 'how much "piano done" matters',
      w2: 'how much "homework done" matters',
      bias: 'the built-in lean',
    },
  });
  let w1, w2, b, count, ex;

  function predict(x1, x2) { return sigmoid(w1*x1 + w2*x2 + b); }

  function renderMath() {
    const [x1, x2] = data[ex].x;
    const y = data[ex].y;
    const z = w1*x1 + w2*x2 + b;
    const o = predict(x1, x2);
    const err = y - o;
    let verdict;
    if (Math.abs(err) < 0.15) verdict = '<span class="ok-mark">already close — the nudges are getting tiny</span>';
    else if (err > 0) verdict = '<span class="bad-mark">too low</span> → push the total UP';
    else verdict = '<span class="bad-mark">too high</span> → push the total DOWN';
    document.getElementById('nd-math').innerHTML =
      `total      = (${w1.toFixed(2)} × ${x1}) + (${w2.toFixed(2)} × ${x2}) + (${b.toFixed(2)}) = <strong>${z.toFixed(2)}</strong><br>` +
      `prediction = sigmoid(${z.toFixed(2)}) = <strong>${o.toFixed(2)}</strong><br>` +
      `truth      = <strong>${y}</strong><br>` +
      `error      = ${y} − ${o.toFixed(2)} = <strong>${err >= 0 ? '+' : ''}${err.toFixed(2)}</strong>   (${verdict})`;
  }

  function renderNudge() {
    const [x1, x2] = data[ex].x;
    const y = data[ex].y;
    const err = y - predict(x1, x2);
    const rows = [
      { name: 'weight 1 (piano)   ', cur: w1, x: x1, who: 'piano' },
      { name: 'weight 2 (homework)', cur: w2, x: x2, who: 'homework' },
      { name: 'bias               ', cur: b,  x: 1,  isBias: true },
    ];
    let html = '';
    for (const r of rows) {
      const d = STEP * err * r.x;
      const next = r.cur + d;
      let why;
      if (!r.isBias && r.x === 0) {
        why = r.who + ' was 0, took no part → <strong>no nudge</strong>';
      } else if (Math.abs(d) < 0.01) {
        why = 'error is almost 0 → barely moves';
      } else if (d > 0) {
        why = (r.isBias ? 'always takes part' : r.who + ' was ON') + ', prediction too low → <span class="ok-mark">UP</span>';
      } else {
        why = (r.isBias ? 'always takes part' : r.who + ' was ON') + ', prediction too high → <span class="bad-mark">DOWN</span>';
      }
      const arrow = (!r.isBias && r.x === 0) ? '=' : '→';
      html += `${r.name}  ${r.cur.toFixed(2)} ${arrow} <strong>${next.toFixed(2)}</strong>   (${why})<br>`;
    }
    document.getElementById('nd-weights').innerHTML = html;
  }

  function renderPreds() {
    let html = '<strong>all four examples right now:</strong><br>';
    for (const { x: [x1, x2], y } of data) {
      const o = predict(x1, x2);
      const ok = (o > 0.5) === (y > 0.5);
      const mark = ok ? '<span class="ok-mark">✓</span>' : '<span class="bad-mark">✗</span>';
      html += `piano=${x1}, homework=${x2}  →  predicted <strong>${o.toFixed(3)}</strong>  (truth ${y})  ${mark}<br>`;
    }
    document.getElementById('nd-preds').innerHTML = html;
  }

  function render() {
    renderMath();
    renderNudge();
    renderPreds();
    dump.update({ w1: w1, w2: w2, bias: b });
    document.getElementById('nd-count').textContent = count;
    document.getElementById('nd-w1').textContent = w1.toFixed(2);
    document.getElementById('nd-w2').textContent = w2.toFixed(2);
    document.getElementById('nd-b').textContent  = b.toFixed(2);
  }

  function nudgeOnce(i) {
    const [x1, x2] = data[i].x;
    const err = data[i].y - predict(x1, x2);
    w1 += STEP * err * x1;
    w2 += STEP * err * x2;
    b  += STEP * err;
    count++;
  }

  function reset() {
    w1 = START.w1; w2 = START.w2; b = START.b;
    count = 0;
    dump.reset();
    render();
  }

  document.querySelectorAll('#nd-picker .toggle').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('#nd-picker .toggle').forEach(o => o.classList.remove('on'));
      t.classList.add('on');
      ex = Number(t.dataset.ex);
      render();
    });
  });

  document.getElementById('nd-nudge').addEventListener('click', () => { nudgeOnce(ex); render(); });
  document.getElementById('nd-cycle').addEventListener('click', () => { for (let i = 0; i < data.length; i++) nudgeOnce(i); render(); });
  document.getElementById('nd-reset').addEventListener('click', reset);

  ex = 1;
  reset();
})();


makeSeparator('sep-or',
  [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],1]],
  ['piano','homework'],
  { start: [-1, 1, 0.5], nearMiss: 'so close: one dot is still on the wrong side' });
