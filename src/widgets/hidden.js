// ===== Interactive 4: XOR with hidden layer =====
(function() {
  const data = [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],0]];
  const H = 4;
  const lr = 0.5;
  const dump = makeModelDump('h-model', {
    decimals: 2,
    headComments: {
      w1: 'weights: inputs → hidden (rows = left, right; columns = 4 hidden neurons)',
      b1: 'biases: one per hidden neuron',
      w2: 'weights: hidden → output (one per hidden neuron)',
      b2: 'bias: the output neuron',
    },
    rowComments: { w1: ['left button', 'right button'] },
  });
  let W1, b1, W2, b2, epoch;

  function reset() {
    W1 = [[],[]];
    for (let i=0;i<2;i++) for (let j=0;j<H;j++) W1[i].push(Math.random()*2-1);
    b1 = new Array(H).fill(0);
    W2 = [];
    for (let j=0;j<H;j++) W2.push(Math.random()*2-1);
    b2 = 0;
    epoch = 0;
    dump.reset();
    render();
  }

  function forward(x1, x2) {
    const h = [];
    for (let j=0;j<H;j++) h.push(sigmoid(x1*W1[0][j] + x2*W1[1][j] + b1[j]));
    let zo = b2;
    for (let j=0;j<H;j++) zo += h[j]*W2[j];
    return { h, o: sigmoid(zo) };
  }

  function step(n=1) {
    for (let i=0;i<n;i++) {
      const dW1 = [[0,0,0,0],[0,0,0,0]];
      const db1 = [0,0,0,0];
      const dW2 = [0,0,0,0];
      let db2 = 0;
      for (const [[x1,x2], y] of data) {
        const {h, o} = forward(x1,x2);
        const err = y - o;
        const d_o = err * o * (1-o);
        for (let j=0;j<H;j++) dW2[j] += d_o * h[j];
        db2 += d_o;
        for (let j=0;j<H;j++) {
          const d_h = d_o * W2[j] * h[j] * (1 - h[j]);
          dW1[0][j] += d_h * x1;
          dW1[1][j] += d_h * x2;
          db1[j]   += d_h;
        }
      }
      for (let j=0;j<H;j++) {
        W2[j] += lr*dW2[j];
        b1[j] += lr*db1[j];
        W1[0][j] += lr*dW1[0][j];
        W1[1][j] += lr*dW1[1][j];
      }
      b2 += lr*db2;
      epoch++;
    }
    render();
  }

  function render() {
    let html = '';
    for (const [[x1,x2], y] of data) {
      const {o} = forward(x1,x2);
      const ok = (o>0.5)===(y>0.5);
      const mark = ok
        ? '<span class="ok-mark">✓</span>'
        : '<span class="bad-mark">✗</span>';
      html += `left=${x1}, right=${x2}  →  predicted <strong>${o.toFixed(3)}</strong>  (truth ${y})  ${mark}<br>`;
    }
    document.getElementById('h-predictions').innerHTML = html;
    document.getElementById('h-epoch').textContent = epoch;
    dump.update({ w1: W1, b1: b1, w2: W2, b2: b2 });
  }

  document.getElementById('h-step').addEventListener('click', () => step(1));
  document.getElementById('h-fast').addEventListener('click', () => step(1000));
  document.getElementById('h-reset').addEventListener('click', reset);
  reset();
})();


// ===== Interactive: animated XOR network =====
(function() {
  const svg = document.getElementById('net-svg');
  if (!svg) return;
  const SW = 600, SH = 340;
  const data = [[[0,0],0],[[1,0],1],[[0,1],1],[[1,1],0]];
  const H = 4;
  const lr = 0.5;
  let W1, b1, W2, b2, epoch;
  let currentInput = [0, 0];

  function sig(z) { return 1 / (1 + Math.exp(-z)); }

  function reset() {
    W1 = [[],[]];
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < H; j++)
        W1[i].push(Math.random() * 2 - 1);
    b1 = new Array(H).fill(0);
    W2 = [];
    for (let j = 0; j < H; j++) W2.push(Math.random() * 2 - 1);
    b2 = 0;
    epoch = 0;
    redraw();
  }

  function forward(x1, x2) {
    const h = [];
    for (let j = 0; j < H; j++)
      h.push(sig(x1*W1[0][j] + x2*W1[1][j] + b1[j]));
    let zo = b2;
    for (let j = 0; j < H; j++) zo += h[j] * W2[j];
    return { h, o: sig(zo) };
  }

  function trainStep() {
    const dW1 = [[0,0,0,0],[0,0,0,0]];
    const db1 = [0,0,0,0];
    const dW2 = [0,0,0,0];
    let db2 = 0;
    for (const [[x1,x2], y] of data) {
      const {h, o} = forward(x1, x2);
      const err = y - o;
      const d_o = err * o * (1 - o);
      for (let j = 0; j < H; j++) dW2[j] += d_o * h[j];
      db2 += d_o;
      for (let j = 0; j < H; j++) {
        const d_h = d_o * W2[j] * h[j] * (1 - h[j]);
        dW1[0][j] += d_h * x1;
        dW1[1][j] += d_h * x2;
        db1[j] += d_h;
      }
    }
    for (let j = 0; j < H; j++) {
      W2[j] += lr * dW2[j];
      b1[j] += lr * db1[j];
      W1[0][j] += lr * dW1[0][j];
      W1[1][j] += lr * dW1[1][j];
    }
    b2 += lr * db2;
    epoch++;
  }

  const inputX = 90;
  const hiddenX = 290;
  const outputX = 500;
  const inputY = [130, 220];
  const hiddenY = [60, 140, 220, 300];
  const outputY = 175;

  function edgeColor(w) { return w >= 0 ? '#2563eb' : '#dc2626'; }
  function edgeWidth(w) { return Math.min(7, Math.abs(w) * 1.6 + 0.4); }
  function nodeFill(act) {
    const a = Math.min(1, Math.abs(act));
    return `rgba(37, 99, 235, ${0.15 + a * 0.85})`;
  }

  function weightLabel(x1, y1, x2, y2, w, offset = 0) {
    const mx = x1 + (x2 - x1) * 0.52;
    const my = y1 + (y2 - y1) * 0.52 + offset;
    const text = w.toFixed(2);
    const width = 28;
    const color = w >= 0 ? '#1d4ed8' : '#b91c1c';
    return `
      <rect x="${mx - width / 2}" y="${my - 9}" width="${width}" height="16" rx="4" fill="white" opacity="0.88"/>
      <text x="${mx}" y="${my + 3}" text-anchor="middle" font-size="10" font-family="ui-monospace, Menlo, monospace" fill="${color}" font-weight="700">${text}</text>
    `;
  }

  function renderMath(h, o) {
    const rows = [];
    let z = b2;
    for (let j = 0; j < H; j++) {
      const contribution = h[j] * W2[j];
      z += contribution;
      const voteClass = contribution >= 0 ? 'yes-vote' : 'no-vote';
      const voteText = contribution >= 0 ? 'YES' : 'NO';
      rows.push(
        `<div class="math-row">
          <span>h${j+1}</span>
          <span>${h[j].toFixed(2)} × ${W2[j].toFixed(2)}</span>
          <span class="vote ${voteClass}">${contribution.toFixed(2)} ${voteText}</span>
        </div>`
      );
    }
    rows.push(
      `<div class="math-row">
        <span>bias</span>
        <span>output bias</span>
        <span class="vote ${b2 >= 0 ? 'yes-vote' : 'no-vote'}">${b2.toFixed(2)}</span>
      </div>`
    );
    rows.push(
      `<div class="math-total">
        weighted sum = ${z.toFixed(2)}<br>
        sigmoid(${z.toFixed(2)}) = ${o.toFixed(3)} ${o > 0.5 ? '→ YES' : '→ NO'}
      </div>`
    );
    document.getElementById('net-math').innerHTML = rows.join('');
  }

  function redraw(highlights = {}) {
    const [x1, x2] = currentInput;
    const {h, o} = forward(x1, x2);

    let html = '<defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>';

    // edges input -> hidden
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < H; j++) {
        const w = W1[i][j];
        const hl = highlights.fwd1;
        const opacity = hl ? 1 : 0.5;
        const filt = hl ? 'filter="url(#glow)"' : '';
        const xStart = inputX + 24;
        const yStart = inputY[i];
        const xEnd = hiddenX - 24;
        const yEnd = hiddenY[j];
        html += `<line x1="${xStart}" y1="${yStart}" x2="${xEnd}" y2="${yEnd}" stroke="${edgeColor(w)}" stroke-width="${edgeWidth(w)}" opacity="${opacity}" ${filt}/>`;
        html += weightLabel(xStart, yStart, xEnd, yEnd, w, i === 0 ? -7 : 7);
      }
    }
    // edges hidden -> output
    for (let j = 0; j < H; j++) {
      const w = W2[j];
      const hl = highlights.fwd2;
      const opacity = hl ? 1 : 0.5;
      const filt = hl ? 'filter="url(#glow)"' : '';
      const xStart = hiddenX + 24;
      const yStart = hiddenY[j];
      const xEnd = outputX - 26;
      const yEnd = outputY;
      html += `<line x1="${xStart}" y1="${yStart}" x2="${xEnd}" y2="${yEnd}" stroke="${edgeColor(w)}" stroke-width="${edgeWidth(w)}" opacity="${opacity}" ${filt}/>`;
      html += weightLabel(xStart, yStart, xEnd, yEnd, w, j < 2 ? -7 : 7);
    }

    // input nodes
    for (let i = 0; i < 2; i++) {
      const v = currentInput[i];
      const filt = highlights.fwd1 ? 'filter="url(#glow)"' : '';
      html += `<circle cx="${inputX}" cy="${inputY[i]}" r="22" fill="${nodeFill(v)}" stroke="#1f2937" stroke-width="2" ${filt}/>`;
      html += `<text x="${inputX}" y="${inputY[i]+5}" text-anchor="middle" font-size="14" fill="${v > 0.5 ? 'white' : '#1f2937'}" font-weight="700">${v}</text>`;
      html += `<text x="${inputX-32}" y="${inputY[i]+4}" font-size="11" fill="#6b7280" text-anchor="end">${i===0?'left':'right'}</text>`;
    }

    // hidden nodes
    for (let j = 0; j < H; j++) {
      const v = h[j];
      const filt = highlights.fwd2 ? 'filter="url(#glow)"' : '';
      html += `<circle cx="${hiddenX}" cy="${hiddenY[j]}" r="22" fill="${nodeFill(v)}" stroke="#1f2937" stroke-width="2" ${filt}/>`;
      html += `<text x="${hiddenX}" y="${hiddenY[j]+5}" text-anchor="middle" font-size="11" fill="${v > 0.5 ? 'white' : '#1f2937'}" font-weight="700">${v.toFixed(2)}</text>`;
    }

    // output node
    const oFilt = highlights.outOk || highlights.outErr ? 'filter="url(#glow)"' : '';
    const oFill = highlights.outErr ? '#fee2e2' : highlights.outOk ? '#d1fae5' : nodeFill(o);
    html += `<circle cx="${outputX}" cy="${outputY}" r="28" fill="${oFill}" stroke="#1f2937" stroke-width="2" ${oFilt}/>`;
    html += `<text x="${outputX}" y="${outputY+5}" text-anchor="middle" font-size="14" fill="${o > 0.5 ? 'white' : '#1f2937'}" font-weight="700">${o.toFixed(2)}</text>`;

    // labels
    html += `<text x="${inputX}" y="320" text-anchor="middle" font-size="11" fill="#6b7280">inputs</text>`;
    html += `<text x="${hiddenX}" y="320" text-anchor="middle" font-size="11" fill="#6b7280">hidden layer (4 neurons)</text>`;
    html += `<text x="${outputX}" y="320" text-anchor="middle" font-size="11" fill="#6b7280">output</text>`;

    svg.innerHTML = html;

    const truth = data.find(d => d[0][0]===currentInput[0] && d[0][1]===currentInput[1])[1];
    document.getElementById('net-pred').textContent = o.toFixed(3);
    document.getElementById('net-truth').textContent = truth;
    document.getElementById('net-epoch').textContent = epoch;
    renderMath(h, o);
  }

  function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  let busy = false;
  async function animateStep() {
    if (busy) return;
    busy = true;
    redraw({ fwd1: true });
    await delay(280);
    redraw({ fwd2: true });
    await delay(280);
    const {o} = forward(currentInput[0], currentInput[1]);
    const truth = data.find(d => d[0][0]===currentInput[0] && d[0][1]===currentInput[1])[1];
    const correct = (o > 0.5) === (truth > 0.5);
    redraw(correct ? { outOk: true } : { outErr: true });
    await delay(280);
    trainStep();
    redraw();
    busy = false;
  }

  document.getElementById('net-step').addEventListener('click', () => animateStep());
  document.getElementById('net-fast').addEventListener('click', () => {
    if (busy) return;
    for (let i = 0; i < 200; i++) trainStep();
    redraw();
  });
  document.getElementById('net-reset').addEventListener('click', () => { if (!busy) reset(); });

  document.querySelectorAll('#net-input-picker [data-input]').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('#net-input-picker [data-input]').forEach(x => x.classList.remove('on'));
      el.classList.add('on');
      const [a, b] = el.getAttribute('data-input').split(',').map(Number);
      currentInput = [a, b];
      redraw();
    });
  });

  reset();
})();

