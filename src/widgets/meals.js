(function mealWidget() {
  const svg = document.getElementById('meal-net-svg');
  if (!svg) return;

  const meals = ['apple pie','sausage roll','cottage pie','baked potato','spaghetti','mac & cheese','ice cream','fruit salad'];
  const featureNames = [
    'main','snack','dessert',
    'hot','cold',
    'sweet','savory',
    'veg','meat',
    'pastry','potato','pasta','dairy','fruit',
  ];
  const rows = [
    // main snack dessert  hot cold  sweet savory  veg meat  pastry potato pasta dairy fruit
    {x:[0,0,1, 1,0, 1,0, 1,0, 1,0,0,0,0], y:0},   // apple pie
    {x:[0,1,0, 1,0, 0,1, 0,1, 1,0,0,0,0], y:1},   // sausage roll
    {x:[1,0,0, 1,0, 0,1, 0,1, 0,1,0,0,0], y:2},   // cottage pie
    {x:[1,0,0, 1,0, 0,1, 1,0, 0,1,0,0,0], y:3},   // baked potato
    {x:[1,0,0, 1,0, 0,1, 0,1, 0,0,1,0,0], y:4},   // spaghetti (bolognese)
    {x:[1,0,0, 1,0, 0,1, 1,0, 0,0,1,0,0], y:5},   // mac & cheese
    {x:[0,0,1, 0,1, 1,0, 1,0, 0,0,0,1,0], y:6},   // ice cream
    {x:[0,0,1, 0,1, 1,0, 1,0, 0,0,0,0,1], y:7},   // fruit salad
  ];
  const D = featureNames.length, C = meals.length;
  const lr = 0.04;
  let H = 8, W1, b1, W2, b2, epoch, dump;
  const state = { course: '', temp: '', taste: '', diet: '', base: '' };

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function relu(x) { return Math.max(0, x); }
  function softmax(z) {
    const m = Math.max(...z);
    const e = z.map(v => Math.exp(v - m));
    const s = e.reduce((a,b) => a + b, 0);
    return e.map(v => v / s);
  }

  function reset() {
    H = Number(document.getElementById('a-hidden-count').value);
    W1 = Array.from({length: D}, () => Array.from({length: H}, () => randn() * Math.sqrt(2 / D)));
    b1 = Array(H).fill(0);
    W2 = Array.from({length: H}, () => Array.from({length: C}, () => randn() * Math.sqrt(2 / H)));
    b2 = Array(C).fill(0);
    epoch = 0;
    dump = makeModelDump('a-weight-dump', {
      decimals: 2,
      rowComments: {
        w1: featureNames,
        w2: Array.from({length: H}, (_, j) => 'hidden ' + (j + 1)),
      },
      headComments: {
        w1: 'weights: clues → hidden (rows = clues, columns = the ' + H + ' hidden neurons)',
        b1: 'biases: one per hidden neuron',
        w2: 'weights: hidden → meals (columns = ' + meals.join(', ') + ')',
        b2: 'biases: one per meal (same order)',
      },
    });
    render();
  }

  function vectorFromState() {
    return [
      state.course === 'main' ? 1 : 0,
      state.course === 'snack' ? 1 : 0,
      state.course === 'dessert' ? 1 : 0,
      state.temp === 'hot' ? 1 : 0,
      state.temp === 'cold' ? 1 : 0,
      state.taste === 'sweet' ? 1 : 0,
      state.taste === 'savory' ? 1 : 0,
      state.diet === 'vegetarian' ? 1 : 0,
      state.diet === 'meat' ? 1 : 0,
      state.base === 'pastry' ? 1 : 0,
      state.base === 'potato' ? 1 : 0,
      state.base === 'pasta' ? 1 : 0,
      state.base === 'dairy' ? 1 : 0,
      state.base === 'fruit' ? 1 : 0,
    ];
  }

  function forward(x) {
    const z1 = [];
    const h = [];
    for (let j = 0; j < H; j++) {
      let z = b1[j];
      for (let i = 0; i < D; i++) z += x[i] * W1[i][j];
      z1.push(z);
      h.push(relu(z));
    }
    const z2 = [];
    for (let k = 0; k < C; k++) {
      let z = b2[k];
      for (let j = 0; j < H; j++) z += h[j] * W2[j][k];
      z2.push(z);
    }
    return { z1, h, p: softmax(z2) };
  }

  function train(n=1) {
    for (let ep = 0; ep < n; ep++) {
      const dW1 = W1.map(row => row.map(() => 0));
      const db1 = b1.map(() => 0);
      const dW2 = W2.map(row => row.map(() => 0));
      const db2 = b2.map(() => 0);
      for (const row of rows) {
        const {z1, h, p} = forward(row.x);
        const dOut = p.slice();
        dOut[row.y] -= 1;
        for (let k = 0; k < C; k++) dOut[k] /= rows.length;
        for (let j = 0; j < H; j++) {
          for (let k = 0; k < C; k++) dW2[j][k] += h[j] * dOut[k];
        }
        for (let k = 0; k < C; k++) db2[k] += dOut[k];
        for (let j = 0; j < H; j++) {
          let sum = 0;
          for (let k = 0; k < C; k++) sum += dOut[k] * W2[j][k];
          const dh = z1[j] > 0 ? sum : 0;
          for (let i = 0; i < D; i++) dW1[i][j] += row.x[i] * dh;
          db1[j] += dh;
        }
      }
      for (let i = 0; i < D; i++) for (let j = 0; j < H; j++) W1[i][j] -= lr * dW1[i][j];
      for (let j = 0; j < H; j++) {
        b1[j] -= lr * db1[j];
        for (let k = 0; k < C; k++) W2[j][k] -= lr * dW2[j][k];
      }
      for (let k = 0; k < C; k++) b2[k] -= lr * db2[k];
      epoch++;
    }
    render();
  }

  function accuracy() {
    let ok = 0;
    for (const row of rows) {
      const p = forward(row.x).p;
      if (p.indexOf(Math.max(...p)) === row.y) ok++;
    }
    return ok / rows.length;
  }

  function edgeColor(w) { return w >= 0 ? '#2563eb' : '#dc2626'; }
  function edgeWidth(w) { return Math.min(5, Math.abs(w) * 1.5 + 0.3); }
  function label(x1,y1,x2,y2,w) {
    const x = x1 + (x2-x1) * 0.55, y = y1 + (y2-y1) * 0.55;
    return `<rect x="${x-14}" y="${y-8}" width="28" height="15" rx="4" fill="white" opacity="0.86"/><text x="${x}" y="${y+3}" text-anchor="middle" font-size="9" font-family="ui-monospace, Menlo, monospace" fill="${edgeColor(w)}" font-weight="700">${w.toFixed(1)}</text>`;
  }

  function spread(count, top, bottom) {
    if (count <= 1) return [(top + bottom) / 2];
    return Array.from({length: count}, (_, i) => top + i * ((bottom - top) / (count - 1)));
  }

  function renderNetwork(x, h, p) {
    const active = x.map((v,i) => v ? i : -1).filter(i => i >= 0);
    const top = p.map((v,i) => [i,v]).sort((a,b) => b[1]-a[1]).slice(0,4);
    const inputX = 70, hiddenX = 300, outX = 535;
    const inputY = spread(active.length, 58, 262);
    const hiddenY = spread(H, 36, 304);
    const outY = spread(top.length, 78, 258);
    const hiddenR = H > 8 ? 15 : H > 6 ? 18 : 21;
    const hiddenFont = H > 8 ? 9 : 10;
    let html = '';
    if (!active.length) {
      svg.innerHTML = '<text x="310" y="150" text-anchor="middle" font-size="14" fill="#6b7280">Choose one or more traits to send clues into the network.</text>';
      return;
    }
    for (let ai = 0; ai < active.length; ai++) {
      const i = active[ai];
      for (let j = 0; j < H; j++) {
        html += `<line x1="${inputX+38}" y1="${inputY[ai]}" x2="${hiddenX-24}" y2="${hiddenY[j]}" stroke="${edgeColor(W1[i][j])}" stroke-width="${edgeWidth(W1[i][j])}" opacity="0.42"/>`;
        if (ai < 3 && (j === ai || j === ai + 1)) html += label(inputX+38, inputY[ai], hiddenX-24, hiddenY[j], W1[i][j]);
      }
    }
    for (let j = 0; j < H; j++) {
      for (let oi = 0; oi < top.length; oi++) {
        const k = top[oi][0];
        html += `<line x1="${hiddenX+24}" y1="${hiddenY[j]}" x2="${outX-42}" y2="${outY[oi]}" stroke="${edgeColor(W2[j][k])}" stroke-width="${edgeWidth(W2[j][k])}" opacity="0.45"/>`;
        if (j === oi || j === oi + 1) html += label(hiddenX+24, hiddenY[j], outX-42, outY[oi], W2[j][k]);
      }
    }
    for (let ai = 0; ai < active.length; ai++) {
      const i = active[ai];
      html += `<circle cx="${inputX}" cy="${inputY[ai]}" r="20" fill="#dbeafe" stroke="#1f2937" stroke-width="2"/><text x="${inputX}" y="${inputY[ai]+4}" text-anchor="middle" font-size="10" fill="#1f2937" font-weight="700">${featureNames[i]}</text>`;
    }
    for (let j = 0; j < H; j++) {
      const fill = `rgba(37,99,235,${Math.min(1, 0.12 + h[j] * 0.8)})`;
      html += `<circle cx="${hiddenX}" cy="${hiddenY[j]}" r="${hiddenR}" fill="${fill}" stroke="#1f2937" stroke-width="2"/><text x="${hiddenX}" y="${hiddenY[j]+4}" text-anchor="middle" font-size="${hiddenFont}" fill="${h[j] > 0.5 ? 'white' : '#1f2937'}" font-weight="700">${h[j].toFixed(1)}</text>`;
    }
    for (let oi = 0; oi < top.length; oi++) {
      const [k, prob] = top[oi];
      const nameSize = meals[k].length > 9 ? 9 : 11;
      html += `<circle cx="${outX}" cy="${outY[oi]}" r="28" fill="#dbeafe" stroke="#1f2937" stroke-width="2"/><text x="${outX}" y="${outY[oi]-2}" text-anchor="middle" font-size="${nameSize}" fill="#1f2937" font-weight="700">${meals[k]}</text><text x="${outX}" y="${outY[oi]+12}" text-anchor="middle" font-size="10" fill="#2563eb">${prob.toFixed(2)}</text>`;
    }
    html += `<text x="${inputX}" y="342" text-anchor="middle" font-size="11" fill="#6b7280">active clues</text><text x="${hiddenX}" y="342" text-anchor="middle" font-size="11" fill="#6b7280">hidden patterns</text><text x="${outX}" y="342" text-anchor="middle" font-size="11" fill="#6b7280">top guesses</text>`;
    svg.innerHTML = html;
  }

  function renderWeightDump() {
    dump.update({ w1: W1, b1: b1, w2: W2, b2: b2 });
    const countEl = document.getElementById('a-param-count');
    if (countEl) countEl.textContent = D * H + H + H * C + C;
  }

  function render() {
    const x = vectorFromState();
    const {h, p} = forward(x);
    const sorted = p.map((v,i) => [meals[i], v]).sort((a,b) => b[1] - a[1]);
    document.getElementById('a-bars').innerHTML = sorted.map(([name, prob]) => (
      `<div class="bar-container"><span class="label">${name}</span><div class="bar-track"><div class="bar" style="width:${prob*100}%;"></div></div><span class="pct">${prob.toFixed(2)}</span></div>`
    )).join('');
    document.getElementById('a-features').innerHTML = `<div class="feature-grid">${
      featureNames.map((name, i) => (
        `<div class="feature-bit ${x[i] ? 'active' : ''}"><span class="name">${name}</span><span class="bit">${x[i]}</span></div>`
      )).join('')
    }</div>`;
    document.getElementById('a-epoch').textContent = epoch;
    document.getElementById('a-acc').textContent = accuracy().toFixed(3);
    renderNetwork(x, h, p);
    renderWeightDump();
  }

  document.getElementById('a-course').addEventListener('change', e => { state.course = e.target.value; render(); });
  document.getElementById('a-temp').addEventListener('change', e => { state.temp = e.target.value; render(); });
  document.getElementById('a-taste').addEventListener('change', e => { state.taste = e.target.value; render(); });
  document.getElementById('a-diet').addEventListener('change', e => { state.diet = e.target.value; render(); });
  document.getElementById('a-base').addEventListener('change', e => { state.base = e.target.value; render(); });
  document.getElementById('a-hidden-count').addEventListener('change', reset);
  document.getElementById('a-step').addEventListener('click', () => train(1));
  document.getElementById('a-fast').addEventListener('click', () => train(1000));
  document.getElementById('a-reset').addEventListener('click', reset);
  document.getElementById('a-dump-toggle').addEventListener('click', () => {
    const dump = document.getElementById('a-weight-dump');
    dump.hidden = !dump.hidden;
    document.getElementById('a-dump-toggle').textContent = dump.hidden
      ? 'show learned numbers'
      : 'hide learned numbers';
  });
  reset();
})();

