(function leftOutWidget() {
  const slider = document.getElementById('lo-kids');
  if (!slider) return;
  const $ = id => document.getElementById(id);
  const ADULTS = 1000, MAX_KIDS = 200, TEST_ADULTS = 1900, TEST_KIDS = 100;
  const HIDDEN = 8, ROUNDS = 600, LR = 1, CROWDS = 20;
  const B1 = 2 * HIDDEN, W2 = 3 * HIDDEN, B2 = 4 * HIDDEN;
  const PEOPLE = {
    adult: { shoulder: [41, 2.8], standing: [168, 9], ducking: [128, 11] },
    kid: { shoulder: [28, 2.4], standing: [118, 7], ducking: [88, 9] },
  };
  const BLUE = '37, 99, 235', ORANGE = '234, 88, 12';
  const W = 318, H = 270, LEFT = 38, TOP = 8, PLOT_W = 272, PLOT_H = 232, CELL = 4;
  const COLS = PLOT_W / CELL, ROWS = PLOT_H / CELL;
  const S0 = 19, S1 = 51, H0 = 55, H1 = 205;
  const px = cm => LEFT + (cm - S0) / (S1 - S0) * PLOT_W;
  const py = cm => TOP + (H1 - cm) / (H1 - H0) * PLOT_H;
  const fx = cm => (cm - 35) / 24, fy = cm => (cm - 130) / 30;
  const fmt = n => n.toLocaleString('en-US');

  const trainView = Book.hidpi($('lo-train')).ctx, testView = Book.hidpi($('lo-test')).ctx;
  const panels = $('lo-panels'), bars = $('lo-bars');
  const grid = new Float32Array(COLS * ROWS);
  const grad = new Float64Array(B2 + 1), hidden = new Float64Array(HIDDEN);
  const calm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let crowd = 1, kids = +slider.value, runId = 0, timer = 0, started = false;
  let trainAdults, trainKids, testAdults, testKids;

  function gaussian(rand) {
    return () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  }

  function makePeople(kind, n, seed) {
    const gauss = gaussian(Book.rng(seed)), spec = PEOPLE[kind];
    const group = { n, cm: new Float64Array(2 * n), x: new Float64Array(2 * n), duck: new Uint8Array(n), wrong: new Uint8Array(n) };
    for (let i = 0; i < n; i++) {
      group.duck[i] = i % 2;
      const [typical, spread] = group.duck[i] ? spec.ducking : spec.standing;
      group.cm[2 * i] = spec.shoulder[0] + spec.shoulder[1] * gauss();
      group.cm[2 * i + 1] = typical + spread * gauss();
      group.x[2 * i] = fx(group.cm[2 * i]);
      group.x[2 * i + 1] = fy(group.cm[2 * i + 1]);
    }
    return group;
  }

  function gather() {
    trainAdults = makePeople('adult', ADULTS, 1000 + crowd);
    trainKids = makePeople('kid', MAX_KIDS, 2000 + crowd);
    testAdults = makePeople('adult', TEST_ADULTS, 3000 + crowd);
    testKids = makePeople('kid', TEST_KIDS, 4000 + crowd);
  }

  function pileOf(count) {
    const n = ADULTS + count, x = new Float64Array(2 * n), duck = new Uint8Array(n);
    x.set(trainAdults.x);
    x.set(trainKids.x.subarray(0, 2 * count), 2 * ADULTS);
    duck.set(trainAdults.duck);
    duck.set(trainKids.duck.subarray(0, count), ADULTS);
    return { n, x, duck };
  }

  function freshWeights() {
    const gauss = gaussian(Book.rng(5000 + crowd));
    const w = new Float64Array(B2 + 1);
    for (let k = 0; k < B1; k++) w[k] = gauss() * 0.5;
    for (let j = 0; j < HIDDEN; j++) w[W2 + j] = gauss() * 0.5;
    return w;
  }

  function duckChance(w, x1, x2) {
    let z = w[B2];
    for (let j = 0; j < HIDDEN; j++) z += w[W2 + j] * Math.tanh(w[2 * j] * x1 + w[2 * j + 1] * x2 + w[B1 + j]);
    return 1 / (1 + Math.exp(-z));
  }

  function nudge(w, pile) {
    grad.fill(0);
    for (let i = 0; i < pile.n; i++) {
      const x1 = pile.x[2 * i], x2 = pile.x[2 * i + 1];
      let z = w[B2];
      for (let j = 0; j < HIDDEN; j++) {
        hidden[j] = Math.tanh(w[2 * j] * x1 + w[2 * j + 1] * x2 + w[B1 + j]);
        z += w[W2 + j] * hidden[j];
      }
      const blame = (1 / (1 + Math.exp(-z)) - pile.duck[i]) / pile.n;
      grad[B2] += blame;
      for (let j = 0; j < HIDDEN; j++) {
        grad[W2 + j] += blame * hidden[j];
        const back = blame * w[W2 + j] * (1 - hidden[j] * hidden[j]);
        grad[2 * j] += back * x1;
        grad[2 * j + 1] += back * x2;
        grad[B1 + j] += back;
      }
    }
    for (let k = 0; k < w.length; k++) w[k] -= LR * grad[k];
  }

  function grade(w, group) {
    let right = 0;
    for (let i = 0; i < group.n; i++) {
      const saysDuck = duckChance(w, group.x[2 * i], group.x[2 * i + 1]) > 0.5;
      group.wrong[i] = saysDuck === (group.duck[i] === 1) ? 0 : 1;
      right += 1 - group.wrong[i];
    }
    return right;
  }

  function survey(w) {
    for (let r = 0; r < ROWS; r++) {
      const x2 = fy(H1 - (r + 0.5) * CELL * (H1 - H0) / PLOT_H);
      for (let c = 0; c < COLS; c++) {
        grid[r * COLS + c] = duckChance(w, fx(S0 + (c + 0.5) * CELL * (S1 - S0) / PLOT_W), x2);
      }
    }
  }

  function paintAnswers(ctx) {
    for (const duck of [0, 1]) {
      ctx.fillStyle = `rgba(${duck ? ORANGE : BLUE}, 0.13)`;
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if ((grid[r * COLS + c] > 0.5 ? 1 : 0) === duck) ctx.fillRect(LEFT + c * CELL, TOP + r * CELL, CELL, CELL);
        }
      }
    }
    const cuts = [];
    ctx.strokeStyle = '#1f2937';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let r = 0; r + 1 < ROWS; r++) {
      for (let c = 0; c + 1 < COLS; c++) {
        const nw = grid[r * COLS + c] - 0.5, ne = grid[r * COLS + c + 1] - 0.5;
        const sw = grid[(r + 1) * COLS + c] - 0.5, se = grid[(r + 1) * COLS + c + 1] - 0.5;
        cuts.length = 0;
        if (nw * ne < 0) cuts.push(c + nw / (nw - ne), r);
        if (sw * se < 0) cuts.push(c + sw / (sw - se), r + 1);
        if (nw * sw < 0) cuts.push(c, r + nw / (nw - sw));
        if (ne * se < 0) cuts.push(c + 1, r + ne / (ne - se));
        for (let k = 0; k + 3 < cuts.length; k += 4) {
          ctx.moveTo(LEFT + (cuts[k] + 0.5) * CELL, TOP + (cuts[k + 1] + 0.5) * CELL);
          ctx.lineTo(LEFT + (cuts[k + 2] + 0.5) * CELL, TOP + (cuts[k + 3] + 0.5) * CELL);
        }
      }
    }
    ctx.stroke();
  }

  function paintDots(ctx, group, count, radius, alpha) {
    for (const duck of [0, 1]) {
      ctx.fillStyle = `rgba(${duck ? ORANGE : BLUE}, ${alpha})`;
      ctx.beginPath();
      for (let i = 0; i < count; i++) {
        if (group.duck[i] !== duck) continue;
        const x = px(group.cm[2 * i]), y = py(group.cm[2 * i + 1]);
        ctx.moveTo(x + radius, y);
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
      }
      ctx.fill();
    }
  }

  function paintMistakes(ctx, group) {
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = '#111827';
    for (let i = 0; i < group.n; i++) {
      if (!group.wrong[i]) continue;
      ctx.fillStyle = `rgb(${group.duck[i] ? ORANGE : BLUE})`;
      ctx.beginPath();
      ctx.arc(px(group.cm[2 * i]), py(group.cm[2 * i + 1]), 3.2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  function paintAxes(ctx) {
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.strokeRect(LEFT + 0.5, TOP + 0.5, PLOT_W - 1, PLOT_H - 1);
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const cm of [100, 150, 200]) ctx.fillText(cm, LEFT - 5, py(cm));
    ctx.textAlign = 'center';
    for (const cm of [20, 30, 40, 50]) ctx.fillText(cm, px(cm), TOP + PLOT_H + 9);
    ctx.font = '11px system-ui, -apple-system, sans-serif';
    ctx.fillText('shoulder width (cm)', LEFT + PLOT_W / 2, H - 7);
    ctx.save();
    ctx.translate(9, TOP + PLOT_H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('head height (cm)', 0, 0);
    ctx.restore();
    ctx.fillStyle = '#374151';
    ctx.font = '600 11px system-ui, -apple-system, sans-serif';
    ctx.fillText('grown-ups', px(41), py(198));
  }

  function paintPanel(ctx, hasModel, drawPeople) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#fff';
    ctx.fillRect(LEFT, TOP, PLOT_W, PLOT_H);
    ctx.save();
    ctx.beginPath();
    ctx.rect(LEFT, TOP, PLOT_W, PLOT_H);
    ctx.clip();
    if (hasModel) paintAnswers(ctx);
    drawPeople();
    ctx.restore();
    paintAxes(ctx);
  }

  function paintPile(hasModel) {
    paintPanel(trainView, hasModel, () => {
      paintDots(trainView, trainAdults, ADULTS, 1.6, 0.55);
      paintDots(trainView, trainKids, kids, 2.2, 0.9);
    });
    const ctx = trainView;
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'center';
    ctx.font = '600 11px system-ui, -apple-system, sans-serif';
    if (kids) {
      ctx.fillText(kids === 1 ? '1 kid' : kids + ' kids', px(28), py(162));
      return;
    }
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#9ca3af';
    ctx.strokeRect(px(21.5), py(140), px(34.5) - px(21.5), py(62) - py(140));
    ctx.setLineDash([]);
    ctx.fillStyle = '#6b7280';
    ctx.fillText('no kids', px(28), py(106));
    ctx.fillText('in the pile', px(28), py(96));
  }

  function paintCrowd(hasModel) {
    paintPanel(testView, hasModel, () => {
      paintDots(testView, testAdults, TEST_ADULTS, 1.5, hasModel ? 0.3 : 0.55);
      paintDots(testView, testKids, TEST_KIDS, 2.2, hasModel ? 0.4 : 0.9);
      if (hasModel) {
        paintMistakes(testView, testAdults);
        paintMistakes(testView, testKids);
      }
    });
    testView.fillStyle = '#374151';
    testView.textAlign = 'center';
    testView.font = '600 11px system-ui, -apple-system, sans-serif';
    testView.fillText('100 kids', px(28), py(162));
  }

  function setBar(who, right, total) {
    const share = right / total, bar = $('lo-bar-' + who);
    bar.style.width = 100 * share + '%';
    bar.style.background = share < 0.7 ? 'var(--bad)' : share < 0.9 ? 'var(--warn)' : 'var(--good)';
    $('lo-pct-' + who).textContent = (100 * share).toFixed(1) + '%';
    $('lo-count-' + who).textContent = `${fmt(right)} of ${fmt(total)}`;
  }

  function show(w, round) {
    survey(w);
    const rightAdults = grade(w, testAdults), rightKids = grade(w, testKids);
    paintPile(true);
    paintCrowd(true);
    setBar('all', rightAdults + rightKids, TEST_ADULTS + TEST_KIDS);
    setBar('adults', rightAdults, TEST_ADULTS);
    setBar('kids', rightKids, TEST_KIDS);
    $('lo-round').textContent = round;
  }

  function describePile() {
    $('lo-kids-v').textContent = kids;
    $('lo-pile').textContent = `${fmt(ADULTS)} grown-ups + ${kids} ${kids === 1 ? 'kid' : 'kids'}`;
    $('lo-share').textContent = (100 * kids / (ADULTS + kids)).toFixed(1) + '%';
    $('lo-crowd-n').textContent = crowd;
  }

  async function train() {
    const id = ++runId;
    started = true;
    panels.classList.remove('lo-stale');
    bars.classList.remove('lo-stale');
    const pile = pileOf(kids), w = freshWeights(), chunk = calm ? 150 : 12;
    show(w, 0);
    for (let round = 0; round < ROUNDS;) {
      await new Promise(resolve => requestAnimationFrame(resolve));
      if (id !== runId) return;
      for (let k = 0; k < chunk; k++, round++) nudge(w, pile);
      show(w, round);
    }
  }

  function queueTraining() {
    runId++;
    describePile();
    paintPile(false);
    if (!started) return;
    panels.classList.add('lo-stale');
    bars.classList.add('lo-stale');
    clearTimeout(timer);
    timer = setTimeout(train, 140);
  }

  slider.addEventListener('input', () => {
    kids = +slider.value;
    queueTraining();
  });
  $('lo-presets').addEventListener('click', e => {
    const btn = e.target.closest('[data-kids]');
    if (!btn) return;
    slider.value = btn.dataset.kids;
    kids = +slider.value;
    queueTraining();
  });
  $('lo-crowd').addEventListener('click', () => {
    crowd = crowd % CROWDS + 1;
    gather();
    queueTraining();
    paintCrowd(false);
  });

  gather();
  describePile();
  paintPile(false);
  paintCrowd(false);
  const box = slider.closest('.interactive');
  if ('IntersectionObserver' in window && box) {
    new IntersectionObserver((entries, watcher) => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      watcher.disconnect();
      if (!started) train();
    }, { threshold: 0.2 }).observe(box);
  } else {
    train();
  }
})();

(function smokeAlarmWidget() {
  const slider = document.getElementById('sa-line');
  if (!slider) return;
  const $ = id => document.getElementById(id);
  const HOUSES = 1000, FIRES = 10;
  const W = 620, H = 270, LEFT = 24, PLOT_W = 572, FIRE_TOP = 32, FLOOR = 232, DOT = 2.86;
  const canvas = $('sa-town'), ctx = Book.hidpi(canvas).ctx;
  const xOf = hundredths => LEFT + hundredths / 100 * PLOT_W;
  const houses = makeTown();

  function makeTown() {
    const rand = Book.rng(284);
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    const town = [];
    for (let i = 0; i < HOUSES; i++) {
      const fire = i < FIRES;
      const smoky = !fire && rand() < 0.08;
      const z = fire ? 2 + 1.7 * gauss() : smoky ? 0.3 + 1.2 * gauss() : -2.4 + 0.9 * gauss();
      town.push({ fire, milli: Math.round(sigmoid(z) * 1000) });
    }
    const stack = new Array(100).fill(0);
    town.filter(h => h.fire).sort((a, b) => a.milli - b.milli).forEach((h, k) => {
      h.x = xOf(h.milli / 10);
      h.y = FIRE_TOP + (k % 3) * 13;
    });
    town.filter(h => !h.fire).sort((a, b) => a.milli - b.milli).forEach(h => {
      const bin = Math.min(99, Math.floor(h.milli / 10)), slot = stack[bin]++;
      h.x = xOf(bin) + 0.25 + (slot % 2) * DOT;
      h.y = FLOOR - (Math.floor(slot / 2) + 1) * DOT;
    });
    return town;
  }

  function paint(line) {
    const x = xOf(line);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(234, 88, 12, 0.08)';
    ctx.fillRect(x, 20, W - x, FLOOR - 16);
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LEFT, 82.5);
    ctx.lineTo(LEFT + PLOT_W, 82.5);
    ctx.moveTo(LEFT, FLOOR + 3.5);
    ctx.lineTo(LEFT + PLOT_W, FLOOR + 3.5);
    ctx.stroke();

    for (const rings of [false, true]) {
      ctx.fillStyle = rings ? '#ea580c' : '#9ca3af';
      for (const h of houses) if (!h.fire && h.rings === rings) ctx.fillRect(h.x, h.y, 2.4, 2.4);
    }
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#fff';
    for (const h of houses) {
      if (!h.fire) continue;
      ctx.fillStyle = h.rings ? '#16a34a' : '#dc2626';
      ctx.beginPath();
      ctx.arc(h.x, h.y, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }

    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 20);
    ctx.lineTo(x, FLOOR + 4);
    ctx.stroke();
    ctx.fillStyle = '#111827';
    ctx.beginPath();
    ctx.moveTo(x - 6, 12);
    ctx.lineTo(x + 6, 12);
    ctx.lineTo(x, 21);
    ctx.fill();

    ctx.textBaseline = 'middle';
    ctx.font = '600 12px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'right';
    if (x > LEFT + 96) ctx.fillText('stays quiet', x - 12, 11);
    ctx.fillStyle = '#c2410c';
    ctx.textAlign = 'left';
    if (x < W - 100) ctx.fillText('alarm rings', x + 12, 11);

    ctx.font = '12px system-ui, -apple-system, sans-serif';
    for (const [caption, y] of [['the 10 houses really on fire', 71], ['the 990 houses not on fire', 95]]) {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
      ctx.fillRect(LEFT + 6, y - 9, ctx.measureText(caption).width + 10, 18);
      ctx.fillStyle = '#374151';
      ctx.fillText(caption, LEFT + 11, y);
    }
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    for (const tick of [0, 25, 50, 75, 100]) ctx.fillText((tick / 100).toFixed(2), xOf(tick), FLOOR + 15);
    ctx.font = '11px system-ui, -apple-system, sans-serif';
    ctx.fillText('smoke score: how fire-like the model thinks each house looks', LEFT + PLOT_W / 2, FLOOR + 31);
  }

  function setMeter(name, share, text) {
    $('sa-' + name + '-bar').style.width = 100 * share + '%';
    $('sa-' + name).textContent = text;
  }

  function render() {
    const line = +slider.value;
    let caught = 0, missed = 0, falseAlarms = 0, quiet = 0;
    for (const h of houses) {
      h.rings = h.milli >= line * 10;
      if (h.fire) h.rings ? caught++ : missed++;
      else h.rings ? falseAlarms++ : quiet++;
    }
    const alarms = caught + falseAlarms, right = caught + quiet;
    paint(line);
    $('sa-line-v').textContent = (line / 100).toFixed(2);
    $('sa-caught').textContent = caught;
    $('sa-missed').textContent = missed;
    $('sa-false').textContent = falseAlarms;
    $('sa-quiet').textContent = quiet;
    $('sa-caught-label').textContent = caught === 1 ? 'fire caught' : 'fires caught';
    $('sa-missed-label').textContent = missed === 1 ? 'fire missed' : 'fires missed';
    $('sa-false-label').textContent = falseAlarms === 1 ? 'false alarm' : 'false alarms';
    setMeter('acc', right / HOUSES, (100 * right / HOUSES).toFixed(1) + '%');
    setMeter('catch', caught / FIRES, `${caught} of ${FIRES}`);
    setMeter('real', alarms ? caught / alarms : 0, alarms ? `${caught} of ${alarms.toLocaleString('en-US')}` : 'no alarms rang');
    $('sa-note').textContent = line === 100
      ? 'The lazy model: no house scores as high as 1.00, so no alarm ever rings.'
      : line === 0
        ? 'The jumpy model: every house scores at least 0.00, so all 1,000 alarms ring.'
        : `An alarm rings in every house that scores ${(line / 100).toFixed(2)} or more: that is ${alarms} of the 1,000 houses.`;
  }

  function moveLine(to) {
    slider.value = Math.max(0, Math.min(100, Math.round(to)));
    render();
  }

  function drag(e) {
    moveLine((Book.pointer(canvas, e).x - LEFT) / PLOT_W * 100);
  }

  slider.addEventListener('input', render);
  $('sa-lazy').addEventListener('click', () => moveLine(100));
  $('sa-jumpy').addEventListener('click', () => moveLine(0));
  $('sa-middle').addEventListener('click', () => moveLine(50));
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    drag(e);
  });
  canvas.addEventListener('pointermove', e => {
    if (e.buttons) drag(e);
  });
  render();
})();
