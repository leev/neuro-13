(function descentWidget() {
  const canvas = document.getElementById('gd-canvas');
  if (!canvas) return;
  const { ctx, w: W, h: H } = Book.hidpi(canvas);
  const X_RANGE = 4.5, PX_PER_LOSS = 40, FLOOR = H - 30;
  const BETA = 0.9, MAX_ROLL = 300, TRAIL = 60;

  const hills = {
    bowl: {
      start: -3.5,
      f: x => 0.35 * x * x + 0.3,
      df: x => 0.7 * x,
    },
    bumpy: {
      start: -3.5,
      f: x => 0.9 + 0.18 * (x - 1.4) ** 2 + (x > 1.4 ? 0.03 * (x - 1.4) ** 4 : 0) - 0.9 * Math.exp(-6.25 * (x + 1.6) ** 2),
      df: x => 0.36 * (x - 1.4) + (x > 1.4 ? 0.12 * (x - 1.4) ** 3 : 0) + 11.25 * (x + 1.6) * Math.exp(-6.25 * (x + 1.6) ** 2),
      labels: [[-1.5, 'small dip'], [1.4, 'deepest valley']],
    },
  };

  const el = id => document.getElementById(id);
  const lrSlider = el('gd-lr'), momentumToggle = el('gd-momentum'), rollBtn = el('gd-fast');
  let hill = hills.bowl, x, v, steps, trail, timer = null;

  function learningRate() { return 0.01 * Math.pow(320, lrSlider.value / 100); }
  function toPx(xv) { return ((xv + X_RANGE) / (2 * X_RANGE)) * W; }
  function toPy(loss) { return FLOOR - loss * PX_PER_LOSS; }
  function offMap() { return Math.abs(x) > X_RANGE; }
  function settled() { return Math.abs(hill.df(x)) < 0.02 && Math.abs(v) < 0.005; }

  function status() {
    if (offMap()) return 'The ball flew off the map. Steps this big make the loss grow instead of shrink.';
    if (steps === 0) return 'Ready at the top of the hill.';
    if (!settled()) return steps >= MAX_ROLL ? 'Still moving after ' + steps + ' steps.' : '';
    if (hill === hills.bumpy && x < -1) return 'Stuck in the small dip after ' + steps + ' steps. The slope here is zero, so there is nothing to push the ball on.';
    return 'Settled at the lowest point after ' + steps + ' steps.';
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, FLOOR); ctx.lineTo(W, FLOOR);
    ctx.stroke();

    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let px = 0; px <= W; px++) {
      const py = toPy(hill.f((px / W) * 2 * X_RANGE - X_RANGE));
      if (px === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'left';
    ctx.fillText('loss ↑', 10, 16);
    ctx.fillText('← lower weight', 10, H - 10);
    ctx.textAlign = 'right';
    ctx.fillText('higher weight →', W - 10, H - 10);
    ctx.textAlign = 'center';
    for (const [lx, text] of hill.labels || []) ctx.fillText(text, toPx(lx), toPy(hill.f(lx)) + 22);

    trail.forEach((tx, i) => {
      if (Math.abs(tx) > X_RANGE) return;
      ctx.fillStyle = 'rgba(220, 38, 38, ' + (0.08 + 0.3 * (i + 1) / trail.length).toFixed(2) + ')';
      ctx.beginPath();
      ctx.arc(toPx(tx), toPy(hill.f(tx)), 5, 0, Math.PI * 2);
      ctx.fill();
    });

    const bx = Math.max(12, Math.min(W - 12, toPx(x)));
    const by = Math.max(13, Math.min(FLOOR, toPy(hill.f(x))));
    ctx.fillStyle = '#dc2626';
    ctx.beginPath();
    ctx.arc(bx, by, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#7f1d1d';
    ctx.lineWidth = 2;
    ctx.stroke();

    const big = n => (Math.abs(n) >= 1000 ? n.toExponential(1) : n.toFixed(2));
    el('gd-lr-v').textContent = learningRate().toFixed(2);
    el('gd-steps').textContent = steps;
    el('gd-weight').textContent = big(x);
    el('gd-loss').textContent = big(hill.f(x));
    el('gd-slope').textContent = big(hill.df(x));
    el('gd-msg').textContent = status();
  }

  function step() {
    trail.push(x);
    if (trail.length > TRAIL) trail.shift();
    v = (momentumToggle.classList.contains('on') ? BETA * v : 0) - learningRate() * hill.df(x);
    x = Math.max(-1e6, Math.min(1e6, x + v));
    steps++;
  }

  function stopRolling() {
    if (timer === null) return;
    clearInterval(timer);
    timer = null;
    rollBtn.textContent = '▶ roll until it stops';
  }

  function roll() {
    if (timer !== null) { stopRolling(); return; }
    rollBtn.textContent = '■ stop';
    let taken = 0;
    timer = setInterval(() => {
      step();
      taken++;
      if (settled() || Math.abs(x) > 1000 || taken >= MAX_ROLL) stopRolling();
      draw();
    }, 40);
  }

  function reset() {
    stopRolling();
    x = hill.start;
    v = 0;
    steps = 0;
    trail = [];
    draw();
  }

  el('gd-step').addEventListener('click', () => { stopRolling(); step(); draw(); });
  rollBtn.addEventListener('click', roll);
  el('gd-reset').addEventListener('click', reset);
  lrSlider.addEventListener('input', draw);
  momentumToggle.addEventListener('click', () => {
    momentumToggle.classList.toggle('on');
    reset();
  });
  document.querySelectorAll('#gd-hill .toggle').forEach(t => {
    t.addEventListener('click', () => {
      document.querySelectorAll('#gd-hill .toggle').forEach(o => o.classList.remove('on'));
      t.classList.add('on');
      hill = hills[t.dataset.hill];
      reset();
    });
  });
  reset();
})();
