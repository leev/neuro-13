(function mazeWidget() {
  const N = 7;
  const EMPTY = 0, WALL = 1, LAVA = 2, GOAL = 3, SNACK = 4;
  const REWARD = [0, 0, -10, 10, 1];
  const MOVES = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  const MOVE_NAMES = ['up', 'right', 'down', 'left'];
  const MAX_MOVES = 200;
  const DEFAULTS = { eps: 0.1, alpha: 0.5, gamma: 0.9 };
  const DEFAULT_MAZE = [
    '....^..',
    '.##.##.',
    '.......',
    '.#####.',
    '.#G....',
    '.^#^##.',
    'S......',
  ];

  const cellName = s => 'ABCDEFG'[s % N] + (Math.floor(s / N) + 1);

  function parseMaze(rows) {
    const cells = new Uint8Array(N * N);
    let start = 0;
    rows.forEach((row, r) => [...row].forEach((ch, c) => {
      const s = r * N + c;
      if (ch === 'S') start = s;
      else cells[s] = { '#': WALL, '^': LAVA, G: GOAL, o: SNACK }[ch] || EMPTY;
    }));
    return { cells, start };
  }

  function landing(world, s, a) {
    const r = Math.floor(s / N) + MOVES[a][0], c = s % N + MOVES[a][1];
    if (r < 0 || r >= N || c < 0 || c >= N || world.cells[r * N + c] === WALL) return s;
    return r * N + c;
  }

  function best(q, s, rand) {
    let value = -Infinity, ties = [];
    for (let a = 0; a < 4; a++) {
      const v = q[s * 4 + a];
      if (v > value) { value = v; ties = [a]; } else if (v === value) ties.push(a);
    }
    return { value, tied: ties.length, action: rand ? ties[Math.floor(rand() * ties.length)] : ties[0] };
  }

  function step(world, q, s, p, rand) {
    const explored = rand() < p.eps;
    const action = explored ? Math.floor(rand() * 4) : best(q, s, rand).action;
    const to = landing(world, s, action);
    const reward = to === s ? 0 : REWARD[world.cells[to]];
    const done = world.cells[to] === GOAL;
    const old = q[s * 4 + action];
    const next = done ? 0 : best(q, to).value;
    const now = old + p.alpha * (reward + p.gamma * next - old);
    q[s * 4 + action] = now;
    return { from: s, action, to, reward, done, explored, old, next, now };
  }

  function runEpisode(world, q, p, rand) {
    let s = world.start, moves = 0, points = 0;
    while (moves < MAX_MOVES) {
      const m = step(world, q, s, p, rand);
      moves++;
      points += m.reward;
      s = m.to;
      if (m.done) return { moves, points, reached: true };
    }
    return { moves, points, reached: false };
  }

  function plan(world, q) {
    const cells = [world.start], seen = new Set(cells);
    let s = world.start;
    for (;;) {
      const b = best(q, s);
      if (b.tied > 1 && b.value <= 0) return { cells, ends: 'lost' };
      s = landing(world, s, b.action);
      cells.push(s);
      if (world.cells[s] === GOAL) return { cells, ends: 'goal' };
      if (seen.has(s)) return { cells, ends: 'loop' };
      seen.add(s);
    }
  }

  function shortestSafe(world) {
    const dist = new Int16Array(N * N).fill(-1);
    const queue = [world.start];
    dist[world.start] = 0;
    while (queue.length) {
      const s = queue.shift();
      if (world.cells[s] === GOAL) return dist[s];
      for (let a = 0; a < 4; a++) {
        const t = landing(world, s, a);
        if (dist[t] >= 0 || world.cells[t] === LAVA) continue;
        dist[t] = dist[s] + 1;
        queue.push(t);
      }
    }
    return -1;
  }

  if (typeof module === 'object' && module.exports) {
    module.exports = { N, EMPTY, WALL, LAVA, GOAL, SNACK, MAX_MOVES, DEFAULTS, DEFAULT_MAZE, cellName, parseMaze, landing, best, step, runEpisode, plan, shortestSafe };
    return;
  }

  const $ = id => document.getElementById(id);
  const canvas = $('rl-maze');
  if (!canvas) return;

  const SEED = 130;
  const CELL = 60, LEFT = 22, TOP = 20;
  const WATCH_MS = 70, PAUSE_MS = 550, FAST_MS = 110, BURST = 1000, BURST_CHUNK = 50;
  const INK = '#1b1f27', ORANGE = '#ea580c';
  const TOOL_KIND = { wall: WALL, lava: LAVA, snack: SNACK, empty: EMPTY };

  const { ctx, w: W, h: H } = Book.hidpi(canvas);
  let world, q, rand, params, pos, moves, points, trail, over, last, history, safe;
  let mode = 'idle', tool = 'look', nextAt = 0, burstLeft = 0, anim = null, raf = 0, painted = -1, drawnEpisodes = -1, stale = true;

  const cx = s => LEFT + (s % N) * CELL + CELL / 2;
  const cy = s => TOP + Math.floor(s / N) * CELL + CELL / 2;
  const signed = v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v);

  function fmt(v) {
    const a = Math.abs(v);
    const text = a >= 100 ? a.toFixed(0) : a >= 9.995 ? a.toFixed(1) : a.toFixed(2);
    return (v < 0 && +text !== 0 ? '−' : '') + text;
  }

  function newEpisode() {
    pos = world.start;
    moves = 0;
    points = 0;
    trail = [pos];
    over = false;
  }

  function advance() {
    stale = true;
    if (over) {
      newEpisode();
      last = null;
      return;
    }
    last = step(world, q, pos, params, rand);
    pos = last.to;
    moves++;
    points += last.reward;
    trail.push(pos);
    if (last.done || moves >= MAX_MOVES) {
      history.push({ moves, points, reached: last.done, safe });
      over = true;
    }
  }

  function finishEpisode() {
    if (over) newEpisode();
    while (!over) advance();
  }

  function eraseMemory() {
    q = new Float64Array(N * N * 4);
    history = [];
    last = null;
    newEpisode();
  }

  function resetEverything() {
    world = parseMaze(DEFAULT_MAZE);
    rand = Book.rng(SEED);
    $('rl-eps').value = DEFAULTS.eps;
    $('rl-alpha').value = DEFAULTS.alpha;
    $('rl-gamma').value = DEFAULTS.gamma;
    readSliders();
    eraseMemory();
    safe = shortestSafe(world);
  }

  function readSliders() {
    params = { eps: +$('rl-eps').value, alpha: +$('rl-alpha').value, gamma: +$('rl-gamma').value };
    const pct = x => Math.round(x * 100) + '%';
    $('rl-eps-v').textContent = pct(params.eps);
    $('rl-alpha-v').textContent = params.alpha.toFixed(2);
    $('rl-gamma-v').textContent = params.gamma.toFixed(2);
    $('rl-eps-say').textContent = params.eps === 0 ? 'It never experiments. It always makes its best-rated move.'
      : params.eps === 1 ? 'Every move is random. It never uses what it knows.'
      : `${pct(params.eps)} of its moves are random experiments.`;
    $('rl-alpha-say').textContent = params.alpha === 1 ? 'Each experience replaces the old value completely.'
      : `Each experience moves a value ${pct(params.alpha)} of the way to the new evidence.`;
    $('rl-gamma-say').textContent = params.gamma === 0 ? 'Only the very next reward counts.'
      : params.gamma === 1 ? 'A reward far in the future counts as much as one now.'
      : `A reward one move later counts ${pct(params.gamma)} as much as one now.`;
  }

  function tint(v) {
    if (v === 0) return null;
    const strength = 0.06 + 0.7 * Math.pow(Math.min(1, Math.abs(v) / 10), 0.75);
    return v > 0 ? `rgba(22, 163, 74, ${strength.toFixed(3)})` : `rgba(220, 38, 38, ${strength.toFixed(3)})`;
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawLava(x, y) {
    ctx.fillStyle = '#fecaca';
    ctx.fillRect(x, y, CELL, CELL);
    for (const [peak, color] of [[26, '#f97316'], [38, '#dc2626']]) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, y + CELL);
      for (let i = 0; i <= 6; i++) ctx.lineTo(x + i * CELL / 6, y + (i % 2 ? peak : peak + 14));
      ctx.lineTo(x + CELL, y + CELL);
      ctx.closePath();
      ctx.fill();
    }
  }

  function drawStar(x, y, pulse) {
    const R = 19 + pulse * 4;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const r = i % 2 ? R * 0.45 : R, t = -Math.PI / 2 + i * Math.PI / 5;
      ctx.lineTo(x + r * Math.cos(t), y + r * Math.sin(t));
    }
    ctx.closePath();
    ctx.fillStyle = '#fbbf24';
    ctx.fill();
    ctx.strokeStyle = '#b45309';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function drawSnack(x, y) {
    ctx.fillStyle = '#d97706';
    ctx.strokeStyle = '#92400e';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#78350f';
    for (const [dx, dy] of [[-4, -2.5], [3.5, -4], [0.5, 4.5], [4.5, 2]]) {
      ctx.beginPath();
      ctx.arc(x + dx, y + dy, 1.8, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function drawArrow(s, a, strong) {
    const [dr, dc] = MOVES[a], x = cx(s), y = cy(s), len = 12;
    ctx.strokeStyle = ctx.fillStyle = strong ? INK : 'rgba(27, 31, 39, 0.38)';
    ctx.lineWidth = strong ? 3 : 2;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x - dc * len, y - dr * len);
    ctx.lineTo(x + dc * (len - 5), y + dr * (len - 5));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + dc * (len + 2), y + dr * (len + 2));
    ctx.lineTo(x + dc * (len - 7) - dr * 6, y + dr * (len - 7) + dc * 6);
    ctx.lineTo(x + dc * (len - 7) + dr * 6, y + dr * (len - 7) - dc * 6);
    ctx.closePath();
    ctx.fill();
  }

  function drawCreature(x, y, facing, hurt) {
    ctx.fillStyle = 'rgba(27, 31, 39, 0.16)';
    ctx.beginPath();
    ctx.ellipse(x, y + 17, 14, 4, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = hurt ? '#b91c1c' : ORANGE;
    ctx.beginPath();
    ctx.arc(x, y, 17, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 2;
    ctx.stroke();
    const [dr, dc] = MOVES[facing];
    for (const side of [-1, 1]) {
      const ex = x + side * 6.5, ey = y - 3;
      ctx.fillStyle = '#fff';
      ctx.beginPath();
      ctx.arc(ex, ey, 4.6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = INK;
      ctx.beginPath();
      ctx.arc(ex + dc * 1.8, ey + dr * 1.8, 2.1, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function draw(now) {
    const route = plan(world, q);
    const onRoute = new Set(route.ends === 'goal' ? route.cells : []);
    ctx.clearRect(0, 0, W, H);
    ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#6b7280';
    for (let i = 0; i < N; i++) {
      ctx.fillText('ABCDEFG'[i], LEFT + i * CELL + CELL / 2, TOP / 2);
      ctx.fillText(i + 1, LEFT / 2, TOP + i * CELL + CELL / 2);
    }

    for (let s = 0; s < N * N; s++) {
      const x = LEFT + (s % N) * CELL, y = TOP + Math.floor(s / N) * CELL, kind = world.cells[s];
      if (kind === WALL) {
        ctx.fillStyle = '#334155';
        roundRect(x + 1.5, y + 1.5, CELL - 3, CELL - 3, 7);
        ctx.fill();
        continue;
      }
      ctx.fillStyle = '#fff';
      ctx.fillRect(x, y, CELL, CELL);
      const b = best(q, s);
      if (kind === LAVA) drawLava(x, y);
      else if (kind !== GOAL && tint(b.value)) {
        ctx.fillStyle = tint(b.value);
        ctx.fillRect(x, y, CELL, CELL);
      }
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
      if (kind === GOAL) {
        drawStar(x + CELL / 2, y + CELL / 2, over && last && last.done && anim ? Math.sin(Math.min(1, (now - anim.start) / 400) * Math.PI) : 0);
        continue;
      }
      if (kind === SNACK) drawSnack(x + 14, y + 14);
      if (s === world.start) {
        ctx.strokeStyle = ORANGE;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(x + CELL / 2, y + CELL / 2, 23, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      for (let a = 0; a < 4; a++) {
        const v = q[s * 4 + a];
        if (v >= -0.05) continue;
        const [dr, dc] = MOVES[a];
        ctx.strokeStyle = `rgba(220, 38, 38, ${(0.25 + 0.75 * Math.min(1, -v / 10)).toFixed(2)})`;
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x + CELL / 2 + dc * 26.5 - dr * 9, y + CELL / 2 + dr * 26.5 - dc * 9);
        ctx.lineTo(x + CELL / 2 + dc * 26.5 + dr * 9, y + CELL / 2 + dr * 26.5 + dc * 9);
        ctx.stroke();
      }
      if (b.tied === 1 || b.value > 0) drawArrow(s, b.action, onRoute.has(s));
      if (kind !== LAVA && Math.abs(b.value) >= 0.005) {
        ctx.font = '600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'right';
        ctx.fillStyle = b.value > 0 ? '#14532d' : '#991b1b';
        ctx.fillText(fmt(b.value), x + CELL - 5, y + CELL - 11);
      }
    }

    if (trail.length > 1) {
      ctx.strokeStyle = 'rgba(234, 88, 12, 0.3)';
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx(trail[0]), cy(trail[0]));
      for (const s of trail) ctx.lineTo(cx(s), cy(s));
      ctx.stroke();
    }

    let x = cx(pos), y = cy(pos);
    const t = anim ? Math.min(1, (now - anim.start) / WATCH_MS) : 1;
    if (anim && t < 1 && last) {
      const [dr, dc] = MOVES[last.action];
      if (last.from === last.to) {
        x += dc * 9 * Math.sin(Math.PI * t);
        y += dr * 9 * Math.sin(Math.PI * t);
      } else {
        x = cx(last.from) + (x - cx(last.from)) * t;
        y = cy(last.from) + (y - cy(last.from)) * t;
      }
    }
    drawCreature(x, y, last ? last.action : 1, world.cells[pos] === LAVA);
  }

  function describeLast() {
    const el = $('rl-math');
    if (!last) {
      el.textContent = history.length
        ? `Episode ${history.length + 1} begins. It is back at the start, and it keeps everything it has learned.`
        : 'Nothing yet. Press "one move" and its first lesson appears here.';
      return;
    }
    const m = last, kind = world.cells[m.to], move = MOVE_NAMES[m.action];
    const why = m.explored ? 'a random experiment' : 'the move it rated best';
    const what = m.from === m.to ? 'bumped into a wall and stayed put'
      : kind === GOAL ? `reached the goal at ${cellName(m.to)}`
      : kind === LAVA ? `stepped into lava at ${cellName(m.to)}`
      : kind === SNACK ? `ate the snack at ${cellName(m.to)}`
      : `landed on ${cellName(m.to)}`;
    const nextLabel = m.done ? '0 (the episode is over)' : fmt(m.next);
    el.textContent =
      `At ${cellName(m.from)} it tried "${move}" (${why}) and ${what}.\n` +
      `reward: ${signed(m.reward)}     best value from where it landed: ${nextLabel}\n\n` +
      `value of "${move} from ${cellName(m.from)}"\n` +
      `  = old + α × (reward + γ × best next − old)\n` +
      `  = ${fmt(m.old)} + ${params.alpha} × (${signed(m.reward).replace('+', '')} + ${params.gamma} × ${fmt(m.next)} − ${fmt(m.old)})\n` +
      `  = ${fmt(m.now)}` + (fmt(m.now) === fmt(m.old) ? '     (no change)' : `     (was ${fmt(m.old)})`) +
      (!over ? '' : m.done ? `\n\nEpisode over: goal reached in ${moves} moves.` : `\n\nThat was move ${MAX_MOVES}. Out of time, so it goes back to the start.`);
  }

  function drawChart() {
    const steps = history.map(e => e.moves);
    const avg = steps.map((_, i) => {
      const from = Math.max(0, i - 19);
      let sum = 0;
      for (let j = from; j <= i; j++) sum += steps[j];
      return sum / (i - from + 1);
    });
    Book.lineChart($('rl-chart'), {
      series: [
        { color: '#fdba74', data: steps },
        { color: '#94a3b8', data: history.map(e => Math.max(0, e.safe)) },
        { color: '#c2410c', data: avg },
      ],
      xMax: Math.max(40, steps.length), yMin: 0, yMax: 60,
      yFormat: v => (v >= 60 ? '60+' : v.toFixed(0)), xLabel: 'episodes',
    });
  }

  function renderDump() {
    const el = $('rl-dump');
    if (el.hidden) return;
    const col = v => fmt(v).padStart(8);
    const lines = ['cell      up   right    down    left   best move'];
    for (let s = 0; s < N * N; s++) {
      if (world.cells[s] === WALL || world.cells[s] === GOAL) continue;
      const b = best(q, s);
      lines.push(cellName(s).padEnd(4) + [0, 1, 2, 3].map(a => col(q[s * 4 + a])).join('') +
        '   ' + (b.tied > 1 && b.value <= 0 ? 'no idea yet' : MOVE_NAMES[b.action]));
    }
    el.textContent = lines.join('\n');
  }

  function render(now) {
    draw(now);
    if (!stale) return;
    stale = false;
    const route = plan(world, q);
    $('rl-episode').textContent = history.length + (over ? 0 : 1);
    $('rl-moves').textContent = moves;
    $('rl-points').textContent = signed(points);
    $('rl-plan').textContent = route.ends === 'goal' ? `${route.cells.length - 1} moves`
      : route.ends === 'loop' ? 'in circles' : 'none yet';
    $('rl-safe').textContent = safe > 0 ? `${safe} moves` : 'none';
    const recent = history.slice(-20);
    const mean = key => (recent.reduce((sum, e) => sum + e[key], 0) / recent.length).toFixed(1).replace('-', '−');
    $('rl-recent-moves').textContent = recent.length ? mean('moves') : '—';
    $('rl-recent-points').textContent = recent.length ? mean('points') : '—';
    describeLast();
    renderDump();
    if (drawnEpisodes !== history.length) {
      drawnEpisodes = history.length;
      drawChart();
    }
  }

  function setMode(next) {
    mode = next;
    $('rl-watch').textContent = mode === 'watch' ? '❚❚ pause' : '▶ watch';
    $('rl-fast').textContent = mode === 'fast' ? '❚❚ pause' : '▶▶ fast';
    $('rl-watch').classList.toggle('stop', mode === 'watch');
    $('rl-fast').classList.toggle('stop', mode === 'fast');
    $('rl-burst').disabled = mode === 'burst';
    nextAt = 0;
    kick();
  }

  function kick() {
    if (!raf) raf = requestAnimationFrame(frame);
  }

  function frame(now) {
    raf = 0;
    if (mode === 'watch' && now >= nextAt) {
      if (over || now - nextAt > 4 * WATCH_MS) nextAt = now;
      do {
        advance();
        nextAt += WATCH_MS;
      } while (now >= nextAt && !over);
      if (over) nextAt = now + PAUSE_MS;
      anim = { start: now };
    } else if (mode === 'fast' && now >= nextAt) {
      finishEpisode();
      anim = null;
      nextAt = now + FAST_MS;
    } else if (mode === 'burst') {
      for (let i = 0; i < BURST_CHUNK; i++) finishEpisode();
      anim = null;
      burstLeft -= BURST_CHUNK;
      if (burstLeft <= 0) setMode('idle');
    }
    render(now);
    if (mode !== 'idle' || (anim && now - anim.start < 400)) kick();
  }

  function refresh() {
    safe = shortestSafe(world);
    drawnEpisodes = -1;
    stale = true;
    anim = null;
    kick();
  }

  function paint(event) {
    const p = Book.pointer(canvas, event);
    const c = Math.floor((p.x - LEFT) / CELL), r = Math.floor((p.y - TOP) / CELL);
    if (c < 0 || c >= N || r < 0 || r >= N) return;
    const s = r * N + c;
    if (s === painted) return;
    painted = s;
    if (tool === 'start') {
      if (world.cells[s] === GOAL) return Book.toast('The start and the goal need different cells');
      world.cells[s] = EMPTY;
      world.start = s;
      newEpisode();
    } else if (tool === 'goal') {
      if (s === world.start) return Book.toast('The start and the goal need different cells');
      world.cells[world.cells.indexOf(GOAL)] = EMPTY;
      world.cells[s] = GOAL;
      newEpisode();
    } else {
      if (s === world.start || world.cells[s] === GOAL) return Book.toast('Use "move start" or "move goal" to shift those');
      world.cells[s] = TOOL_KIND[tool];
      if (s === pos && tool === 'wall') newEpisode();
    }
    last = null;
    refresh();
  }

  function drag(event) {
    if (event.buttons) paint(event); else canvas.onpointermove = null;
  }

  canvas.addEventListener('pointerdown', e => {
    if (tool === 'look') return;
    e.preventDefault();
    painted = -1;
    paint(e);
    canvas.onpointermove = tool === 'start' || tool === 'goal' ? null : drag;
  });
  canvas.addEventListener('pointerup', () => { canvas.onpointermove = null; });
  canvas.addEventListener('pointercancel', () => { canvas.onpointermove = null; });

  $('rl-tools').addEventListener('click', e => {
    const picked = e.target.closest('[data-tool]');
    if (!picked) return;
    tool = picked.dataset.tool;
    $('rl-tools').querySelectorAll('[data-tool]').forEach(t => t.classList.toggle('on', t === picked));
    canvas.style.touchAction = tool === 'look' ? '' : 'none';
    canvas.style.cursor = tool === 'look' ? 'default' : 'crosshair';
  });

  for (const id of ['rl-eps', 'rl-alpha', 'rl-gamma']) $(id).addEventListener('input', readSliders);

  $('rl-step').addEventListener('click', () => {
    setMode('idle');
    advance();
    anim = { start: performance.now() };
    kick();
  });
  $('rl-finish').addEventListener('click', () => {
    setMode('idle');
    finishEpisode();
    refresh();
  });
  $('rl-watch').addEventListener('click', () => setMode(mode === 'watch' ? 'idle' : 'watch'));
  $('rl-fast').addEventListener('click', () => setMode(mode === 'fast' ? 'idle' : 'fast'));
  $('rl-burst').addEventListener('click', () => {
    burstLeft = BURST;
    setMode('burst');
  });
  $('rl-erase').addEventListener('click', () => {
    setMode('idle');
    eraseMemory();
    refresh();
    Book.toast('Memory erased. The maze and the sliders are unchanged.');
  });
  $('rl-reset').addEventListener('click', () => {
    setMode('idle');
    resetEverything();
    refresh();
  });
  $('rl-dump-toggle').addEventListener('click', () => {
    const el = $('rl-dump');
    el.hidden = !el.hidden;
    $('rl-dump-toggle').textContent = el.hidden ? 'show its whole memory' : 'hide its memory';
    renderDump();
  });

  resetEverything();
  render(performance.now());
})();
