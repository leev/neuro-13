(function homePage() {
  const TAU = Math.PI * 2;

  function spiralData(arms, perArm, turns, noise, rand) {
    const n = arms * perArm;
    const xs = new Float64Array(n * 2), labels = new Uint8Array(n);
    for (let c = 0; c < arms; c++) {
      for (let i = 0; i < perArm; i++) {
        const t = i / (perArm - 1);
        const r = 0.14 + 0.86 * t;
        const angle = c * TAU / arms + t * turns * TAU;
        const k = c * perArm + i;
        xs[2 * k] = r * Math.cos(angle) + (rand() - 0.5) * noise;
        xs[2 * k + 1] = r * Math.sin(angle) + (rand() - 0.5) * noise;
        labels[k] = c;
      }
    }
    return { n, arms, xs, labels };
  }

  function makeLearner(data, hidden, rand) {
    const H = hidden, K = data.arms;
    const w1 = 0, b1 = 2 * H, w2 = b1 + H, b2 = w2 + H * H, w3 = b2 + H, b3 = w3 + K * H, count = b3 + K;
    const theta = new Float64Array(count), grad = new Float64Array(count);
    const m = new Float64Array(count), v = new Float64Array(count);
    const h1 = new Float64Array(H), h2 = new Float64Array(H), p = new Float64Array(K);
    const d1 = new Float64Array(H), d2 = new Float64Array(H);
    let t = 0;

    const fill = (from, to, fanIn) => {
      const scale = Math.sqrt(1 / fanIn);
      for (let i = from; i < to; i++) theta[i] = (rand() * 2 - 1) * scale;
    };
    fill(w1, b1, 2);
    fill(w2, b2, H);
    fill(w3, b3, H);

    function predict(x, y) {
      for (let j = 0; j < H; j++) h1[j] = Math.tanh(theta[b1 + j] + theta[w1 + 2 * j] * x + theta[w1 + 2 * j + 1] * y);
      for (let j = 0; j < H; j++) {
        let z = theta[b2 + j];
        const row = w2 + j * H;
        for (let i = 0; i < H; i++) z += theta[row + i] * h1[i];
        h2[j] = Math.tanh(z);
      }
      let top = -Infinity;
      for (let k = 0; k < K; k++) {
        let z = theta[b3 + k];
        const row = w3 + k * H;
        for (let i = 0; i < H; i++) z += theta[row + i] * h2[i];
        p[k] = z;
        if (z > top) top = z;
      }
      let sum = 0;
      for (let k = 0; k < K; k++) { p[k] = Math.exp(p[k] - top); sum += p[k]; }
      for (let k = 0; k < K; k++) p[k] /= sum;
      return p;
    }

    function lossAndGrad() {
      grad.fill(0);
      let loss = 0, right = 0;
      for (let s = 0; s < data.n; s++) {
        const x = data.xs[2 * s], y = data.xs[2 * s + 1], label = data.labels[s];
        predict(x, y);
        loss -= Math.log(Math.max(p[label], 1e-12));
        let best = 0;
        for (let k = 1; k < K; k++) if (p[k] > p[best]) best = k;
        if (best === label) right++;
        d2.fill(0);
        for (let k = 0; k < K; k++) {
          const dz = p[k] - (k === label ? 1 : 0);
          const row = w3 + k * H;
          grad[b3 + k] += dz;
          for (let i = 0; i < H; i++) { grad[row + i] += dz * h2[i]; d2[i] += dz * theta[row + i]; }
        }
        d1.fill(0);
        for (let j = 0; j < H; j++) {
          const dz = d2[j] * (1 - h2[j] * h2[j]);
          const row = w2 + j * H;
          grad[b2 + j] += dz;
          for (let i = 0; i < H; i++) { grad[row + i] += dz * h1[i]; d1[i] += dz * theta[row + i]; }
        }
        for (let j = 0; j < H; j++) {
          const dz = d1[j] * (1 - h1[j] * h1[j]);
          grad[b1 + j] += dz;
          grad[w1 + 2 * j] += dz * x;
          grad[w1 + 2 * j + 1] += dz * y;
        }
      }
      for (let i = 0; i < count; i++) grad[i] /= data.n;
      return { loss: loss / data.n, right };
    }

    function step(rate) {
      const result = lossAndGrad();
      t++;
      const c1 = 1 - Math.pow(0.9, t), c2 = 1 - Math.pow(0.999, t);
      for (let i = 0; i < count; i++) {
        m[i] = 0.9 * m[i] + 0.1 * grad[i];
        v[i] = 0.999 * v[i] + 0.001 * grad[i] * grad[i];
        theta[i] -= rate * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
      }
      return result;
    }

    return { count, theta, grad, predict, lossAndGrad, step };
  }

  function heroArt(host, panel) {
    const PER_ARM = 44, TURNS = 1.2, NOISE = 0.1, HIDDEN = 20, RATE = 0.02;
    const RANGE = 1.2, CAP = 1500, STREAK = 10, STEPS_PER_SECOND = 30, HOLD_MS = 2800, FADE_MS = 450;
    const GRIDS = [72, 56, 40, 28], TRAIN_MS = 6, FRAME_MS = 10, SLOW_PASS_MS = 140;
    const PAPER = '#fffdf8';
    const parts = (window.BOOK && window.BOOK.parts) || [];
    const hexes = parts.length >= 3 && parts.length <= 6
      ? parts.map(part => part.color)
      : ['#2563eb', '#0d9488', '#7c3aed', '#ea580c', '#db2777'];
    const rgb = hexes.map(hex => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)));
    const data = spiralData(hexes.length, PER_ARM, TURNS, NOISE, window.Book ? Book.rng(7) : Math.random);
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const field = document.createElement('canvas');
    const fctx = field.getContext('2d');
    host.setAttribute('role', 'img');
    host.setAttribute('aria-label', `A neural network sorting ${data.n} dots of ${hexes.length} colors: the colored regions it has learned so far swirl into a pinwheel as it trains.`);
    host.appendChild(canvas);

    let image = null, level = 0, grid = 0, row = 0, passStart = 0, slowPasses = 0, fieldStep = -1, passStep = -1;
    let size = 0, scale = 1;
    let net, steps, streak, solvedAt, holdLeft, opacity, fading, shownRight;
    let userPaused = reduced, onScreen = true, raf = 0, last = 0, budget = 0;
    const ui = {};

    function setGrid(n) {
      grid = n;
      image = new ImageData(n, n);
      row = 0;
    }

    function restart() {
      net = makeLearner(data, HIDDEN, Math.random);
      steps = 0;
      streak = 0;
      solvedAt = 0;
      holdLeft = HOLD_MS;
      fading = false;
      opacity = reduced ? 1 : 0;
      shownRight = net.lossAndGrad().right;
      row = 0;
      fieldStep = -1;
    }

    function trainOne() {
      const { right } = net.step(RATE);
      steps++;
      streak = right === data.n ? streak + 1 : 0;
      if (!solvedAt && streak >= STREAK) solvedAt = steps;
      else if (!solvedAt && steps >= CAP) solvedAt = -1;
    }

    function paintRows(deadline) {
      if (!image || (row === 0 && fieldStep === steps)) return false;
      if (row === 0) {
        passStep = steps;
        passStart = performance.now();
      }
      const px = image.data, K = hexes.length;
      while (row < grid) {
        const wy = ((row + 0.5) / grid * 2 - 1) * RANGE;
        for (let gx = 0; gx < grid; gx++) {
          const wx = ((gx + 0.5) / grid * 2 - 1) * RANGE;
          const p = net.predict(wx, wy);
          let top = 0;
          for (let k = 1; k < K; k++) if (p[k] > p[top]) top = k;
          let runnerUp = 0;
          for (let k = 0; k < K; k++) if (k !== top && p[k] > runnerUp) runnerUp = p[k];
          const r = Math.sqrt(wx * wx + wy * wy) / RANGE;
          const e = r <= 0.66 ? 1 : r >= 1 ? 0 : 1 - (r - 0.66) / 0.34;
          const alpha = (0.1 + 0.6 * (p[top] - runnerUp)) * e * e * (3 - 2 * e);
          const o = 4 * (row * grid + gx), c = rgb[top];
          px[o] = c[0];
          px[o + 1] = c[1];
          px[o + 2] = c[2];
          px[o + 3] = Math.round(alpha * 255);
        }
        row++;
        if ((row & 3) === 0 && performance.now() > deadline) break;
      }
      if (row < grid) return false;
      row = 0;
      fieldStep = passStep;
      if (field.width !== grid) field.width = field.height = grid;
      fctx.putImageData(image, 0, 0);
      if (performance.now() - passStart > SLOW_PASS_MS && level < GRIDS.length - 1 && ++slowPasses >= 3) {
        slowPasses = 0;
        setGrid(GRIDS[++level]);
      }
      return true;
    }

    const shown = { round: '', score: '' };

    function readout() {
      if (!ui.round) return;
      const round = solvedAt > 0 ? `Solved in ${solvedAt} rounds` : `Round <span class="num">${steps}</span>`;
      const pct = shownRight === data.n ? 100 : Math.min(99, Math.round(100 * shownRight / data.n));
      const score = `<span class="num pct">${pct}</span>% of dots right`;
      if (shown.round !== round) ui.round.innerHTML = shown.round = round;
      if (shown.score !== score) ui.score.innerHTML = shown.score = score;
      ui.round.classList.toggle('solved', solvedAt > 0);
    }

    function compose() {
      if (!size) return;
      const px = size * scale;
      ctx.clearRect(0, 0, px, px);
      ctx.globalAlpha = opacity;
      if (fieldStep >= 0) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(field, 0, 0, px, px);
      }
      const radius = Math.max(3, size / 125) * scale;
      shownRight = 0;
      for (let i = 0; i < data.n; i++) {
        const x = data.xs[2 * i], y = data.xs[2 * i + 1], label = data.labels[i];
        const p = net.predict(x, y);
        let best = 0;
        for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
        const right = best === label;
        if (right) shownRight++;
        ctx.beginPath();
        ctx.arc((x / RANGE + 1) / 2 * px, (y / RANGE + 1) / 2 * px, radius, 0, TAU);
        ctx.fillStyle = right ? hexes[label] : PAPER;
        ctx.fill();
        ctx.lineWidth = (right ? 1.25 : 2) * scale;
        ctx.strokeStyle = right ? PAPER : hexes[label];
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      readout();
    }

    const running = () => !userPaused && onScreen && !document.hidden;

    function frame(now) {
      raf = 0;
      if (!running()) return;
      const dt = Math.max(0, Math.min(100, now - last));
      last = now;
      const began = performance.now();
      let redraw = false;
      if (fading) {
        opacity -= dt / FADE_MS;
        if (opacity <= 0) restart();
        redraw = true;
      } else {
        if (opacity < 1) {
          opacity = Math.min(1, opacity + dt / FADE_MS);
          redraw = true;
        }
        if (solvedAt) {
          holdLeft -= dt;
          if (holdLeft <= 0) fading = true;
        }
        budget = Math.min(3, budget + dt * STEPS_PER_SECOND / 1000);
        while (budget >= 1) {
          budget--;
          trainOne();
          if (performance.now() - began > TRAIN_MS) break;
        }
      }
      if (paintRows(began + FRAME_MS)) redraw = true;
      if (redraw) compose(); else readout();
      raf = requestAnimationFrame(frame);
    }

    function schedule() {
      if (raf || !running()) return;
      last = passStart = performance.now();
      raf = requestAnimationFrame(frame);
    }

    function still() {
      paintRows(Infinity);
      compose();
    }

    function finishQuietly() {
      let n = 40;
      while (n-- > 0 && !solvedAt) trainOne();
      readout();
      if (solvedAt) still(); else setTimeout(finishQuietly, 0);
    }

    function setPaused(paused) {
      userPaused = paused;
      if (ui.toggle) {
        ui.toggle.textContent = paused ? 'Play' : 'Pause';
        ui.toggle.setAttribute('aria-label', paused ? 'Play the training animation' : 'Pause the training animation');
      }
      schedule();
    }

    function resize() {
      const width = host.clientWidth;
      if (!width || width === size) return;
      size = width;
      scale = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = canvas.height = Math.round(size * scale);
      if (!grid) setGrid(GRIDS[level]);
      if (running()) compose(); else still();
    }

    if (panel) {
      panel.innerHTML = `
        <div class="hero-readout">
          <span class="chip" data-hero="round"></span>
          <span class="chip" data-hero="score"></span>
          <span class="gap"></span>
          <button type="button" class="icon-btn" data-hero="toggle"></button>
          <button type="button" class="icon-btn" data-hero="again" aria-label="Start a new training run from random weights">New run</button>
        </div>
        <p class="hero-caption"><strong>A real neural network, ${reduced ? 'trained a moment ago' : 'learning right now'} in your browser.</strong>
        It starts from random numbers and nudges its <span data-hero="count"></span> weights until every dot sits on its own color.
        ${reduced ? 'Press play to watch it learn.' : 'Then it wipes its memory and starts again.'}</p>`;
      panel.querySelectorAll('[data-hero]').forEach(el => { ui[el.dataset.hero] = el; });
      ui.toggle.addEventListener('click', () => {
        if (userPaused && solvedAt && holdLeft === HOLD_MS) restart();
        setPaused(!userPaused);
      });
      ui.again.addEventListener('click', () => {
        restart();
        if (userPaused) opacity = 1;
        if (userPaused && reduced) finishQuietly(); else still();
      });
    }

    restart();
    if (ui.count) ui.count.textContent = net.count.toLocaleString('en-US');
    setPaused(userPaused);
    new ResizeObserver(resize).observe(host);
    resize();
    if (reduced) finishQuietly();
    new IntersectionObserver(entries => {
      onScreen = entries[entries.length - 1].isIntersecting;
      schedule();
    }).observe(host);
    document.addEventListener('visibilitychange', schedule);
    schedule();
  }

  function chapterLabels() {
    if (!window.Book) return;
    document.querySelectorAll('[data-chapter-label]').forEach(label => {
      const chapter = Book.chapters.find(c => c.slug === label.dataset.chapterLabel);
      const card = label.closest('a');
      if (!chapter) {
        card.closest('li').hidden = true;
        return;
      }
      const part = Book.parts.find(p => p.id === chapter.part);
      label.textContent = `Chapter ${chapter.number}`;
      card.style.setProperty('--part', part.color);
      card.style.setProperty('--part-tint', part.tint);
    });
  }

  function resetControl() {
    const row = document.querySelector('[data-reset-row]');
    if (!row || !window.Book) return;
    const ask = row.querySelector('[data-reset-ask]'), confirm = row.querySelector('[data-reset-confirm]');
    const show = asking => {
      ask.hidden = asking;
      confirm.hidden = !asking;
    };
    ask.addEventListener('click', () => {
      show(true);
      confirm.querySelector('[data-reset-cancel]').focus();
    });
    row.querySelector('[data-reset-cancel]').addEventListener('click', () => {
      show(false);
      ask.focus();
    });
    const sync = () => {
      const any = Book.done.size > 0 || Object.keys(Book.quizzes).length > 0;
      row.hidden = !any;
      if (any) return;
      show(false);
      document.querySelectorAll('[data-continue][data-start-href]').forEach(a => {
        a.href = a.dataset.startHref;
        a.textContent = a.dataset.startLabel;
      });
    };
    document.addEventListener('book:progress', sync);
    sync();
  }

  const host = document.getElementById('hero-art');
  if (host) heroArt(host, document.getElementById('hero-panel'));
  chapterLabels();
  resetControl();
})();
