(function diffusionWidget() {
  function loadModel(file) {
    const layers = [];
    for (let l = 0; l + 1 < file.sizes.length; l++) {
      layers.push({
        nIn: file.sizes[l],
        nOut: file.sizes[l + 1],
        w: Float64Array.from(file.weights_x1000[l], v => v / 1000),
        b: Float64Array.from(file.biases_x1000[l], v => v / 1000),
      });
    }
    return {
      shapes: file.shapes, levels: file.levels, waves: file.waves, keep: file.keep, noise: file.noise, layers,
      acts: file.sizes.map(n => new Float64Array(n)),
      dots: file.dots_x1000.map(d => Float32Array.from(d, v => v / 1000)),
    };
  }

  function guessStatic(model, x, y, level, label) {
    const { waves, layers, acts } = model;
    const input = acts[0], nw = waves.length;
    input[0] = x;
    input[1] = y;
    for (let f = 0; f < nw; f++) {
      input[2 + f] = Math.sin(x * waves[f]);
      input[2 + nw + f] = Math.sin(y * waves[f]);
      input[2 + 2 * nw + f] = Math.cos(x * waves[f]);
      input[2 + 3 * nw + f] = Math.cos(y * waves[f]);
    }
    const tail = 2 + 4 * nw;
    input[tail] = model.noise[level];
    input[tail + 1] = model.keep[level];
    for (let s = 0; s < model.shapes.length; s++) input[tail + 2 + s] = s === label ? 1 : 0;

    for (let l = 0; l < layers.length; l++) {
      const { nIn, nOut, w, b } = layers[l];
      const src = acts[l], dst = acts[l + 1];
      dst.set(b);
      for (let i = 0; i < nIn; i++) {
        const v = src[i];
        // About three in four hidden neurons output 0 after ReLU; skipping them is most of the speed.
        if (v === 0) continue;
        const row = i * nOut;
        for (let j = 0; j < nOut; j++) dst[j] += v * w[row + j];
      }
      if (l < layers.length - 1) {
        for (let j = 0; j < nOut; j++) if (dst[j] < 0) dst[j] = 0;
      }
    }
    return acts[layers.length];
  }

  function unruinStep(model, pos, from, to, label) {
    const { keep, noise } = model;
    for (let i = 0; i < pos.length; i += 2) {
      const guess = guessStatic(model, pos[i], pos[i + 1], from, label);
      const gx = guess[0], gy = guess[1];
      const cleanX = (pos[i] - noise[from] * gx) / keep[from];
      const cleanY = (pos[i + 1] - noise[from] * gy) / keep[from];
      pos[i] = keep[to] * cleanX + noise[to] * gx;
      pos[i + 1] = keep[to] * cleanY + noise[to] * gy;
    }
  }

  // Node loads this file to check the sampler against diffusion.py; there is no page to wire up.
  if (typeof module === 'object' && module.exports) {
    module.exports = { loadModel, unruinStep };
    return;
  }

  const $ = id => document.getElementById(id);
  if (!$('df-ruin-canvas')) return;

  const REACH = 2.7, PACE = 45, REPLAY_PACE = 18;
  const WARM = 'rgba(255, 176, 92, 0.9)', FAINT = 'rgba(255, 176, 92, 0.5)', COOL = 'rgba(56, 189, 248, 0.62)';
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const calm = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const count = n => n.toLocaleString('en-US');
  let model = null, current = null;

  function staticCloud(dots, random = Math.random) {
    const cloud = new Float64Array(2 * dots);
    for (let i = 0; i < cloud.length; i++) {
      cloud[i] = Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    }
    return cloud;
  }

  function makeScreen(canvas) {
    const { ctx, w, h } = Book.hidpi(canvas);
    const view = { x: 0, y: 0, zoom: 1 };
    function paint(...clouds) {
      const scale = view.zoom * w / (2 * REACH), size = Math.max(1.6, w / 190) * Math.sqrt(view.zoom);
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = '#0b1020';
      ctx.fillRect(0, 0, w, h);
      ctx.globalCompositeOperation = 'lighter';
      for (const [pos, color] of clouds) {
        ctx.fillStyle = color;
        for (let i = 0; i < pos.length; i += 2) {
          ctx.fillRect(w / 2 + (pos[i] - view.x) * scale - size / 2, h / 2 - (pos[i + 1] - view.y) * scale - size / 2, size, size);
        }
      }
    }
    paint.view = view;
    return paint;
  }

  function press(chips, chosen) {
    chips.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b === chosen)));
  }

  function flip(button) {
    const on = button.getAttribute('aria-pressed') !== 'true';
    button.setAttribute('aria-pressed', String(on));
    return on;
  }

  function newTake(start, label, steps) {
    return {
      label,
      levels: Array.from({ length: steps + 1 }, (_, i) => Math.round(model.levels * (1 - i / steps))),
      pos: Float64Array.from(start),
      frames: [Float32Array.from(start)],
      ms: 0,
    };
  }

  async function glide(a, b, ms, run, draw) {
    const mid = new Float32Array(a.length), began = performance.now();
    while (!run.stopped) {
      const t = Math.min(1, (performance.now() - began) / ms), ease = t * t * (3 - 2 * t);
      for (let i = 0; i < a.length; i++) mid[i] = a[i] + (b[i] - a[i]) * ease;
      draw(mid);
      if (t >= 1) return;
      await sleep(16);
    }
  }

  // Plays a take from its first frame, computing any frame it does not have yet.
  // Starting a new take anywhere on the page stops the one in progress.
  async function play(take, onFrame, glideMs = 0) {
    const run = { stopped: false };
    if (current) current.stopped = true;
    current = run;
    onFrame(take.frames[0], 0);
    for (let k = 1; k < take.levels.length; k++) {
      const began = performance.now();
      await sleep(0);
      if (run.stopped) break;
      const fresh = k === take.frames.length;
      if (fresh) {
        const computing = performance.now();
        unruinStep(model, take.pos, take.levels[k - 1], take.levels[k], take.label);
        take.ms += performance.now() - computing;
        take.frames.push(Float32Array.from(take.pos));
      }
      if (glideMs) await glide(take.frames[k - 1], take.frames[k], glideMs, run, mid => onFrame(mid, k - 1));
      else await sleep((fresh ? PACE : REPLAY_PACE) - (performance.now() - began));
      if (run.stopped) break;
      onFrame(take.frames[k], k);
    }
    if (current === run) current = null;
    return !run.stopped;
  }

  function ruinWidget() {
    const paint = makeScreen($('df-ruin-canvas'));
    const slider = $('df-ruin-level'), playBtn = $('df-ruin-play'), chips = $('df-ruin-shapes');
    let shape = Math.max(0, model.shapes.indexOf('heart')), frames = [], timer = null;

    function reroll() {
      frames = [model.dots[shape]];
      for (let s = 1; s <= model.levels; s++) {
        const shrink = model.keep[s] / model.keep[s - 1], kick = Math.sqrt(1 - shrink * shrink);
        const before = frames[s - 1], kicks = staticCloud(before.length / 2);
        frames.push(Float32Array.from(before, (v, i) => shrink * v + kick * kicks[i]));
      }
    }

    function show() {
      const s = +slider.value;
      paint([frames[s], WARM]);
      $('df-ruin-level-v').textContent = s;
      $('df-ruin-keep').textContent = model.keep[s].toFixed(2);
      $('df-ruin-noise').textContent = model.noise[s].toFixed(2);
    }

    function pause() {
      clearInterval(timer);
      timer = null;
      playBtn.textContent = '▶ ruin it';
    }

    playBtn.addEventListener('click', () => {
      if (timer) return pause();
      if (+slider.value >= model.levels) slider.value = 0;
      playBtn.textContent = '❚❚ pause';
      timer = setInterval(() => {
        slider.value = +slider.value + 1;
        show();
        if (+slider.value >= model.levels) pause();
      }, 70);
    });
    $('df-ruin-reset').addEventListener('click', () => {
      pause();
      reroll();
      slider.value = 0;
      show();
    });
    slider.addEventListener('input', () => { pause(); show(); });
    chips.addEventListener('click', e => {
      const chip = e.target.closest('button[data-shape]');
      const picked = chip ? model.shapes.indexOf(chip.dataset.shape) : -1;
      if (picked < 0) return;
      shape = picked;
      press(chips, chip);
      reroll();
      show();
    });

    slider.max = model.levels;
    slider.disabled = playBtn.disabled = $('df-ruin-reset').disabled = false;
    chips.querySelectorAll('button').forEach(b => { b.disabled = false; });
    press(chips, chips.querySelector(`[data-shape="${model.shapes[shape]}"]`));
    reroll();
    show();
  }

  function unruinWidget() {
    const paint = makeScreen($('df-gen-canvas'));
    const slider = $('df-gen-step'), runBtn = $('df-gen-run'), againBtn = $('df-gen-again');
    const trainBtn = $('df-gen-train'), zoomBtn = $('df-gen-zoom'), tag = $('df-gen-tag');
    const label = Math.max(0, model.shapes.indexOf('heart'));
    const training = model.dots[label], dots = training.length / 2;
    const wide = tag.textContent, tip = { x: 0, y: Infinity, zoom: 6 };
    for (let i = 1; i < training.length; i += 2) tip.y = Math.min(tip.y, training[i] + 0.3);
    let take = newTake(staticCloud(dots), label, model.levels), shown = 0;

    function show(frame, k) {
      shown = k;
      if (trainBtn.getAttribute('aria-pressed') === 'true') paint([training, COOL], [frame, WARM]);
      else paint([frame, WARM]);
      slider.value = k;
      $('df-gen-step-v').textContent = k;
      $('df-gen-level').textContent = take.levels[k];
      $('df-gen-runs').textContent = count(k * dots);
    }

    async function go() {
      const mine = take;
      runBtn.disabled = slider.disabled = true;
      await play(mine, show);
      if (mine !== take) return;
      const done = take.frames.length === take.levels.length;
      runBtn.disabled = false;
      runBtn.textContent = done ? '▶ replay' : '▶ un-ruin it';
      slider.disabled = !done;
      if (done) $('df-gen-time').textContent = (take.ms / 1000).toFixed(1) + ' s';
    }

    runBtn.addEventListener('click', go);
    againBtn.addEventListener('click', () => {
      take = newTake(staticCloud(dots), label, model.levels);
      $('df-gen-time').textContent = '—';
      go();
    });
    slider.addEventListener('input', () => {
      const k = Math.min(+slider.value, take.frames.length - 1);
      show(take.frames[k], k);
    });
    trainBtn.addEventListener('click', () => {
      flip(trainBtn);
      show(take.frames[shown], shown);
    });
    zoomBtn.addEventListener('click', () => {
      const close = flip(zoomBtn);
      Object.assign(paint.view, close ? tip : { x: 0, y: 0, zoom: 1 });
      tag.textContent = close ? 'the tip of the heart, 6× closer' : wide;
      show(take.frames[shown], shown);
    });

    slider.max = model.levels;
    runBtn.disabled = againBtn.disabled = trainBtn.disabled = zoomBtn.disabled = false;
    show(take.frames[0], 0);
  }

  function stepsWidget() {
    const paint = makeScreen($('df-steps-canvas'));
    const chips = $('df-steps-chips');
    const label = Math.max(0, model.shapes.indexOf('heart'));
    const dots = model.dots[label].length / 2;
    const start = staticCloud(dots, Book.rng(4)), takes = new Map();

    chips.addEventListener('click', e => {
      const chip = e.target.closest('button[data-steps]');
      if (!chip) return;
      const steps = Math.min(+chip.dataset.steps, model.levels);
      if (!takes.has(steps)) takes.set(steps, newTake(start, label, steps));
      press(chips, chip);
      $('df-steps-n').textContent = steps;
      $('df-steps-runs').textContent = count(steps * dots);
      play(takes.get(steps), frame => paint([frame, WARM]), steps <= 10 && !calm ? 900 / Math.sqrt(steps) : 0);
    });

    chips.querySelectorAll('button').forEach(b => { b.disabled = false; });
    paint([start, WARM]);
  }

  function promptWidget() {
    const paint = makeScreen($('df-prompt-canvas')), paintMini = makeScreen($('df-prompt-static'));
    const chips = $('df-prompt-chips'), said = $('df-prompt-said');
    const dots = model.dots[0].length / 2;
    let start, takes;

    function renew() {
      if (current) current.stopped = true;
      start = staticCloud(dots);
      takes = new Map();
      press(chips, null);
      said.textContent = 'nothing yet';
      paint([start, WARM]);
      paintMini([start, FAINT]);
    }

    chips.addEventListener('click', e => {
      const chip = e.target.closest('button[data-shape]');
      const label = chip ? model.shapes.indexOf(chip.dataset.shape) : -1;
      if (label < 0) return;
      if (!takes.has(label)) takes.set(label, newTake(start, label, model.levels));
      press(chips, chip);
      said.textContent = chip.textContent;
      play(takes.get(label), frame => paint([frame, WARM]));
    });
    $('df-prompt-new').addEventListener('click', renew);

    chips.querySelectorAll('button').forEach(b => { b.disabled = false; });
    $('df-prompt-new').disabled = false;
    renew();
  }

  async function boot() {
    const notes = document.querySelectorAll('.df-status');
    let file;
    try {
      const res = await fetch('py/diffusion-weights.json');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      file = await res.json();
    } catch (err) {
      notes.forEach(el => {
        el.textContent = 'Could not load the trained network (py/diffusion-weights.json). If you opened this page as a file, ' +
          'run "make serve" in the book folder and open http://localhost:8000';
      });
      return;
    }
    model = loadModel(file);
    notes.forEach(el => { el.hidden = true; });
    ruinWidget();
    unruinWidget();
    stepsWidget();
    promptWidget();
  }

  boot();
})();
