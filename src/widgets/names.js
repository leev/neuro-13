(function namesChapter() {
  const heatCanvas = document.getElementById('nc-heat');
  if (!heatCanvas) return;
  const $ = id => document.getElementById(id);

  const V = 27, CTX = 3, D = 2, H = 64, I = CTX * D;
  const STEPS = 20000, BATCH = 32, LR_START = 0.3, LR_END = 0.01;
  const EVAL_EVERY = 400, MAX_LEN = 24, MIN_OWN = 30;
  const SNAP_AT = [50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000];
  const FILM_AT = [100, 400, 2000];
  const VOWELS = new Set([1, 5, 9, 15, 21]);
  const INK = { vowel: '#ea580c', consonant: '#2563eb', end: '#4b5563' };
  const WASH = { vowel: 'rgba(234, 88, 12, 0.16)', consonant: 'rgba(37, 99, 235, 0.12)', end: 'rgba(75, 85, 99, 0.18)' };
  const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';
  const NOT_FOR_KIDS = /fuck|shit|piss|bitch|slut|whore|porn|penis|vagin|nazi|rape|nigg|cunt|cock|dick|puss|anus|tits|sex/;
  const FOODS = 'apple apricot avocado banana blackberry blueberry cantaloupe cherry clementine coconut cranberry currant date dragonfruit elderberry fig gooseberry grape grapefruit guava honeydew kiwi kumquat lemon lime lychee mandarin mango melon mulberry nectarine orange papaya passionfruit peach pear persimmon pineapple plum pomegranate raspberry strawberry tangerine watermelon artichoke asparagus beet broccoli cabbage carrot cauliflower celery chard chickpea corn cucumber eggplant garlic ginger kale leek lentil lettuce mushroom okra olive onion parsley parsnip pea pepper potato pumpkin radish rhubarb rutabaga shallot spinach squash tomato turnip yam zucchini almond cashew hazelnut peanut pecan pistachio walnut barley oat quinoa rice wheat basil cilantro cinnamon mint nutmeg oregano paprika rosemary saffron thyme vanilla noodle pancake waffle pretzel muffin bagel biscuit cracker dumpling taco burrito pizza lasagna spaghetti ravioli sandwich omelet pudding custard brownie cupcake cookie doughnut';

  const glyph = k => (k ? String.fromCharCode(96 + k) : '•');
  const kindOf = k => (k === 0 ? 'end' : VOWELS.has(k) ? 'vowel' : 'consonant');
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const percent = x => (x > 0 && x < 0.005 ? '<1%' : Math.round(x * 100) + '%');
  const LABELS = {
    names: { badge: 'real name', one: 'real name', many: 'real names', things: 'names' },
    dinosaurs: { badge: 'real dinosaur', one: 'real dinosaur', many: 'real dinosaurs', things: 'names' },
    own: { badge: 'from your list', one: 'word from your list', many: 'words from your list', things: 'words' },
  };

  function gauss() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  function encode(words) {
    let n = 0;
    for (const w of words) n += w.length + 1;
    const X = new Uint8Array(n * CTX), Y = new Uint8Array(n);
    let i = 0;
    for (const w of words) {
      let a = 0, b = 0, c = 0;
      for (let k = 0; k <= w.length; k++) {
        const y = k < w.length ? w.charCodeAt(k) - 96 : 0;
        X[i * CTX] = a; X[i * CTX + 1] = b; X[i * CTX + 2] = c; Y[i] = y;
        a = b; b = c; c = y; i++;
      }
    }
    return { X, Y, n };
  }

  function countingLoss(counts, set) {
    let loss = 0;
    for (let e = 0; e < set.n; e++) {
      const row = set.X[e * CTX + 2] * V;
      let total = 0;
      for (let k = 0; k < V; k++) total += counts[row + k];
      loss -= Math.log((counts[row + set.Y[e]] + 1) / (total + V));
    }
    return loss / set.n;
  }

  function prepare(words, label) {
    const nTrain = Math.floor(words.length * 0.9);
    const trainWords = words.slice(0, nTrain), heldWords = words.slice(nTrain);
    const train = encode(trainWords), held = encode(heldWords);
    const counts = new Float64Array(V * V);
    for (let e = 0; e < train.n; e++) counts[train.X[e * CTX + 2] * V + train.Y[e]]++;
    return {
      label, train, held, counts,
      trainSet: new Set(trainWords), heldSet: new Set(heldWords),
      countLoss: countingLoss(counts, held),
    };
  }

  const lists = {};
  function loadList(file, label) {
    if (!lists[file]) {
      lists[file] = fetch('py/' + file)
        .then(res => {
          if (!res.ok) throw new Error('missing');
          return res.text();
        })
        .then(text => {
          const words = text.split(/\s+/).filter(w => /^[a-z]+$/.test(w));
          if (words.length < 100) throw new Error('incomplete');
          return prepare(words, label);
        })
        .catch(() => {
          delete lists[file];
          throw new Error(`Could not load ${file}. If you opened this page as a file, run "make serve" in the book folder and open http://localhost:8000`);
        });
    }
    return lists[file];
  }

  function parseWords(text) {
    const plain = text.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
    const words = [...new Set(plain.split(/[^a-z]+/).filter(w => w.length >= 2 && w.length <= 20))];
    const random = Book.rng(13);
    for (let i = words.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [words[i], words[j]] = [words[j], words[i]];
    }
    return words;
  }

  function createNet() {
    const net = {
      C: new Float64Array(V * D), W1: new Float64Array(H * I), b1: new Float64Array(H),
      W2: new Float64Array(V * H), b2: new Float64Array(V),
      gC: new Float64Array(V * D), gW1: new Float64Array(H * I), gb1: new Float64Array(H),
      gW2: new Float64Array(V * H), gb2: new Float64Array(V),
      x: new Float64Array(I), h: new Float64Array(H), p: new Float64Array(V), dh: new Float64Array(H),
    };
    const fanIn = (5 / 3) / Math.sqrt(I);
    for (let i = 0; i < net.C.length; i++) net.C[i] = gauss() * 0.3;
    for (let i = 0; i < net.W1.length; i++) net.W1[i] = gauss() * fanIn;
    for (let i = 0; i < net.W2.length; i++) net.W2[i] = gauss() * 0.01;
    return net;
  }

  function forward(net, c0, c1, c2, temperature) {
    const { C, W1, b1, W2, b2, x, h, p } = net;
    for (let d = 0; d < D; d++) {
      x[d] = C[c0 * D + d]; x[D + d] = C[c1 * D + d]; x[2 * D + d] = C[c2 * D + d];
    }
    for (let j = 0; j < H; j++) {
      let s = b1[j];
      const o = j * I;
      for (let i = 0; i < I; i++) s += W1[o + i] * x[i];
      h[j] = Math.tanh(s);
    }
    let max = -Infinity;
    for (let k = 0; k < V; k++) {
      let s = b2[k];
      const o = k * H;
      for (let j = 0; j < H; j++) s += W2[o + j] * h[j];
      s /= temperature;
      p[k] = s;
      if (s > max) max = s;
    }
    let sum = 0;
    for (let k = 0; k < V; k++) { p[k] = Math.exp(p[k] - max); sum += p[k]; }
    for (let k = 0; k < V; k++) p[k] /= sum;
  }

  function gradients(net, set, batch) {
    const { W1, W2, gC, gW1, gb1, gW2, gb2, x, h, p, dh } = net;
    const { X, Y } = set;
    gC.fill(0); gW1.fill(0); gb1.fill(0); gW2.fill(0); gb2.fill(0);
    for (let n = 0; n < batch.length; n++) {
      const e = batch[n], c0 = X[e * CTX], c1 = X[e * CTX + 1], c2 = X[e * CTX + 2];
      forward(net, c0, c1, c2, 1);
      p[Y[e]] -= 1;
      dh.fill(0);
      for (let k = 0; k < V; k++) {
        const g = p[k], o = k * H;
        gb2[k] += g;
        for (let j = 0; j < H; j++) { gW2[o + j] += g * h[j]; dh[j] += g * W2[o + j]; }
      }
      for (let j = 0; j < H; j++) {
        const g = dh[j] * (1 - h[j] * h[j]), o = j * I;
        gb1[j] += g;
        for (let d = 0; d < D; d++) {
          gW1[o + d] += g * x[d]; gW1[o + D + d] += g * x[D + d]; gW1[o + 2 * D + d] += g * x[2 * D + d];
          gC[c0 * D + d] += g * W1[o + d]; gC[c1 * D + d] += g * W1[o + D + d]; gC[c2 * D + d] += g * W1[o + 2 * D + d];
        }
      }
    }
  }

  function descend(net, rate) {
    const { C, W1, b1, W2, b2, gC, gW1, gb1, gW2, gb2 } = net;
    for (let i = 0; i < C.length; i++) C[i] -= rate * gC[i];
    for (let i = 0; i < W1.length; i++) W1[i] -= rate * gW1[i];
    for (let i = 0; i < b1.length; i++) b1[i] -= rate * gb1[i];
    for (let i = 0; i < W2.length; i++) W2[i] -= rate * gW2[i];
    for (let i = 0; i < b2.length; i++) b2[i] -= rate * gb2[i];
  }

  function meanLoss(net, set, stride) {
    let loss = 0, n = 0;
    for (let e = 0; e < set.n; e += stride) {
      forward(net, set.X[e * CTX], set.X[e * CTX + 1], set.X[e * CTX + 2], 1);
      loss -= Math.log(net.p[set.Y[e]]);
      n++;
    }
    return loss / n;
  }

  function pick(p) {
    let r = Math.random(), k = 0;
    for (; k < V - 1; k++) { r -= p[k]; if (r < 0) break; }
    return k;
  }

  function contextOf(text) {
    const ids = [0, 0, 0];
    for (const ch of text.slice(-CTX)) { ids.shift(); ids.push(ch.charCodeAt(0) - 96); }
    return ids;
  }

  function writeName(net, temperature, start) {
    let [a, b, c] = contextOf(start), out = start;
    while (out.length < MAX_LEN) {
      forward(net, a, b, c, temperature);
      const k = pick(net.p);
      if (k === 0) break;
      out += glyph(k);
      a = b; b = c; c = k;
    }
    return out;
  }

  function nameFromCounts(counts) {
    let prev = 0, out = '';
    while (out.length < MAX_LEN) {
      let total = 0;
      for (let k = 0; k < V; k++) total += counts[prev * V + k];
      let r = Math.random() * total, k = 0;
      for (; k < V - 1; k++) { r -= counts[prev * V + k]; if (r < 0) break; }
      if (k === 0) break;
      out += glyph(k);
      prev = k;
    }
    return out;
  }

  function kidSafe(make) {
    for (let tries = 0; tries < 20; tries++) {
      const word = make();
      if (word && !NOT_FOR_KIDS.test(word)) return word;
    }
    return 'bleep';
  }

  function verdict(word, data) {
    return data.trainSet.has(word) ? 'seen' : data.heldSet.has(word) ? 'unseen' : 'new';
  }

  function chip(word, data) {
    const v = verdict(word, data);
    const badge = v === 'new' ? 'new' : v === 'seen' ? data.label.badge : data.label.badge + ', never shown to it';
    return `<span class="nm-chip nm-${v}">${word}<i>${badge}</i></span>`;
  }

  function tallyText(words, data) {
    const fresh = words.filter(w => verdict(w, data) === 'new').length;
    const unseen = words.filter(w => verdict(w, data) === 'unseen').length;
    return `<strong>${fresh} of ${words.length}</strong> new (${percent(fresh / words.length)}) · ` +
      `<strong>${words.length - fresh - unseen}</strong> copied from its training list · ` +
      `<strong>${unseen}</strong> ${unseen === 1 ? data.label.one : data.label.many} it was never shown`;
  }

  function drawBars(canvas, p, message) {
    const { ctx, w, h } = Book.hidpi(canvas);
    const left = 8, bottom = 24, top = 20, slot = (w - 2 * left) / V;
    ctx.clearRect(0, 0, w, h);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (!p) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '13px ' + MONO;
      ctx.fillText(message, w / 2, h / 2);
      return;
    }
    let tallest = 0;
    for (let k = 0; k < V; k++) tallest = Math.max(tallest, p[k]);
    const full = Math.max(0.25, tallest);
    for (let n = 0; n < V; n++) {
      const k = (n + 1) % V, x = left + n * slot, kind = kindOf(k);
      const height = (p[k] / full) * (h - top - bottom);
      ctx.fillStyle = WASH[kind];
      ctx.fillRect(x + 2, top, slot - 4, h - top - bottom);
      ctx.fillStyle = INK[kind];
      ctx.fillRect(x + 2, h - bottom - height, slot - 4, height);
      ctx.font = '600 12px ' + MONO;
      ctx.fillText(k ? glyph(k) : 'end', x + slot / 2, h - bottom / 2);
      if (p[k] >= 0.04) {
        ctx.font = '10px ' + MONO;
        ctx.fillStyle = '#374151';
        ctx.fillText(Math.round(p[k] * 100) + '%', x + slot / 2, Math.max(8, h - bottom - height - 8));
      }
    }
  }

  function makeLab(prefix, source) {
    const el = name => $(prefix + '-' + name);
    const trainBtn = el('train'), stopBtn = el('stop');
    const map = el('map') ? Book.hidpi(el('map')) : null;
    const batch = new Int32Array(BATCH);
    const afterTraining = ['ask-input', 'ask-next', 'ask-clear', 'temp', 'gen', 'check'].map(el).filter(Boolean);
    let data = null, net = null, running = false, stopAsked = false, mapRadius = 1, zoomed = false;
    let curve = { train: [], held: [] }, made = [];

    function drawChart() {
      const points = STEPS / EVAL_EVERY + 1;
      const seen = curve.train.concat(curve.held, data ? [data.countLoss] : []);
      const yMin = seen.length ? Math.max(0, Math.floor((Math.min(...seen) - 0.05) / 0.4) * 0.4) : 2;
      const top = Math.min(8, Math.max(3.6, ...curve.held.map(v => v + 0.2)));
      const things = data ? data.label.things : 'names';
      Book.lineChart(el('chart'), {
        series: [
          { label: 'counting table', color: '#9ca3af', data: data ? new Array(points).fill(data.countLoss) : [] },
          { label: `${things} it trains on`, color: '#2563eb', data: curve.train },
          { label: `held-out ${things}`, color: '#ea580c', data: curve.held },
        ],
        xMax: points,
        yMin,
        yMax: yMin + 0.4 * Math.ceil((top - yMin) / 0.4 - 1e-9),
        yFormat: v => v.toFixed(1),
        xLabel: `training steps (0 to ${STEPS.toLocaleString()})`,
      });
    }

    function drawMap() {
      if (!map) return;
      const { ctx, w, h } = map;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(w / 2, 6); ctx.lineTo(w / 2, h - 6);
      ctx.moveTo(6, h / 2); ctx.lineTo(w - 6, h / 2);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      if (!net) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '13px ' + MONO;
        ctx.fillText('press train to scatter the letters', w / 2, h / 2 - 16);
        return;
      }
      for (let i = 0; i < net.C.length; i++) mapRadius = Math.max(mapRadius, Math.abs(net.C[i]) * 1.12);
      ctx.font = '700 13px ' + MONO;
      for (let k = 0; k < V; k++) {
        const { x, y } = spot(k), kind = kindOf(k);
        if (x < 9 || y < 9 || x > w - 9 || y > h - 9) continue;
        ctx.fillStyle = WASH[kind];
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = INK[kind];
        ctx.fillText(glyph(k), x, y + 0.5);
      }
      if (zoomed) {
        ctx.fillStyle = '#6b7280';
        ctx.font = '11px ' + MONO;
        ctx.textAlign = 'left';
        ctx.fillText('middle of the map, 3× closer', 8, h - 10);
      }
    }

    function spot(k) {
      const scale = (map.w / 2 - 14) / (zoomed ? mapRadius / 3 : mapRadius);
      return { x: map.w / 2 + net.C[k * D] * scale, y: map.h / 2 - net.C[k * D + 1] * scale };
    }

    function readMap(event) {
      if (!net) return;
      const at = Book.pointer(el('map'), event);
      let best = -1, bestDist = 18;
      for (let k = 0; k < V; k++) {
        const { x, y } = spot(k), dist = Math.hypot(x - at.x, y - at.y);
        if (dist < bestDist) { bestDist = dist; best = k; }
      }
      if (best < 0) return;
      const name = best ? `the letter “${glyph(best)}”` : 'the end marker';
      el('map-readout').innerHTML = `The embedding row for ${name} is <strong>[${net.C[best * D].toFixed(2)}, ${net.C[best * D + 1].toFixed(2)}]</strong>: ` +
        `${net.C[best * D].toFixed(2)} across, ${net.C[best * D + 1].toFixed(2)} up.`;
    }

    function showLosses(trainLoss, heldLoss) {
      el('loss-train').textContent = trainLoss.toFixed(2);
      el('loss-held').textContent = heldLoss.toFixed(2);
      el('choices').textContent = Math.exp(heldLoss).toFixed(1);
    }

    function measure(exact) {
      const trainLoss = meanLoss(net, data.train, exact ? 1 : Math.ceil(data.train.n / 3000));
      const heldLoss = meanLoss(net, data.held, exact ? 1 : Math.ceil(data.held.n / 3000));
      showLosses(trainLoss, heldLoss);
      return { trainLoss, heldLoss };
    }

    function plot(step, exact) {
      const { trainLoss, heldLoss } = measure(exact);
      const at = step / EVAL_EVERY;
      curve.train[at] = trainLoss;
      curve.held[at] = heldLoss;
      drawChart();
    }

    function snapshot(step, keep) {
      const words = Array.from({ length: 10 }, () => kidSafe(() => writeName(net, 1, '')));
      el('samples').innerHTML = words.map(w => chip(w, data)).join('');
      el('tally').innerHTML = tallyText(words, data);
      if (!keep || !el('film')) return;
      const row = document.createElement('div');
      row.className = 'nm-film-row';
      row.innerHTML = `<span class="nm-film-step">step ${step.toLocaleString()}</span><span class="nm-film-loss">loss ${el('loss-held').textContent}</span>` +
        `<span class="nm-film-names">${words.slice(0, 5).join(' &nbsp; ')}</span>`;
      el('film').appendChild(row);
    }

    function setBusy(busy) {
      running = busy;
      trainBtn.disabled = busy;
      stopBtn.disabled = !busy;
      for (const control of afterTraining) control.disabled = busy || !net;
    }

    function clearOutputs() {
      curve = { train: [], held: [] };
      made = [];
      mapRadius = 1;
      for (const name of ['film', 'gen-out', 'gen-tally', 'check-out', 'check-tally', 'ask-note']) if (el(name)) el(name).innerHTML = '';
    }

    function reset() {
      net = null;
      data = null;
      clearOutputs();
      el('samples').innerHTML = '';
      el('tally').innerHTML = '';
      el('step').textContent = '0';
      for (const name of ['loss-train', 'loss-held', 'choices']) el(name).textContent = '?';
      el('status').textContent = '';
      trainBtn.textContent = 'train the network';
      setBusy(false);
      drawChart();
      drawMap();
      renderAsk();
    }

    async function train() {
      if (running) return;
      el('status').textContent = '';
      try {
        data = await source();
      } catch (err) {
        el('status').textContent = err.message;
        return;
      }
      net = createNet();
      clearOutputs();
      stopAsked = false;
      setBusy(true);
      trainBtn.textContent = '… training';
      let step = 0, nextSnap = 0;
      try {
        plot(0, false);
        snapshot(0, true);
        drawMap();
        await sleep(400);
        while (step < STEPS && !stopAsked) {
          // A background tab gets about one timer tick a second, so each tick has to do more.
          const sliceEnd = performance.now() + (document.hidden ? 400 : 32);
          let pause = 0;
          do {
            for (let n = 0; n < BATCH; n++) batch[n] = Math.floor(Math.random() * data.train.n);
            gradients(net, data.train, batch);
            descend(net, (LR_START + (LR_END - LR_START) * step / STEPS) / BATCH);
            step++;
            if (step % EVAL_EVERY === 0) plot(step, false);
            if (step === SNAP_AT[nextSnap]) {
              const keep = FILM_AT.includes(step);
              if (keep && step % EVAL_EVERY) measure(false);
              snapshot(step, keep);
              nextSnap++;
              pause = step <= 2000 ? 260 : 0;
              break;
            }
          } while (step < STEPS && performance.now() < sliceEnd);
          el('step').textContent = step.toLocaleString();
          drawMap();
          await sleep(pause);
        }
        el('status').textContent = '… checking every name';
        await sleep(0);
        if (step % EVAL_EVERY === 0) plot(step, true); else measure(true);
        snapshot(step, true);
        el('status').textContent = stopAsked ? `Stopped at step ${step.toLocaleString()}. The half-trained network still works: try it below.` : '';
      } finally {
        trainBtn.textContent = 'train again from scratch';
        setBusy(false);
      }
      drawMap();
      renderAsk();
    }

    function temperature() {
      return el('temp') ? +el('temp').value : 1;
    }

    function typed() {
      const box = el('ask-input');
      const clean = box.value.toLowerCase().replace(/[^a-z]/g, '').slice(0, 14);
      if (clean !== box.value) box.value = clean;
      return clean;
    }

    function renderAsk() {
      if (!el('ask-bars')) return;
      el('temp-v').textContent = temperature().toFixed(1);
      const text = typed(), ids = contextOf(text);
      const hidden = text.length > CTX ? `<span class="nm-unseen-letters">${text.slice(0, -CTX)}</span>` : '';
      el('ask-ctx').innerHTML = `<span class="nm-ctx-label">the network sees only:</span> ${hidden}` +
        ids.map(k => `<span class="nm-box${k ? '' : ' nm-blank'}">${glyph(k)}</span>`).join('') + '<span class="nm-box nm-ask">?</span>';
      if (!net) {
        drawBars(el('ask-bars'), null, 'train the network first');
        return;
      }
      forward(net, ids[0], ids[1], ids[2], temperature());
      drawBars(el('ask-bars'), net.p);
    }

    function sampleOne() {
      const text = typed(), ids = contextOf(text);
      forward(net, ids[0], ids[1], ids[2], temperature());
      if (text.length >= 14) {
        el('ask-note').innerHTML = `This box only holds 14 letters, so we stop here: ${chip(text, data)} Press clear to start another.`;
        return;
      }
      const k = pick(net.p), chance = percent(net.p[k]);
      if (k === 0) {
        el('ask-note').innerHTML = text
          ? `It drew <strong>end</strong> (a ${chance} chance), so the name is finished: ${chip(text, data)} Press clear to start another.`
          : `It drew <strong>end</strong> straight away (a ${chance} chance). An empty name. Press it again.`;
        return;
      }
      el('ask-input').value = text + glyph(k);
      el('ask-note').innerHTML = `It drew <strong>${glyph(k)}</strong> (a ${chance} chance) and stuck it on the end. Now it asks again.`;
      renderAsk();
    }

    function inventTen() {
      const start = el('ask-input') ? typed() : '';
      const words = Array.from({ length: 10 }, () => kidSafe(() => writeName(net, temperature(), start)));
      made = made.concat(words);
      el('gen-out').innerHTML = words.map(w => chip(w, data)).join('');
      el('gen-tally').innerHTML = `So far: ${tallyText(made, data)}`;
    }

    function checkMany() {
      const words = Array.from({ length: 200 }, () => kidSafe(() => writeName(net, 1, '')));
      const order = { unseen: 0, seen: 1, new: 2 };
      words.sort((a, b) => order[verdict(a, data)] - order[verdict(b, data)]);
      el('check-out').innerHTML = words.map(w => chip(w, data)).join('');
      el('check-tally').innerHTML = tallyText(words, data);
    }

    trainBtn.addEventListener('click', train);
    stopBtn.addEventListener('click', () => { stopAsked = true; });
    if (map) {
      el('map').addEventListener('pointermove', readMap);
      el('map').addEventListener('click', event => {
        zoomed = !zoomed;
        drawMap();
        readMap(event);
      });
    }
    if (el('ask-bars')) {
      el('ask-input').addEventListener('input', () => { el('ask-note').innerHTML = ''; renderAsk(); });
      el('temp').addEventListener('input', renderAsk);
      el('ask-next').addEventListener('click', sampleOne);
      el('ask-clear').addEventListener('click', () => { el('ask-input').value = ''; el('ask-note').innerHTML = ''; renderAsk(); });
    }
    if (el('gen')) el('gen').addEventListener('click', inventTen);
    if (el('check')) el('check').addEventListener('click', checkMany);
    reset();
    return { reset, busy: () => running };
  }

  function windowMaker() {
    const box = $('nw-input');
    function render() {
      const word = box.value.toLowerCase().replace(/[^a-z]/g, '').slice(0, 12);
      if (word !== box.value) box.value = word;
      const set = encode(word ? [word] : []);
      let html = '';
      for (let e = 0; e < set.n; e++) {
        const cells = [0, 1, 2].map(i => set.X[e * CTX + i]);
        html += '<div class="nm-window">' + cells.map(k => `<span class="nm-box${k ? '' : ' nm-blank'}">${glyph(k)}</span>`).join('') +
          `<span class="nm-arrow">→</span><span class="nm-box nm-answer${set.Y[e] ? '' : ' nm-blank'}">${glyph(set.Y[e])}</span>` +
          `<span class="nm-window-note">${set.Y[e] ? '' : 'the name ends here'}</span></div>`;
      }
      $('nw-rows').innerHTML = html || '<p class="nm-readout">Type a name to slice it up.</p>';
      $('nw-letters').textContent = word.length;
      $('nw-count').textContent = set.n;
    }
    box.addEventListener('input', render);
    render();
  }

  function countingTable(data) {
    const { counts } = data;
    const left = 30, top = 30, cell = 20;
    const totals = new Float64Array(V);
    for (let r = 0; r < V; r++) for (let c = 0; c < V; c++) totals[r] += counts[r * V + c];
    let chosen = { r: 17, c: 21 };

    function draw() {
      const { ctx, w, h } = Book.hidpi(heatCanvas);
      ctx.clearRect(0, 0, w, h);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let k = 0; k < V; k++) {
        const hot = k === chosen.r, hotCol = k === chosen.c;
        ctx.font = (hot ? '700 ' : '500 ') + '12px ' + MONO;
        ctx.fillStyle = hot ? '#7c3aed' : '#6b7280';
        ctx.fillText(glyph(k), left / 2, top + k * cell + cell / 2);
        ctx.font = (hotCol ? '700 ' : '500 ') + '12px ' + MONO;
        ctx.fillStyle = hotCol ? '#7c3aed' : '#6b7280';
        ctx.fillText(glyph(k), left + k * cell + cell / 2, top / 2);
      }
      for (let r = 0; r < V; r++) {
        for (let c = 0; c < V; c++) {
          const share = totals[r] ? counts[r * V + c] / totals[r] : 0;
          ctx.fillStyle = share ? `rgba(109, 40, 217, ${0.05 + 0.95 * Math.pow(share, 0.6)})` : '#f6f5f2';
          ctx.fillRect(left + c * cell + 1, top + r * cell + 1, cell - 2, cell - 2);
        }
      }
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 2;
      ctx.strokeRect(left + chosen.c * cell, top + chosen.r * cell, cell, cell);
    }

    function describe() {
      const { r, c } = chosen, n = counts[r * V + c];
      const where = r ? `After “${glyph(r)}”` : 'At the start of a name';
      const what = c ? `the next letter was “${glyph(c)}”` : 'the name ended';
      const ranked = [...Array(V).keys()].sort((a, b) => counts[r * V + b] - counts[r * V + a]).slice(0, 3).filter(k => counts[r * V + k] > 0);
      const top3 = ranked.map(k => `${k ? '“' + glyph(k) + '”' : 'end'} ${percent(counts[r * V + k] / totals[r])}`).join(', ');
      $('nc-readout').innerHTML = totals[r]
        ? `${where}, ${what} <strong>${n.toLocaleString()}</strong> times out of ${totals[r].toLocaleString()}. That is <strong>${percent(n / totals[r])}</strong>.` +
          `<br><span class="nm-dim">${r ? 'Most likely after “' + glyph(r) + '”' : 'Most likely first letters'}: ${top3}.</span>`
        : `No name in the list contains “${glyph(r)}”.`;
    }

    function choose(event) {
      const at = Book.pointer(heatCanvas, event);
      const r = Math.floor((at.y - top) / cell), c = Math.floor((at.x - left) / cell);
      if (r < 0 || c < 0 || r >= V || c >= V || (r === chosen.r && c === chosen.c)) return;
      chosen = { r, c };
      draw();
      describe();
    }

    heatCanvas.addEventListener('pointermove', choose);
    heatCanvas.addEventListener('pointerdown', choose);
    $('nc-gen').disabled = false;
    $('nc-gen').addEventListener('click', () => {
      $('nc-out').innerHTML = Array.from({ length: 10 }, () => chip(kidSafe(() => nameFromCounts(counts)), data)).join('');
    });
    $('nc-names').textContent = data.trainSet.size.toLocaleString();
    $('nc-pairs').textContent = data.train.n.toLocaleString();
    draw();
    describe();
  }

  function dataSwitcher() {
    const toggles = $('nx-data'), own = $('nx-own');
    let choice = 'names';

    function countOwn() {
      const n = parseWords(own.value).length;
      $('nx-own-count').textContent = n < MIN_OWN ? `${n} words so far. It needs at least ${MIN_OWN}.` : `${n} different words. Ready to train.`;
    }

    const lab = makeLab('nx', async () => {
      if (choice !== 'own') return loadList(choice + '.txt', LABELS[choice]);
      const words = parseWords(own.value);
      if (words.length < MIN_OWN) throw new Error(`That is only ${words.length} different words. Paste at least ${MIN_OWN}; 50 or more works better.`);
      return prepare(words, LABELS.own);
    });

    toggles.addEventListener('click', e => {
      const toggle = e.target.closest('[data-set]');
      if (!toggle || toggle.dataset.set === choice) return;
      if (lab.busy()) { Book.toast('Press stop first'); return; }
      choice = toggle.dataset.set;
      toggles.querySelectorAll('.toggle').forEach(t => t.classList.toggle('on', t === toggle));
      $('nx-own-box').hidden = choice !== 'own';
      lab.reset();
    });
    function ownChanged() {
      Book.store.set('names.own', own.value);
      $('nx-status').textContent = '';
      countOwn();
    }
    own.addEventListener('input', ownChanged);
    $('nx-own-example').addEventListener('click', () => {
      own.value = FOODS.split(' ').join('\n');
      ownChanged();
    });
    own.value = Book.store.get('names.own', '');
    countOwn();
  }

  windowMaker();
  makeLab('nm', () => loadList('names.txt', LABELS.names));
  dataSwitcher();
  loadList('names.txt', LABELS.names).then(countingTable, err => { $('nc-readout').textContent = err.message; });
})();
