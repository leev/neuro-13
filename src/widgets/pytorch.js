(function blameFlowWidget() {
  const svg = document.getElementById('ag-graph');
  if (!svg) return;
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const W = 90, H = 72, BADGE = 12, PUSH = 0.001, LEARNING_RATE = 1;
  const calm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const OPS = {
    mul: { sign: '×', run: (a, b) => [a.data * b.data, [b.data, a.data]] },
    add: { sign: '+', run: (a, b) => [a.data + b.data, [1, 1]] },
    squish: { sign: 'squish', run: a => { const s = 1 / (1 + Math.exp(-a.data)); return [s, [s * (1 - s)]]; } },
  };

  const PRESETS = {
    warm: {
      view: '0 0 600 256',
      minWidth: 520,
      controls: [
        { id: 'a', value: 2, min: -5, max: 5, step: 1 },
        { id: 'b', value: -3, min: -5, max: 5, step: 1 },
        { id: 'c', value: 10, min: -10, max: 10, step: 1 },
      ],
      nodes: [
        { id: 'b', at: [52, 62], leaf: s => s.b },
        { id: 'a', at: [52, 208], leaf: s => s.a },
        { id: 'd', at: [216, 136], op: 'mul', from: ['a', 'b'] },
        { id: 'c', at: [216, 44], leaf: s => s.c },
        { id: 'e', at: [384, 84], op: 'add', from: ['d', 'c'] },
        { id: 'f', at: [548, 150], op: 'mul', from: ['e', 'a'] },
      ],
      wiggle: ['a', 'b', 'c'],
      intro: 'Three numbers go in: <b>a</b>, <b>b</b> and <b>c</b>. Nothing has been calculated yet, and every blame starts at 0. Press <b>step</b>.',
      outro: 'That is the whole backward pass: every number now knows its blame. Drag a slider and watch all of it change at once, or check a blame by wiggling below.',
    },
    neuron: {
      view: '0 0 760 306',
      minWidth: 640,
      controls: [
        { id: 'x', value: 2, min: -3, max: 3, step: 0.5 },
        { id: 'w', value: 0.5, min: -3, max: 3, step: 0.1, learned: true },
        { id: 'b', value: -1, min: -3, max: 3, step: 0.1, learned: true },
        { id: 'target', value: 1, min: 0, max: 1, step: 1 },
      ],
      nodes: [
        { id: 'x', at: [48, 72], leaf: s => s.x },
        { id: 'w', at: [48, 204], leaf: s => s.w },
        { id: 'wx', name: 'w × x', at: [181, 138], op: 'mul', from: ['w', 'x'] },
        { id: 'b', at: [181, 258], leaf: s => s.b },
        { id: 'total', at: [314, 198], op: 'add', from: ['wx', 'b'] },
        { id: 'prediction', at: [446, 198], op: 'squish', from: ['total'] },
        { id: 'minus', name: '−target', at: [446, 66], leaf: s => -s.target },
        { id: 'miss', at: [579, 132], op: 'add', from: ['prediction', 'minus'] },
        { id: 'loss', at: [712, 132], op: 'mul', from: ['miss', 'miss'] },
      ],
      wiggle: ['x', 'w', 'b'],
      intro: 'One neuron and its loss, drawn as a graph: an input <b>x</b>, a weight <b>w</b>, a bias <b>b</b>, and a target to compare the prediction with. Press <b>step</b>.',
      outro: 'Every number knows its blame. The blame on <b>w</b> and <b>b</b> says which way to nudge them: press <b>nudge w and b</b> below to take one training step.',
    },
  };

  let presetId = 'warm', preset, inputs, step, graph, playing = 0, frame = 0, nudges = 0, startLoss = null;

  function fmt(v, places = 3) {
    const r = +v.toFixed(places);
    return String(r === 0 ? 0 : r).replace('-', '−');
  }

  function build(values = inputs) {
    const byId = {};
    const nodes = preset.nodes.map(spec => {
      const n = { ...spec, name: spec.name || spec.id, grad: 0, got: false };
      if (spec.leaf) {
        n.data = spec.leaf(values);
        n.parents = [];
        n.slopes = [];
      } else {
        n.parents = spec.from.map(id => byId[id]);
        [n.data, n.slopes] = OPS[spec.op].run(...n.parents);
      }
      byId[spec.id] = n;
      return n;
    });
    const ops = nodes.filter(n => n.parents.length);
    const end = nodes[nodes.length - 1];
    const order = [], seen = new Set();
    (function visit(n) {
      if (seen.has(n)) return;
      seen.add(n);
      n.parents.forEach(visit);
      order.push(n);
    })(end);
    const backOrder = order.reverse().filter(n => n.parents.length);
    const steps = [{ kind: 'start' }, ...ops.map(node => ({ kind: 'forward', node })), { kind: 'seed' },
      ...backOrder.map(node => ({ kind: 'back', node }))];
    return { nodes, byId, end, steps };
  }

  function runTo(index) {
    const g = build();
    const known = new Set(g.nodes.filter(n => !n.parents.length));
    const blamed = new Set();
    let handoffs = [];
    g.steps.slice(0, index + 1).forEach(s => {
      if (s.kind === 'forward') known.add(s.node);
      if (s.kind === 'seed') { g.end.grad = 1; g.end.got = true; }
      if (s.kind === 'back') {
        blamed.add(s.node);
        handoffs = s.node.parents.map((parent, i) => {
          const share = s.node.slopes[i] * s.node.grad;
          const h = { parent, slope: s.node.slopes[i], share, before: parent.grad, hadSome: parent.got };
          parent.grad += share;
          parent.got = true;
          return h;
        });
      }
    });
    return { ...g, known, blamed, handoffs, now: g.steps[index], done: index === g.steps.length - 1 };
  }

  function wire(parent, child, slot, twin) {
    const p0 = [parent.at[0] + W / 2, parent.at[1]], p1 = [child.at[0] - W / 2, child.at[1]];
    const bend = twin ? (slot ? 64 : -64) : 0;
    return { p0, p1, c: [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + bend], below: bend > 0 };
  }

  function along(w, t) {
    const u = 1 - t;
    return [u * u * w.p0[0] + 2 * u * t * w.c[0] + t * t * w.p1[0], u * u * w.p0[1] + 2 * u * t * w.c[1] + t * t * w.p1[1]];
  }

  function draw() {
    const { now } = graph;
    let wires = '', tags = '', boxes = '';
    const live = [];
    graph.nodes.forEach(child => {
      const twin = child.parents.length === 2 && child.parents[0] === child.parents[1];
      child.parents.forEach((parent, i) => {
        const w = wire(parent, child, i, twin);
        const known = graph.known.has(child);
        let state = known ? 'known' : 'pending';
        if (graph.blamed.has(child)) state = 'blamed';
        if (now.node === child) { state = now.kind; live.push(w); }
        wires += `<path class="ag-wire ${state}" d="M${w.p0} Q${w.c} ${w.p1}"/>`;
        if (!known) return;
        const text = fmt(child.slopes[i]), [mx, my] = along(w, 0.5), tw = 6.5 * text.length + 9, ty = my + (w.below ? 13 : -13);
        tags += `<g class="ag-tag ${state}"><rect x="${mx - tw / 2}" y="${ty - 9}" width="${tw}" height="18" rx="9"/>` +
          `<text x="${mx}" y="${ty + 4}">${text}</text></g>`;
      });
    });
    graph.nodes.forEach(n => {
      const [x, y] = n.at, known = graph.known.has(n);
      const shown = known ? fmt(n.data) : '?', mid = x + (n.parents.length ? 5 : 0);
      const size = shown.length > 6 ? 14 : shown.length > 5 ? 16.5 : 19;
      const cls = ['ag-node', n.parents.length ? 'op' : 'leaf', known ? '' : 'unknown', n.got ? 'got' : '',
        now.node === n ? now.kind : '', now.kind === 'seed' && n === graph.end ? 'back' : ''].filter(Boolean).join(' ');
      boxes += `<g class="${cls}" data-node="${n.id}" data-value="${known ? n.data : ''}" data-blame="${n.got ? n.grad : ''}">` +
        `<rect class="ag-box" x="${x - W / 2}" y="${y - H / 2}" width="${W}" height="${H}" rx="10"/>` +
        `<text class="ag-name" x="${mid}" y="${y - 20}">${n.name}</text>` +
        `<text class="ag-value" x="${mid}" y="${y + 4}" style="font-size:${size}px">${shown}</text>` +
        `<line class="ag-rule" x1="${x - W / 2}" y1="${y + 13}" x2="${x + W / 2}" y2="${y + 13}"/>` +
        `<text class="ag-blame-label" x="${x - W / 2 + 7}" y="${y + 28}">blame</text>` +
        `<text class="ag-blame" x="${x + W / 2 - 7}" y="${y + 29}">${fmt(n.grad)}</text>`;
      if (n.parents.length) {
        const sign = OPS[n.op].sign, bx = x - W / 2;
        boxes += sign === 'squish'
          ? `<circle class="ag-badge" cx="${bx}" cy="${y}" r="${BADGE}"/><path class="ag-curve" d="M${bx - 7} ${y + 5} C${bx - 1} ${y + 5} ${bx + 1} ${y - 5} ${bx + 7} ${y - 5}"/>`
          : `<circle class="ag-badge" cx="${bx}" cy="${y}" r="${BADGE}"/><text class="ag-sign" x="${bx}" y="${y + 5}">${sign}</text>`;
      }
      boxes += '</g>';
    });
    cancelAnimationFrame(frame);
    svg.innerHTML = wires + boxes + tags;
    return live;
  }

  function pulse(live, backward) {
    if (calm || !live.length) return;
    const dots = live.map(() => {
      const dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('r', 6);
      dot.setAttribute('class', backward ? 'ag-dot back' : 'ag-dot');
      svg.appendChild(dot);
      return dot;
    });
    const began = performance.now();
    (function tick(time) {
      const t = Math.min(1, Math.max(0, (time - began) / 650));
      live.forEach((w, i) => {
        const [x, y] = along(w, backward ? 1 - t : t);
        dots[i].setAttribute('cx', x);
        dots[i].setAttribute('cy', y);
      });
      if (t < 1) frame = requestAnimationFrame(tick);
      else dots.forEach(dot => dot.remove());
    })(began);
  }

  function follow() {
    const box = svg.parentNode, node = graph.now.node || (graph.now.kind === 'seed' ? graph.end : null);
    if (!box || box.scrollWidth <= box.clientWidth) return;
    const scale = box.scrollWidth / +preset.view.split(' ')[2];
    const xs = node ? [node, ...node.parents].map(n => n.at[0]) : [0];
    const lo = (Math.min(...xs) - W / 2) * scale - 8, hi = (Math.max(...xs) + W / 2) * scale + 8;
    const fits = hi - lo <= box.clientWidth;
    const left = !node ? 0 : fits ? (lo + hi - box.clientWidth) / 2 : (node.at[0] + W / 2) * scale + 8 - box.clientWidth;
    box.scrollTo({ left, behavior: calm ? 'auto' : 'smooth' });
  }

  function forwardStory(n) {
    const [p, q] = n.parents, name = `<b>${n.name}</b>`;
    if (n.op === 'squish') {
      return `<b>Forward.</b> ${name} = squish(${p.name}) = squish(${fmt(p.data)}) = <b>${fmt(n.data)}</b>. ` +
        `The note on the wire is how steep the squish curve is right here: ${fmt(n.data)} × (1 − ${fmt(n.data)}) = ${fmt(n.slopes[0])}.`;
    }
    const recipe = `${p.name} ${OPS[n.op].sign} ${q.name}`;
    const sum = `<b>Forward.</b> ${recipe === n.name ? name : `${name} = ${recipe}`} = ${fmt(p.data)} ${OPS[n.op].sign} ${fmt(q.data)} = <b>${fmt(n.data)}</b>. `;
    if (n.op === 'add') return sum + 'Adding passes any change straight through, so both wires get the note “slope 1”.';
    if (p === q) return sum + `Both wires come from ${p.name}, and each gets the note “slope ${fmt(p.data)}”: the value of the <em>other</em> copy.`;
    return sum + `Notes on the wires: if ${p.name} moves a tiny bit, ${n.name} moves ${fmt(n.slopes[0])} times as much (that is ${q.name}’s value). ` +
      `If ${q.name} moves, ${n.name} moves ${fmt(n.slopes[1])} times as much (${p.name}’s value).`;
  }

  function backStory(n) {
    const [first, second] = graph.handoffs;
    if (second && first.parent === second.parent) {
      return `<b>Backward.</b> <b>${n.name}</b> has blame ${fmt(n.grad)} and hands it back along each wire, multiplied by the wire’s slope. ` +
        `Both wires lead to ${first.parent.name}, so it is paid twice: ${fmt(first.slope)} × ${fmt(n.grad)} = ${fmt(first.share)} along one wire, ` +
        `plus ${fmt(second.slope)} × ${fmt(n.grad)} = ${fmt(second.share)} along the other. Total: <b>${fmt(first.share + second.share)}</b>.`;
    }
    const parts = graph.handoffs.map(h => {
      const share = `${h.parent.name} gets ${fmt(h.slope)} × ${fmt(n.grad)} = <b>${fmt(h.share)}</b>`;
      return h.hadSome ? `${share}, added to the ${fmt(h.before)} it already had: <b>${fmt(h.before + h.share)}</b>` : share;
    });
    return `<b>Backward.</b> <b>${n.name}</b> has blame ${fmt(n.grad)} and hands it back along each wire, multiplied by the wire’s slope. ${parts.join('. ')}.`;
  }

  function story() {
    const { now, end, done } = graph;
    if (now.kind === 'start') return preset.intro;
    if (now.kind === 'forward') {
      return forwardStory(now.node) + (now.node === end ? ' The forward pass is finished, and every wire carries its note.' : '');
    }
    if (now.kind === 'seed') {
      return `<b>Backward.</b> Start at the end. <b>${end.name}</b> gets blame <b>1</b>: move ${end.name} by a tiny bit and ${end.name} moves by exactly that bit.`;
    }
    return backStory(now.node) + (done ? ' ' + preset.outro : '');
  }

  function renderControls() {
    $('ag-inputs').innerHTML = preset.controls.map(c =>
      `<div class="slider-row"><label for="ag-in-${c.id}">${c.id}${c.learned ? ' (a weight)' : ''}</label>` +
      `<input type="range" id="ag-in-${c.id}" data-input="${c.id}" min="${c.min}" max="${c.max}" step="${c.step}" value="${inputs[c.id]}">` +
      `<span class="value" id="ag-val-${c.id}">${fmt(inputs[c.id])}</span></div>`).join('');
    $('ag-wiggle-buttons').innerHTML = preset.wiggle.map(id =>
      `<button type="button" class="secondary" data-wiggle="${id}">wiggle ${id}</button>`).join('');
  }

  function render(moved) {
    graph = runTo(step);
    const live = draw();
    if (moved) {
      pulse(live, graph.now.kind === 'back');
      follow();
    }
    const { now, steps, done, end } = graph;
    const forwardCount = steps.filter(s => s.kind === 'forward').length;
    $('ag-caption').innerHTML = story();
    $('ag-progress').textContent = now.kind === 'start' ? 'ready'
      : now.kind === 'forward' ? `forward pass, step ${step} of ${forwardCount}`
      : `backward pass, step ${step - forwardCount} of ${steps.length - 1 - forwardCount}`;
    $('ag-step').disabled = done;
    $('ag-back').disabled = step === 0;
    $('ag-play').textContent = playing ? '❚❚ pause' : '▶ play';
    $('ag-play').disabled = done && !playing;
    preset.controls.forEach(c => { $('ag-val-' + c.id).textContent = fmt(inputs[c.id]); });
    $('ag-wiggle-buttons').querySelectorAll('button').forEach(b => { b.disabled = !done; });
    if (!done) $('ag-wiggle-out').textContent = 'Finish the backward pass, then pick a number to wiggle.';
    $('ag-train').hidden = presetId !== 'neuron';
    $('ag-nudge').disabled = !done;
    if (presetId === 'neuron') {
      $('ag-nudges').textContent = nudges;
      $('ag-loss').textContent = fmt(end.data, 4);
      $('ag-loss-start').textContent = startLoss === null ? fmt(end.data, 4) : fmt(startLoss, 4);
    }
  }

  function stop() {
    clearInterval(playing);
    playing = 0;
  }

  function go(to) {
    step = Math.max(0, Math.min(graph.steps.length - 1, to));
    if (step === graph.steps.length - 1) stop();
    render(true);
  }

  function choose(id) {
    stop();
    presetId = id;
    preset = PRESETS[id];
    inputs = Object.fromEntries(preset.controls.map(c => [c.id, c.value]));
    step = 0;
    nudges = 0;
    startLoss = null;
    $('ag-presets').querySelectorAll('[data-preset]').forEach(t => {
      t.classList.toggle('on', t.dataset.preset === id);
      t.setAttribute('aria-pressed', String(t.dataset.preset === id));
    });
    svg.setAttribute('viewBox', preset.view);
    svg.style.minWidth = preset.minWidth + 'px';
    if (svg.parentNode) svg.parentNode.scrollLeft = 0;
    renderControls();
    $('ag-wiggle-out').textContent = '';
    render(false);
  }

  $('ag-step').addEventListener('click', () => { stop(); go(step + 1); });
  $('ag-back').addEventListener('click', () => { stop(); go(step - 1); });
  $('ag-reset').addEventListener('click', () => { stop(); go(0); });
  $('ag-play').addEventListener('click', () => {
    if (playing) { stop(); render(false); return; }
    playing = setInterval(() => go(step + 1), 1700);
    go(step + 1);
  });

  $('ag-presets').addEventListener('click', e => {
    const t = e.target.closest('[data-preset]');
    if (t) choose(t.dataset.preset);
  });

  $('ag-inputs').addEventListener('input', e => {
    const id = e.target.dataset.input;
    if (!id) return;
    inputs[id] = +e.target.value;
    nudges = 0;
    startLoss = null;
    $('ag-wiggle-out').textContent = '';
    render(false);
  });

  $('ag-wiggle-buttons').addEventListener('click', e => {
    const btn = e.target.closest('[data-wiggle]');
    if (!btn || !graph.done) return;
    const id = btn.dataset.wiggle, end = graph.end;
    const pushed = build({ ...inputs, [id]: inputs[id] + PUSH }).end.data;
    const moved = pushed - end.data;
    $('ag-wiggle-out').innerHTML =
      `Push <b>${id}</b> from ${fmt(inputs[id], 7)} to ${fmt(inputs[id] + PUSH, 7)} and redo the forward pass: ` +
      `${end.name} goes from ${fmt(end.data, 7)} to ${fmt(pushed, 7)}. It moved ${fmt(moved, 7)}, and ` +
      `${fmt(moved, 7)} ÷ ${PUSH} = <b>${fmt(moved / PUSH, 4)}</b>. The backward pass said <b>${fmt(graph.byId[id].grad, 4)}</b>.`;
  });

  $('ag-nudge').addEventListener('click', () => {
    if (!graph.done) return;
    if (startLoss === null) startLoss = graph.end.data;
    const before = graph.end.data;
    const moves = preset.controls.filter(c => c.learned).map(c => {
      const old = inputs[c.id];
      inputs[c.id] = old - LEARNING_RATE * graph.byId[c.id].grad;
      $('ag-in-' + c.id).value = inputs[c.id];
      return `${c.id} ${fmt(old)} → ${fmt(inputs[c.id])}`;
    });
    nudges++;
    $('ag-wiggle-out').textContent = '';
    render(false);
    $('ag-caption').innerHTML = `<b>Nudged</b> each weight against its blame (new = old − ${LEARNING_RATE} × blame): ${moves.join(', ')}. ` +
      `Then the forward and backward passes ran again with the new numbers. Loss: ${fmt(before, 4)} → <b>${fmt(graph.end.data, 4)}</b>. Keep pressing.`;
  });

  choose('warm');
})();
