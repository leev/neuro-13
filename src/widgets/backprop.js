(function backpropChapter() {
  const EXAMPLES = [
    { x: [0, 0], y: 0 },
    { x: [1, 0], y: 1 },
    { x: [0, 1], y: 1 },
    { x: [1, 1], y: 0 },
  ];
  const START = { lh1: 0.8, rh1: -0.7, bh1: -0.3, lh2: -0.4, rh2: 0.9, bh2: 0.2, h1o: 0.5, h2o: -0.6, bo: -0.1 };
  const KEYS = Object.keys(START);
  const NAMES = {
    lh1: 'left→H1', rh1: 'right→H1', bh1: 'bias H1',
    lh2: 'left→H2', rh2: 'right→H2', bh2: 'bias H2',
    h1o: 'H1→out', h2o: 'H2→out', bo: 'bias out',
  };
  const LEARNING_RATE = 2;

  function forward(net, x) {
    const t1 = x[0] * net.lh1 + x[1] * net.rh1 + net.bh1;
    const h1 = sigmoid(t1);
    const t2 = x[0] * net.lh2 + x[1] * net.rh2 + net.bh2;
    const h2 = sigmoid(t2);
    const tOut = h1 * net.h1o + h2 * net.h2o + net.bo;
    return { t1, h1, t2, h2, tOut, out: sigmoid(tOut) };
  }

  function backward(net, x, target) {
    const f = forward(net, x);
    const miss = f.out - target;
    const slopeOut = f.out * (1 - f.out);
    const atOut = miss * slopeOut;
    const slope1 = f.h1 * (1 - f.h1);
    const at1 = atOut * net.h1o * slope1;
    const slope2 = f.h2 * (1 - f.h2);
    const at2 = atOut * net.h2o * slope2;
    return {
      ...f, miss, loss: 0.5 * miss * miss, slopeOut, atOut, slope1, at1, slope2, at2,
      blame: {
        lh1: at1 * x[0], rh1: at1 * x[1], bh1: at1,
        lh2: at2 * x[0], rh2: at2 * x[1], bh2: at2,
        h1o: atOut * f.h1, h2o: atOut * f.h2, bo: atOut,
      },
    };
  }

  function nudge(net, blame) {
    const next = {};
    for (const k of KEYS) next[k] = net[k] - LEARNING_RATE * blame[k];
    return next;
  }

  function lossOn(net, example) {
    const miss = forward(net, example.x).out - example.y;
    return 0.5 * miss * miss;
  }

  function averageLoss(net) {
    let total = 0;
    for (const example of EXAMPLES) total += lossOn(net, example);
    return total / EXAMPLES.length;
  }

  function fmt(v, places = 3) {
    const s = v.toFixed(places);
    return Number(s) === 0 ? s.replace('-', '') : s;
  }

  const count = n => n.toLocaleString('en-US');

  function par(v, places = 3) {
    const s = fmt(v, places);
    return s[0] === '-' ? '(' + s + ')' : s;
  }

  const VIEW_W = 716;
  const NODES = {
    left: { x: 70, y: 112, r: 24, name: 'left' },
    right: { x: 70, y: 268, r: 24, name: 'right' },
    h1: { x: 330, y: 112, r: 30, name: 'H1', bias: 'bh1' },
    h2: { x: 330, y: 268, r: 30, name: 'H2', bias: 'bh2' },
    out: { x: 560, y: 190, r: 30, name: 'out', bias: 'bo' },
    loss: { x: 612, y: 190 },
  };
  const WIRES = {
    lh1: ['left', 'h1', 0.5], rh1: ['right', 'h1', 0.3],
    lh2: ['left', 'h2', 0.3], rh2: ['right', 'h2', 0.5],
    h1o: ['h1', 'out', 0.45], h2o: ['h2', 'out', 0.45],
  };

  function pill(cx, cy, text, cls, key) {
    const w = text.length * (cls.includes('bp-blame') ? 6.7 : 7.5) + 12;
    return `<g class="${cls}"${key ? ` data-key="${key}"` : ''}>` +
      `<rect x="${(cx - w / 2).toFixed(1)}" y="${cy - 9.5}" width="${w.toFixed(1)}" height="19" rx="6"/>` +
      `<text x="${cx.toFixed(1)}" y="${cy + 4.3}" text-anchor="middle">${text}</text></g>`;
  }

  function drawNet(v) {
    const hot = new Set(v.hot || []);
    const fresh = new Set(v.fresh || []);
    const changed = new Set(v.changed || []);
    const blame = v.blame || {};
    const blameAt = v.blameAt || {};
    const tone = v.tone || 'fwd';
    const focus = hot.size > 0;
    const state = key => [
      v.net[key] >= 0 ? 'pos' : 'neg',
      hot.has(key) ? 'hot ' + tone : '',
      changed.has(key) ? 'changed' : '',
      v.pick === key ? 'picked' : '',
      v.pickable ? 'pickable' : '',
    ].join(' ');
    let wires = '', halos = '', flows = '', nodes = '', labels = '';

    for (const [key, [from, to, t]] of Object.entries(WIRES)) {
      const a = NODES[from], b = NODES[to], w = v.net[key];
      const width = (1 + Math.min(Math.abs(w), 8) * 0.9).toFixed(2);
      const line = `x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"`;
      if (hot.has(key) || v.pick === key) halos += `<line class="bp-halo ${v.pick === key ? 'picked' : tone}" ${line} stroke-width="${+width + 9}"/>`;
      wires += `<line class="bp-wire ${w >= 0 ? 'pos' : 'neg'} ${hot.has(key) || v.pick === key ? 'hot' : focus ? 'dim' : ''}" ${line} stroke-width="${width}"/>`;
      const lx = a.x + (b.x - a.x) * t, ly = a.y + (b.y - a.y) * t;
      labels += pill(lx, ly, fmt(w), 'bp-pill ' + state(key), key);
      if (blame[key] !== undefined) labels += pill(lx, ly + 20, 'blame ' + fmt(blame[key], 4), 'bp-blame' + (fresh.has(key) ? ' fresh' : ''));
    }

    for (const [from, to] of v.flows || []) {
      const a = NODES[from], b = NODES[to];
      const line = `x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"`;
      flows += `<line class="bp-flow under" ${line}/><line class="bp-flow ${tone}" ${line}/>`;
    }

    for (const [id, n] of Object.entries(NODES)) {
      if (id === 'loss') continue;
      const value = v.values[id];
      const known = value !== undefined;
      const tint = known ? (0.05 + 0.5 * value).toFixed(2) : 0;
      nodes += `<g class="bp-node ${hot.has(id) ? 'hot ' + tone : ''}">` +
        `<circle class="ring" cx="${n.x}" cy="${n.y}" r="${n.r + 6}"/>` +
        `<circle class="body" cx="${n.x}" cy="${n.y}" r="${n.r}"/>` +
        `<circle cx="${n.x}" cy="${n.y}" r="${n.r - 1}" fill="rgb(13, 148, 136)" fill-opacity="${tint}"/>`;
      if (n.bias) {
        nodes += `<text class="bp-name" x="${n.x}" y="${n.y - 7}" text-anchor="middle">${n.name}</text>` +
          `<text class="bp-value${known ? '' : ' unknown'}" x="${n.x}" y="${n.y + 12}" text-anchor="middle">${known ? fmt(value) : '?'}</text>`;
      } else {
        nodes += `<text class="bp-name" x="${n.x}" y="${n.y - n.r - 9}" text-anchor="middle">${n.name}</text>` +
          `<text class="bp-value big" x="${n.x}" y="${n.y + 6}" text-anchor="middle">${known ? value : '?'}</text>`;
      }
      nodes += '</g>';
      if (n.bias) {
        labels += pill(n.x, n.y - n.r - 15, 'bias ' + fmt(v.net[n.bias]), 'bp-pill ' + state(n.bias), n.bias);
        if (blame[n.bias] !== undefined) labels += pill(n.x, n.y - n.r - 35, 'blame ' + fmt(blame[n.bias], 4), 'bp-blame' + (fresh.has(n.bias) ? ' fresh' : ''));
        if (blameAt[id] !== undefined) labels += pill(n.x, n.y + n.r + 15, 'blame ' + fmt(blameAt[id], 4), 'bp-blame at' + (fresh.has(id) ? ' fresh' : ''));
      }
    }

    const scored = v.miss !== undefined;
    const lossBox = `<g class="bp-lossbox ${hot.has('loss') ? 'hot ' + tone : ''}">` +
      `<line x1="590" y1="190" x2="612" y2="190"/><rect x="612" y="152" width="96" height="76" rx="8"/>` +
      `<text x="660" y="173" text-anchor="middle">target ${v.target}</text>` +
      `<text x="660" y="194" text-anchor="middle" class="${scored ? '' : 'unknown'}">miss ${scored ? fmt(v.miss) : '?'}</text>` +
      `<text x="660" y="215" text-anchor="middle" class="${scored ? 'strong' : 'unknown'}">loss ${scored ? fmt(v.loss, 4) : '?'}</text></g>`;

    const captions = [[70, 'inputs'], [330, 'hidden layer'], [560, 'output'], [660, 'the mistake']]
      .map(([x, text]) => `<text class="bp-caption" x="${x}" y="352" text-anchor="middle">${text}</text>`).join('');

    return `<svg viewBox="0 0 ${VIEW_W} 364" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${v.label}">` +
      halos + wires + flows + lossBox + nodes + labels + captions + '</svg>';
  }

  function bringIntoView(diagram, ids) {
    const room = diagram.scrollWidth - diagram.clientWidth;
    const xs = (ids || []).map(id => (NODES[id] || NODES[(WIRES[id] || [])[1]] || {}).x).filter(x => x !== undefined);
    if (room <= 0 || !xs.length) return;
    const center = xs.reduce((a, b) => a + b, 0) / xs.length / VIEW_W * diagram.scrollWidth;
    diagram.scrollTo({ left: Math.max(0, Math.min(room, center - diagram.clientWidth / 2)), behavior: 'smooth' });
  }

  function wireToggles(row, onPick) {
    row.addEventListener('click', e => {
      const tog = e.target.closest('.toggle');
      if (tog) onPick(tog);
    });
    row.addEventListener('keydown', e => {
      const tog = e.target.closest('.toggle');
      if (!tog || (e.key !== 'Enter' && e.key !== ' ')) return;
      e.preventDefault();
      onPick(tog);
    });
  }

  (function stepWidget() {
    const diagram = document.getElementById('bp-diagram');
    if (!diagram) return;
    const $ = id => document.getElementById(id);
    const LAST = 11, MORE = 2000, TRAIN_MS = 3500;
    const STAGE_OF = [null, 'forward', 'forward', 'forward', 'loss', 'blame', 'blame', 'blame', 'blame', 'blame', 'blame', 'nudge'];
    let net, example = 1, step = 0, undo = null, rounds = 0, history = [], span = 0, unlocked = false, busy = false, trained = false;

    const math = lines => `<pre class="bp-math">${lines.join('\n')}</pre>`;
    const b = text => `<b>${text}</b>`;
    const o = text => `<b class="bp-o">${text}</b>`;
    const column = rows => {
      const wide = Math.max(...rows.map(r => r.label.length));
      return rows.map(r => r.label.padEnd(wide + 2) + (r.wrap || (s => s))(r.value.padStart(8)) + (r.extra ? '   ' + r.extra : ''));
    };

    function handBack(r, n, id) {
      const hidden = id === 'h1' ? 'H1' : 'H2';
      const weight = id === 'h1' ? n.h1o : n.h2o;
      const act = id === 'h1' ? r.h1 : r.h2;
      const slope = id === 'h1' ? r.slope1 : r.slope2;
      const at = id === 'h1' ? r.at1 : r.at2;
      const notes = [];
      if (weight < 0 && at !== 0) notes.push(`The wire ${hidden}→out has a negative weight, so the blame flips sign on its way back: whatever out wants more of, ${hidden} should send less of.`);
      const ratio = at === 0 ? 0 : Math.abs(r.atOut / at);
      if (ratio >= 1.5) {
        const why = Math.abs(weight) < 1 ? 'it was multiplied by two numbers smaller than 1' : `the slope of ${hidden}'s squish is only ${fmt(slope)}`;
        notes.push(`Notice the size too: about ${count(Math.round(ratio))} times smaller than the blame at out, because ${why}.`);
      } else if (ratio > 0 && ratio <= 0.67) notes.push('Here the blame grew on the way back, because the weight on this wire is large.');
      return {
        title: `Backward: hand the blame back to ${hidden}`,
        lead: `${hidden} never touches the loss. It only talks to out, along one wire. Rule two: a neuron's blame is the blame arriving, times the weight of the wire it came back along, times the slope of the neuron's own squish.`,
        body: math(column([
          { label: '  blame arriving from out', value: fmt(r.atOut, 4) },
          { label: `× weight on the wire ${hidden}→out`, value: fmt(weight) },
          { label: `× slope of ${hidden}'s squish`, value: fmt(slope), extra: `= ${fmt(act)} × (1 - ${fmt(act)})` },
          { label: `= blame at ${hidden}`, value: fmt(at, 4), wrap: o },
        ])),
        note: notes.join(' '),
        hot: [id, 'out', id + 'o'], flows: [['out', id]], fresh: [id],
      };
    }

    function intoHidden(r, n, x, id) {
      const hidden = id === 'h1' ? 'H1' : 'H2';
      const at = id === 'h1' ? r.at1 : r.at2;
      const keys = id === 'h1' ? ['lh1', 'rh1', 'bh1'] : ['lh2', 'rh2', 'bh2'];
      let note;
      if (!x[0] && !x[1]) note = 'Both buttons were 0, so neither wire carried anything and neither gets any blame. Only the bias, which always takes part, is left holding it.';
      else if (x[0] && x[1]) note = 'Both buttons were pressed, so both wires carried a signal and both take a share.';
      else note = `The ${x[0] ? 'right' : 'left'} button was 0, so its wire carried nothing and gets no blame at all. You cannot blame a wire that took no part.`;
      return {
        title: `Backward: blame for the three numbers feeding ${hidden}`,
        lead: 'Rule one again: blame at the neuron, times the signal the wire carried.',
        body: math([
          `${(NAMES[keys[0]]).padEnd(9)}= blame at ${hidden} × left  = ${fmt(at, 4)} × ${x[0]} = ${o(fmt(r.blame[keys[0]], 4))}`,
          `${(NAMES[keys[1]]).padEnd(9)}= blame at ${hidden} × right = ${fmt(at, 4)} × ${x[1]} = ${o(fmt(r.blame[keys[1]], 4))}`,
          `${(NAMES[keys[2]]).padEnd(9)}= blame at ${hidden} × 1     = ${o(fmt(r.blame[keys[2]], 4))}`,
        ]),
        note,
        hot: [id, 'left', 'right', ...keys], fresh: keys,
      };
    }

    function describe() {
      const e = EXAMPLES[example], x = e.x;
      const n = step === LAST ? undo.net : net;
      const r = backward(n, x, e.y);
      switch (step) {
        case 0: {
          let lead;
          if (busy) lead = 'Each round takes the next example in the list and does exactly what you did by hand: forward, loss, blame backwards, nudge. Watch the wires change.';
          else if (trained) {
            const right = EXAMPLES.filter(ex => (forward(net, ex.x).out > 0.5) === (ex.y > 0.5)).length;
            lead = `${count(rounds)} rounds done, and ${right === 4 ? 'all four rows now come out on the right side of 0.5' : right + ' of the 4 rows come out right'}. Pick any example and step through the trained network: look at how small the miss, and so every blame, has become.`;
          } else if (rounds) lead = `${rounds === 1 ? 'One round' : count(rounds) + ' rounds'} done. The nudged network is ready to go again. Pick an example (a different one is more interesting) and step through, or let the computer take over.`;
          else lead = 'Nine numbers: a weight on each of the six wires and a bias on each of the three neurons. They are hand-picked starting values, so the network knows nothing yet. Pick an example above, then press <b>next step</b>.' + (unlocked ? '' : ' The train button unlocks once you have finished one round by hand.');
          return { title: busy ? 'Training…' : trained ? 'Trained' : rounds ? 'Ready for another round' : 'A fresh network and one example', lead };
        }
        case 1: return {
          title: 'Forward: H1 adds up its inputs',
          body: math([
            'total  = (left × weight) + (right × weight) + bias',
            `       = (${x[0]} × ${fmt(n.lh1)}) + (${x[1]} × ${fmt(n.rh1)}) + ${par(n.bh1)}`,
            `       = ${fmt(r.t1)}`,
            `output = squish(${fmt(r.t1)}) = ${b(fmt(r.h1))}`,
          ]),
          note: 'Multiply, add, squish: the same three moves as the very first neuron in this book.',
          hot: ['left', 'right', 'h1', 'lh1', 'rh1', 'bh1'], flows: [['left', 'h1'], ['right', 'h1']],
        };
        case 2: return {
          title: 'Forward: H2 does the same with its own numbers',
          body: math([
            'total  = (left × weight) + (right × weight) + bias',
            `       = (${x[0]} × ${fmt(n.lh2)}) + (${x[1]} × ${fmt(n.rh2)}) + ${par(n.bh2)}`,
            `       = ${fmt(r.t2)}`,
            `output = squish(${fmt(r.t2)}) = ${b(fmt(r.h2))}`,
          ]),
          note: 'Same inputs, different weights, so a different output.',
          hot: ['left', 'right', 'h2', 'lh2', 'rh2', 'bh2'], flows: [['left', 'h2'], ['right', 'h2']],
        };
        case 3: return {
          title: 'Forward: out turns H1 and H2 into a prediction',
          body: math([
            'total      = (H1 × weight) + (H2 × weight) + bias',
            `           = (${fmt(r.h1)} × ${fmt(n.h1o)}) + (${fmt(r.h2)} × ${fmt(n.h2o)}) + ${par(n.bo)}`,
            `           = ${fmt(r.tOut)}`,
            `prediction = squish(${fmt(r.tOut)}) = ${b(fmt(r.out))}`,
          ]),
          note: `That is the whole forward pass. The network says ${fmt(r.out)}. The right answer is ${e.y}.`,
          hot: ['h1', 'h2', 'out', 'h1o', 'h2o', 'bo'], flows: [['h1', 'out'], ['h2', 'out']],
        };
        case 4: return {
          title: 'Measure the mistake',
          body: math([
            `miss = prediction - target  = ${fmt(r.out)} - ${e.y} = ${b(fmt(r.miss))}`,
            `loss = half of miss × miss  = 0.5 × ${par(r.miss)} × ${par(r.miss)} = ${b(fmt(r.loss, 4))}`,
          ]),
          note: `The miss is ${r.miss < 0 ? 'negative: the prediction came out too low' : 'positive: the prediction came out too high'}. The loss is the one number training tries to shrink. Now turn around and walk back through the network, asking every number the same question: how much of this was your fault?`,
          hot: ['out', 'loss'], flows: [['out', 'loss']],
        };
        case 5: return {
          title: 'Backward: how much is out to blame?',
          lead: 'Blame asks: if this number went up a tiny bit, how fast would the loss go up? For a neuron, the number in question is its total, the sum before the squish. Out\'s total reaches the loss through two links, its squish and then the miss, so multiply the two rates.',
          body: math([
            'slope of out\'s squish = prediction × (1 - prediction)',
            `                      = ${fmt(r.out)} × ${fmt(1 - r.out)} = ${fmt(r.slopeOut)}`,
            'blame at out          = miss × slope',
            `                      = ${fmt(r.miss)} × ${fmt(r.slopeOut)} = ${o(fmt(r.atOut, 4))}`,
          ]),
          note: r.atOut < 0
            ? 'Negative blame says: raise this number and the loss goes down. Out should have added up to more.'
            : 'Positive blame says: raise this number and the loss goes up. Out should have added up to less.',
          hot: ['out', 'loss'], flows: [['loss', 'out']], fresh: ['out'],
        };
        case 6: return {
          title: 'Backward: blame for the three numbers feeding out',
          lead: 'Rule one: a wire\'s blame is the blame at the neuron it feeds, times the signal it carried.',
          body: math([
            `H1→out   = blame at out × H1's output = ${fmt(r.atOut, 4)} × ${fmt(r.h1)} = ${o(fmt(r.blame.h1o, 4))}`,
            `H2→out   = blame at out × H2's output = ${fmt(r.atOut, 4)} × ${fmt(r.h2)} = ${o(fmt(r.blame.h2o, 4))}`,
            `bias out = blame at out × 1           = ${o(fmt(r.blame.bo, 4))}`,
          ]),
          note: 'A bias behaves like a wire whose signal is always 1. And the wire that carried the bigger signal takes the bigger share: changing its weight would have moved out\'s total more.',
          hot: ['h1', 'h2', 'out', 'h1o', 'h2o', 'bo'], fresh: ['h1o', 'h2o', 'bo'],
        };
        case 7: return handBack(r, n, 'h1');
        case 8: return handBack(r, n, 'h2');
        case 9: return intoHidden(r, n, x, 'h1');
        case 10: {
          const d = intoHidden(r, n, x, 'h2');
          const via = x[0] ? ['left→H2', 'left', x[0], r.blame.lh2] : x[1] ? ['right→H2', 'right', x[1], r.blame.rh2] : ['bias H2', '1', 1, r.blame.bh2];
          d.body += `<p class="bp-note">${d.note}</p>` + '<p>All nine numbers now have their blame. Look back at how one of them got its number: five rates multiplied along the path from that number to the loss. That is the chain rule.</p>' + math([
            `${via[0]} = miss × out's slope × weight H2→out × H2's slope × ${via[1]}`,
            `${' '.repeat(via[0].length)} = ${fmt(r.miss, 4)} × ${fmt(r.slopeOut, 4)} × ${par(n.h2o)} × ${fmt(r.slope2, 4)} × ${via[2]} = ${o(fmt(via[3], 4))}`,
          ]);
          d.note = '';
          return d;
        }
        default: {
          const after = forward(net, x);
          const lossAfter = lossOn(net, e);
          const rows = KEYS.map(k => `<tr><td><code>${NAMES[k]}</code></td><td>${fmt(n[k])}</td><td class="bp-o">${fmt(r.blame[k], 4)}</td><td><b>${fmt(net[k])}</b></td></tr>`).join('');
          return {
            title: 'Nudge all nine numbers',
            lead: `Every number moves against its blame: <code>new = old - learning rate × blame</code>. The learning rate here is ${LEARNING_RATE}.`,
            body: math([
              `prediction for this example   ${fmt(r.out)} → ${b(fmt(after.out))}   (target ${e.y})`,
              `loss for this example         ${fmt(r.loss, 4)} → ${b(fmt(lossAfter, 4))}`,
            ]) + `<table class="bp-table"><tr><th>number</th><th>old</th><th>blame</th><th>new</th></tr>${rows}</table>`,
            note: lossAfter < r.loss
              ? 'One round of learning, done. The prediction moved toward the target and the loss went down. Nothing was guessed: every nudge came from multiplying a few numbers you can see.'
              : 'This time the loss went up: the step was too big and overshot. It happens, and later rounds pull it back.',
            changed: KEYS.filter(k => fmt(n[k]) !== fmt(net[k])),
          };
        }
      }
    }

    function view(d) {
      const e = EXAMPLES[example];
      const r = backward(step === LAST ? undo.net : net, e.x, e.y);
      const after = step === LAST ? forward(net, e.x) : r;
      const values = { left: e.x[0], right: e.x[1] };
      if (step >= 1) values.h1 = after.h1;
      if (step >= 2) values.h2 = after.h2;
      if (step >= 3) values.out = after.out;
      const v = {
        net, values, target: e.y, hot: d.hot, flows: d.flows, fresh: d.fresh, changed: d.changed,
        tone: step >= 5 ? 'back' : 'fwd', blame: {}, blameAt: {},
        label: 'A network with two inputs, hidden neurons H1 and H2, and one output neuron, with every weight and bias shown as a number.',
      };
      if (step >= 4) {
        const miss = after.out - e.y;
        v.miss = miss;
        v.loss = 0.5 * miss * miss;
      }
      if (step >= 5 && step < LAST) {
        v.blameAt.out = r.atOut;
        if (step >= 6) for (const k of ['h1o', 'h2o', 'bo']) v.blame[k] = r.blame[k];
        if (step >= 7) v.blameAt.h1 = r.at1;
        if (step >= 8) v.blameAt.h2 = r.at2;
        if (step >= 9) for (const k of ['lh1', 'rh1', 'bh1']) v.blame[k] = r.blame[k];
        if (step >= 10) for (const k of ['lh2', 'rh2', 'bh2']) v.blame[k] = r.blame[k];
      }
      return v;
    }

    function drawChart() {
      const total = Math.max(history.length, span, 2);
      const stride = Math.ceil(total / 1200);
      const data = history.filter((_, i) => i % stride === 0 || i === history.length - 1);
      Book.lineChart($('bp-chart'), {
        series: [{ label: 'average loss, all four examples', color: '#ea580c', data }],
        xMax: Math.ceil(total / stride) + 1, yMin: 0, yMax: Math.max(0.16, Math.ceil(Math.max(...data) / 0.04) * 0.04), yFormat: y => y.toFixed(2),
        xLabel: `rounds of learning (${count(rounds)} so far)`,
      });
    }

    function renderStats() {
      $('bp-rounds').textContent = count(rounds);
      $('bp-loss').textContent = fmt(averageLoss(net), 4);
      $('bp-preds').innerHTML = '<table class="bp-table"><tr><th>left</th><th>right</th><th>network says</th><th>target</th><th></th></tr>' + EXAMPLES.map(ex => {
        const p = forward(net, ex.x).out;
        const ok = (p > 0.5) === (ex.y > 0.5);
        return `<tr><td>${ex.x[0]}</td><td>${ex.x[1]}</td><td><b>${fmt(p)}</b></td><td>${ex.y}</td>` +
          `<td class="${ok ? 'ok-mark' : 'bad-mark'}">${ok ? '✓' : '✗'}</td></tr>`;
      }).join('') + '</table>';
      drawChart();
    }

    function render() {
      const d = describe();
      diagram.innerHTML = drawNet(view(d));
      bringIntoView(diagram, d.hot);
      $('bp-panel').innerHTML =
        `<div class="bp-count">${busy ? 'the computer is driving' : `round ${count(step === LAST ? rounds : rounds + 1)} · step ${step} of ${LAST}`}</div>` +
        `<h4>${d.title}</h4>` +
        (d.lead ? `<p>${d.lead}</p>` : '') + (d.body || '') + (d.note ? `<p class="bp-note">${d.note}</p>` : '');
      const stage = STAGE_OF[step];
      const order = ['forward', 'loss', 'blame', 'nudge'];
      document.querySelectorAll('#bp-stages .stage-pill').forEach(el => {
        const at = order.indexOf(el.dataset.stage), now = order.indexOf(stage);
        el.classList.toggle('active', stage === el.dataset.stage);
        el.classList.toggle('done', now > at);
      });
      document.querySelectorAll('#bp-examples .toggle').forEach(el => {
        el.classList.toggle('on', +el.dataset.ex === example);
        el.setAttribute('aria-pressed', +el.dataset.ex === example);
      });
      $('bp-back').disabled = busy || step === 0;
      $('bp-next').disabled = busy;
      $('bp-next').textContent = step === LAST ? 'another round →' : step === LAST - 1 ? 'apply the nudges →' : 'next step →';
      $('bp-restart').disabled = busy;
      $('bp-train').disabled = busy || !unlocked;
      $('bp-train').textContent = busy ? 'training…' : `train ${count(MORE)} more rounds`;
      $('bp-panel').classList.toggle('stepping', step > 0);
    }

    function restart() {
      net = { ...START };
      step = 0; undo = null; rounds = 0; span = 0; trained = false;
      history = [averageLoss(net)];
      render();
      renderStats();
    }

    function next() {
      if (busy) return;
      trained = false;
      if (step === LAST) {
        step = 0;
        undo = null;
      } else if (step === LAST - 1) {
        const e = EXAMPLES[example];
        undo = { net };
        net = nudge(net, backward(net, e.x, e.y).blame);
        rounds++;
        history.push(averageLoss(net));
        unlocked = true;
        step = LAST;
        renderStats();
      } else step++;
      render();
    }

    function back() {
      if (busy || step === 0) return;
      if (step === LAST) {
        net = undo.net;
        undo = null;
        rounds--;
        history.pop();
        renderStats();
      }
      step--;
      render();
    }

    async function train() {
      if (busy || !unlocked) return;
      busy = true; trained = false; step = 0; undo = null;
      span = history.length + MORE;
      render();
      const started = performance.now();
      let done = 0;
      while (done < MORE) {
        const due = Math.min(MORE, Math.max(done + 1, Math.ceil(MORE * (performance.now() - started) / TRAIN_MS)));
        for (; done < due; done++) {
          const e = EXAMPLES[done % EXAMPLES.length];
          net = nudge(net, backward(net, e.x, e.y).blame);
          rounds++;
          history.push(averageLoss(net));
        }
        diagram.innerHTML = drawNet(view({}));
        renderStats();
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      busy = false; trained = true;
      render();
      renderStats();
    }

    wireToggles($('bp-examples'), tog => {
      if (busy) return;
      example = +tog.dataset.ex;
      if (step === LAST) undo = null;
      step = 0;
      render();
    });
    $('bp-next').addEventListener('click', next);
    $('bp-back').addEventListener('click', back);
    $('bp-restart').addEventListener('click', () => { if (!busy) restart(); });
    $('bp-train').addEventListener('click', train);
    restart();
  })();

  (function wiggleWidget() {
    const diagram = document.getElementById('wg-diagram');
    if (!diagram) return;
    const $ = id => document.getElementById(id);
    const PLACES = { 0.1: 3, 0.01: 3, 0.001: 3, 0.0001: 4 };
    let example = 1, pick = 'lh1', size = 0.001, shown = [], result = '';

    $('wg-keys').innerHTML = KEYS.map(k => `<div class="toggle" role="button" tabindex="0" data-key="${k}"><span>${NAMES[k]}</span></div>`).join('');

    const shownGap = (a, b) => Math.abs(Number(fmt(a, 5)) - Number(fmt(b, 5)));

    function measure(keys) {
      const e = EXAMPLES[example];
      let passes = 1;
      const before = lossOn(START, e);
      const rows = keys.map(key => {
        passes++;
        const after = lossOn({ ...START, [key]: START[key] + size }, e);
        return { key, before, after, slope: (after - before) / size };
      });
      return { rows, passes, blame: backward(START, e.x, e.y).blame };
    }

    function wiggleOne() {
      const { rows: [m], passes, blame } = measure([pick]);
      const gap = shownGap(m.slope, blame[pick]);
      const name = NAMES[pick];
      let verdict;
      if (m.slope === 0 && blame[pick] === 0) verdict = 'Both are exactly zero. This wire carried a 0, so changing its weight changes nothing, and backprop knew that without trying it.';
      else if (gap === 0) verdict = 'With a wiggle this small, the two numbers agree to every digit shown.';
      else if (size >= 0.01) verdict = `The two numbers differ by ${fmt(gap, 5)}. Close, but a wiggle of ${size} is a clumsy ruler: the loss curves a little over that distance. Pick a smaller wiggle and watch the gap shrink.`;
      else verdict = `The two numbers differ by only ${fmt(gap, 5)}. That leftover gap is the wiggle's fault, not backprop's: ${size} is small, but not infinitely small. Backprop's number is the exact slope.`;
      result = '<h4>One wiggle</h4><pre class="bp-math">' + [
        `loss with ${name} = ${fmt(START[pick], PLACES[size])}`.padEnd(34) + fmt(m.before, 8).padStart(12),
        `loss with ${name} = ${fmt(START[pick] + size, PLACES[size])}`.padEnd(34) + fmt(m.after, 8).padStart(12),
        'the loss moved by'.padEnd(34) + fmt(m.after - m.before, 8).padStart(12),
        `moved ÷ wiggle (${size})`.padEnd(34) + `<b>${fmt(m.slope, 5).padStart(9)}</b>      ← measured by wiggling`,
        `backprop's blame for ${name}`.padEnd(34) + `<b class="bp-o">${fmt(blame[pick], 5).padStart(9)}</b>      ← from the chain rule`,
      ].join('\n') + `</pre><p class="bp-note">${verdict} This check cost ${passes} forward passes: one before the wiggle and one after.</p>`;
      shown = [pick];
      render();
    }

    function wiggleAll() {
      const { rows, passes, blame } = measure(KEYS);
      const body = rows.map(m => `<tr><td><code>${NAMES[m.key]}</code></td><td>${fmt(m.slope, 5)}</td>` +
        `<td class="bp-o">${fmt(blame[m.key], 5)}</td><td>${fmt(shownGap(m.slope, blame[m.key]), 5)}</td></tr>`).join('');
      result = `<h4>All nine, wiggled by ${size}</h4>` +
        `<table class="bp-table"><tr><th>number</th><th>by wiggling</th><th>by backprop</th><th>gap</th></tr>${body}</table>` +
        `<p class="bp-note">Same nine answers, two very different bills. Wiggling ran the network <b>${passes} times</b>: once for the starting loss, then once more for each number. ` +
        'Backprop ran it forward <b>once</b> and backward <b>once</b>.</p>';
      shown = KEYS;
      render();
    }

    function render() {
      const e = EXAMPLES[example];
      const r = backward(START, e.x, e.y);
      const blame = {};
      for (const k of shown) blame[k] = r.blame[k];
      diagram.innerHTML = drawNet({
        net: START, values: { left: e.x[0], right: e.x[1], h1: r.h1, h2: r.h2, out: r.out },
        target: e.y, miss: r.miss, loss: r.loss, blame, pick, pickable: true,
        label: 'The same network with its starting numbers. Click a weight or bias to choose it.',
      });
      bringIntoView(diagram, [NODES.h1.bias === pick ? 'h1' : NODES.h2.bias === pick ? 'h2' : NODES.out.bias === pick ? 'out' : pick]);
      document.querySelectorAll('#wg-examples .toggle').forEach(el => {
        el.classList.toggle('on', +el.dataset.ex === example);
        el.setAttribute('aria-pressed', +el.dataset.ex === example);
      });
      document.querySelectorAll('#wg-keys .toggle').forEach(el => {
        el.classList.toggle('on', el.dataset.key === pick);
        el.setAttribute('aria-pressed', el.dataset.key === pick);
      });
      document.querySelectorAll('#wg-sizes .toggle').forEach(el => {
        el.classList.toggle('on', +el.dataset.size === size);
        el.setAttribute('aria-pressed', +el.dataset.size === size);
      });
      $('wg-one').textContent = `wiggle ${NAMES[pick]} by ${size}`;
      $('wg-panel').innerHTML = result ||
        '<h4>Nothing measured yet</h4><p>Choose one of the nine numbers, then press the wiggle button.</p>';
    }

    function clear() {
      shown = [];
      result = '';
      render();
    }

    wireToggles($('wg-examples'), tog => { example = +tog.dataset.ex; clear(); });
    wireToggles($('wg-keys'), tog => { pick = tog.dataset.key; clear(); });
    wireToggles($('wg-sizes'), tog => { size = +tog.dataset.size; clear(); });
    diagram.addEventListener('click', e => {
      const hit = e.target.closest('.pickable[data-key]');
      if (!hit) return;
      pick = hit.dataset.key;
      clear();
    });
    $('wg-one').addEventListener('click', wiggleOne);
    $('wg-all').addEventListener('click', wiggleAll);
    render();
  })();
})();
