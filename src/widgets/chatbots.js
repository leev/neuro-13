(function pipelineWidget() {
  const nextBtn = document.getElementById('pipe-next');
  if (!nextBtn) return;
  const $ = id => document.getElementById(id);
  const TOKENS = [['tell', 33331], [' me', 502], [' a', 257], [' story', 1621], [' about', 546],
    [' a', 257], [' dragon', 10441], [' who', 508], [' likes', 7832], [' pizza', 14256]];
  const ATTENTION = [0.04, 0.02, 0.02, 0.14, 0.04, 0.02, 0.32, 0.05, 0.28, 0.07];
  const FIRST_GUESSES = [[' Once', 0.42], [' Sure', 0.15], [' In', 0.08], [' There', 0.06], [' One', 0.04]];
  const STAGES = 4;
  const show = text => text.replace(/ /g, '·');

  $('pipe-tokens').innerHTML = TOKENS.map(([text, id]) =>
    `<code class="token-pill">${show(text)}<span class="token-id">id ${id}</span></code>`).join('');

  $('pipe-embeds').innerHTML = TOKENS.map(([text, id]) => {
    const random = Book.rng(id);
    let squares = '';
    for (let k = 0; k < 32; k++) {
      const v = random() * 2 - 1;
      squares += `<div class="embed-square" style="background:rgba(${v >= 0 ? '37, 99, 235' : '220, 38, 38'}, ${Math.abs(v).toFixed(2)})"></div>`;
    }
    return `<div class="embed-block"><code class="name">${show(text)}</code><div class="row"><div class="embed-row">${squares}</div></div></div>`;
  }).join('');

  const strongest = Math.max(...ATTENTION);
  $('pipe-attn').innerHTML = TOKENS.map(([text], i) =>
    `<div class="attn-line"><code class="to">${show(text)}</code>` +
    `<div class="bar-wrap"><div class="bar-fill" style="width:${100 * ATTENTION[i] / strongest}%"></div></div>` +
    `<span class="val">${ATTENTION[i].toFixed(2)}</span></div>`).join('');

  $('pipe-predict').innerHTML = FIRST_GUESSES.map(([text, p]) =>
    `<div class="bar-container"><code class="label cb-word">${show(text)}</code>` +
    `<div class="bar-track"><div class="bar" style="width:${p * 100}%"></div></div>` +
    `<span class="pct">${p.toFixed(2)}</span></div>`).join('');

  let stage = 1;
  function go(to) {
    stage = to;
    for (let s = 1; s <= STAGES; s++) {
      $('pipe-stage-' + s).hidden = s > stage;
      $('pipe-pill-' + s).classList.toggle('active', s === stage);
      $('pipe-pill-' + s).classList.toggle('done', s < stage);
    }
    nextBtn.disabled = stage === STAGES;
    nextBtn.textContent = stage === STAGES ? 'that is the whole trip' : 'next stage →';
  }
  nextBtn.addEventListener('click', () => go(Math.min(STAGES, stage + 1)));
  $('pipe-reset').addEventListener('click', () => go(1));
  go(1);
})();


(function storyWidget() {
  const nextBtn = document.getElementById('cg-next');
  if (!nextBtn) return;
  const $ = id => document.getElementById(id);
  const PROMPT = 'tell me a story about a dragon who likes pizza';
  const STEPS = [
    [[' Once', 0.42], [' Sure', 0.15], [' In', 0.08], [' There', 0.06], [' One', 0.04]],
    [[' upon', 0.85], [' there', 0.04], [' in', 0.03], [' was', 0.02], [' a', 0.02]],
    [[' a', 0.91], [' the', 0.03], [' an', 0.02], [' some', 0.01], [' my', 0.01]],
    [[' time', 0.96], [' day', 0.01], [' night', 0.01], [' morning', 0.01], [' year', 0.005]],
    [[',', 0.78], [' there', 0.08], [' in', 0.05], [' a', 0.02], [' long', 0.02]],
    [[' in', 0.34], [' there', 0.30], [' a', 0.10], [' deep', 0.05], [' on', 0.04]],
    [[' a', 0.62], [' the', 0.15], [' an', 0.05], [' some', 0.03], [' his', 0.02]],
    [[' cave', 0.31], [' mountain', 0.18], [' land', 0.09], [' kingdom', 0.07], [' forest', 0.05]],
    [[' high', 0.21], [' deep', 0.18], [' at', 0.10], [' on', 0.08], [' hidden', 0.06]],
    [[' in', 0.45], [' on', 0.20], [' atop', 0.08], [' above', 0.04], [' near', 0.03]],
    [[' the', 0.71], [' a', 0.10], [' some', 0.03], [' my', 0.02], [' those', 0.02]],
    [[' mountains', 0.42], [' clouds', 0.20], [' hills', 0.08], [' peaks', 0.05], [' sky', 0.04]],
    [[',', 0.62], [' there', 0.10], [' lived', 0.08], [' was', 0.04], [' a', 0.03]],
    [[' lived', 0.41], [' there', 0.22], [' a', 0.06], [' dwelt', 0.05], [' sat', 0.02]],
    [[' a', 0.79], [' the', 0.08], [' an', 0.03], [' one', 0.02], [' his', 0.01]],
    [[' dragon', 0.84], [' friendly', 0.04], [' young', 0.03], [' lonely', 0.02], [' hungry', 0.02]],
    [[' named', 0.62], [' who', 0.10], [' called', 0.05], [' with', 0.03], [' and', 0.02]],
    [[' Pepperoni', 0.18], [' Cheese', 0.09], [' Mozzarella', 0.06], [' Crust', 0.04], [' Pizza', 0.04]],
    [['.', 0.55], [',', 0.20], [' who', 0.10], [' and', 0.03], ['!', 0.02]],
    [[' Pepperoni', 0.45], [' He', 0.18], [' She', 0.10], [' Unlike', 0.04], [' Most', 0.03]],
    [[' loved', 0.42], [' adored', 0.10], [' liked', 0.09], [' devoured', 0.07], [' was', 0.05]],
    [[' pizza', 0.91], [' food', 0.03], [' people', 0.01], [' adventure', 0.01], [' gold', 0.01]],
    [[' more', 0.42], [' so', 0.14], [' above', 0.10], [' over', 0.06], [' deeply', 0.04]],
    [[' than', 0.92], ['.', 0.02], [' and', 0.01], [' but', 0.01], [' yet', 0.01]],
    [[' anything', 0.45], [' any', 0.20], [' all', 0.10], [' gold', 0.05], [' treasure', 0.04]],
    [[' else', 0.71], [' in', 0.18], [' on', 0.04], [' he', 0.02], ['.', 0.01]],
    [['.', 0.78], [',', 0.10], [' in', 0.05], [' that', 0.02], [' and', 0.02]],
    [['<end-of-text>', 0.64], [' He', 0.12], [' Every', 0.07], [' One', 0.05], [' But', 0.03]],
  ];
  const WRITTEN = STEPS.length - 1;

  let at = 0, fresh = 0;

  function render() {
    const words = STEPS.slice(0, Math.min(at, WRITTEN)).map(step => step[0][0]);
    const settled = words.slice(0, words.length - fresh).join('');
    const added = words.slice(words.length - fresh).join('');
    $('cg-prompt').textContent = PROMPT;
    $('cg-story').innerHTML = settled.trimStart() + (added ? `<span class="new-token">${settled ? added : added.trimStart()}</span>` : '') ||
      '<span class="cb-muted">(the reply appears here, one token at a time)</span>';
    $('cg-count').textContent = Math.min(at, WRITTEN);
    $('cg-runs').textContent = at;
    nextBtn.disabled = $('cg-fast').disabled = at >= STEPS.length;
    if (at >= STEPS.length) {
      $('cg-candidates').innerHTML = '<p class="cb-done">The model picked the special end-of-text token, so the app stops asking for more. The reply is finished.</p>';
      return;
    }
    $('cg-candidates').innerHTML = '<div class="cb-caption">Top 5 guesses for the next token</div>' +
      STEPS[at].map(([text, p], i) =>
        `<div class="bar-container"><code class="label cb-word${i === 0 ? ' pick' : ''}">${text.replace(/ /g, '·').replace(/</g, '&lt;')}</code>` +
        `<div class="bar-track"><div class="bar" style="width:${p * 100}%"></div></div>` +
        `<span class="pct">${p.toFixed(2)}</span></div>`).join('');
  }

  function advance(count) {
    const before = at;
    at = Math.min(STEPS.length, at + count);
    fresh = Math.min(at, WRITTEN) - Math.min(before, WRITTEN);
    render();
  }

  nextBtn.addEventListener('click', () => advance(1));
  $('cg-fast').addEventListener('click', () => advance(5));
  $('cg-reset').addEventListener('click', () => { at = 0; fresh = 0; render(); });
  render();
})();


(function samplingWidget() {
  function softmax(scores, temperature) {
    const top = Math.max(...scores);
    const raised = scores.map(z => Math.exp((z - top) / temperature));
    const total = raised.reduce((a, b) => a + b, 0);
    return raised.map(v => v / total);
  }

  function reshape(scores, temperature, topK, topP) {
    const order = scores.map((_, i) => i).sort((i, j) => scores[j] - scores[i]);
    const cut = scores.map(() => null);
    if (temperature <= 0) {
      const probs = scores.map(() => 0);
      probs[order[0]] = 1;
      return { probs, cut, order };
    }
    let probs = softmax(scores, temperature);
    const renormalize = () => {
      const kept = probs.reduce((sum, p, i) => sum + (cut[i] ? 0 : p), 0);
      probs = probs.map((p, i) => (cut[i] ? 0 : p / kept));
    };
    order.slice(topK).forEach(i => { cut[i] = 'top-k'; });
    renormalize();
    let running = 0, full = false;
    for (const i of order) {
      if (cut[i]) continue;
      if (full) cut[i] = 'top-p';
      else {
        running += probs[i];
        full = running >= topP - 1e-12;
      }
    }
    renormalize();
    return { probs, cut, order };
  }

  function draw(probs, random) {
    let r = random(), last = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] <= 0) continue;
      last = i;
      r -= probs[i];
      if (r < 0) return i;
    }
    return last;
  }

  const $ = id => document.getElementById(id);
  if (!$('sp-temp')) return;

  const WORDS = ['pizza', 'cheese', 'noodles', 'berries', 'toast', 'gold', 'anything', 'cake', 'soup', 'fire', 'broccoli', 'homework'];
  const SCORES = [5.0, 4.1, 3.7, 3.3, 2.9, 2.6, 2.2, 1.9, 1.5, 1.1, 0.4, -0.4];
  const N = WORDS.length;
  const FIRST = 0, RARE = WORDS.indexOf('broccoli');
  const plain = softmax(SCORES, 1);
  const pct = p => (p >= 0.9995 ? '100%' : p > 0 && p < 0.0005 ? '<0.1%' : (p * 100).toFixed(1) + '%');
  const describeOdds = odds => (odds >= 1e9 ? 'more than a billion' : odds >= 100 ? Math.round(odds).toLocaleString('en-US') : odds.toFixed(1));

  $('sp-bars').innerHTML = WORDS.map((word, i) =>
    `<div class="sp-row" id="sp-row-${i}"><code class="sp-word">${word}</code><span class="sp-score">${SCORES[i].toFixed(1)}</span>` +
    `<div class="sp-track"><div class="sp-bar"></div><div class="sp-tally"></div><div class="sp-ghost" style="left:${plain[i] * 100}%"></div></div>` +
    `<span class="sp-pct"></span><span class="sp-n"></span></div>`).join('');
  const rows = WORDS.map((_, i) => {
    const row = $('sp-row-' + i);
    const part = cls => row.querySelector('.' + cls);
    return { row, bar: part('sp-bar'), tally: part('sp-tally'), pct: part('sp-pct'), n: part('sp-n') };
  });

  let shaped, counts = new Array(N).fill(0), total = 0, lastPick = -1;

  function renderTally() {
    rows.forEach((r, i) => {
      r.n.textContent = total ? '×' + counts[i] : '';
      r.tally.style.width = (total ? 100 * counts[i] / total : 0) + '%';
      r.row.classList.toggle('picked', i === lastPick);
    });
    $('sp-total').textContent = total;
    $('sp-distinct').textContent = counts.filter(c => c > 0).length;
    $('sp-fill').innerHTML = lastPick < 0 ? '<span class="sp-blank">&nbsp;</span>' : `<mark>${WORDS[lastPick]}</mark>`;
  }

  function clearTally() {
    counts = new Array(N).fill(0);
    total = 0;
    lastPick = -1;
    renderTally();
  }

  function render() {
    const temperature = +$('sp-temp').value, topK = +$('sp-topk').value, topP = +$('sp-topp').value;
    $('sp-temp-v').textContent = temperature.toFixed(2);
    $('sp-topk-v').textContent = topK === N ? 'all ' + N : topK;
    $('sp-topp-v').textContent = topP >= 1 ? 'off' : topP.toFixed(2);
    shaped = reshape(SCORES, temperature, topK, topP);
    rows.forEach((r, i) => {
      r.bar.style.width = shaped.probs[i] * 100 + '%';
      r.row.classList.toggle('cut', !!shaped.cut[i]);
      r.pct.textContent = shaped.cut[i] ? shaped.cut[i] + ' cut' : pct(shaped.probs[i]);
    });
    const alive = shaped.probs.filter(p => p > 0).length;
    $('sp-alive').textContent = alive;

    const lines = [];
    if (temperature <= 0) {
      lines.push('<b>Temperature 0:</b> no dice at all. The token with the highest score is chosen every time, so top-k and top-p have nothing left to do.');
    } else {
      const gap = SCORES[FIRST] - SCORES[RARE];
      const odds = Math.exp(gap / temperature);
      lines.push(`<b>Temperature ${temperature.toFixed(2)}:</b> every score is divided by ${temperature.toFixed(2)}. ` +
        `${WORDS[FIRST]} goes from ${SCORES[FIRST].toFixed(1)} to ${(SCORES[FIRST] / temperature).toFixed(2)} and ${WORDS[RARE]} from ${SCORES[RARE].toFixed(1)} to ${(SCORES[RARE] / temperature).toFixed(2)}, ` +
        `so the gap between them is ${(gap / temperature).toFixed(2)}. Softmax turns that gap into odds of e<sup>${(gap / temperature).toFixed(2)}</sup> ≈ ${describeOdds(odds)} to 1.`);
      lines.push(topK >= N ? `<b>Top-k off:</b> all ${N} tokens stay in.`
        : `<b>Top-k ${topK}:</b> only the ${topK} highest score${topK === 1 ? '' : 's'} stay${topK === 1 ? 's' : ''} in. ${N - topK} cut.`);
      const byP = shaped.cut.filter(c => c === 'top-p').length;
      lines.push(topP >= 1 ? '<b>Top-p off:</b> nothing more is cut.'
        : `<b>Top-p ${topP.toFixed(2)}:</b> going down from the top, tokens are kept until their probabilities add up to ${Math.round(topP * 100)}%. ${byP} more cut.`);
      lines.push(`<b>Then</b> the ${alive} survivor${alive === 1 ? '' : 's'} ${alive === 1 ? 'is' : 'are'} scaled back up to add to 100%, and one is drawn at random with those odds.`);
    }
    $('sp-math').innerHTML = lines.map(line => `<li>${line}</li>`).join('');
  }

  function sample(times) {
    for (let t = 0; t < times; t++) {
      lastPick = draw(shaped.probs, Math.random);
      counts[lastPick]++;
      total++;
    }
    renderTally();
  }

  for (const id of ['sp-temp', 'sp-topk', 'sp-topp']) {
    $(id).addEventListener('input', () => { clearTally(); render(); });
  }
  $('sp-one').addEventListener('click', () => sample(1));
  $('sp-hundred').addEventListener('click', () => sample(100));
  $('sp-clear').addEventListener('click', clearTally);
  $('sp-defaults').addEventListener('click', () => {
    $('sp-temp').value = 1;
    $('sp-topk').value = N;
    $('sp-topp').value = 1;
    clearTally();
    render();
  });
  render();
  renderTally();
})();
