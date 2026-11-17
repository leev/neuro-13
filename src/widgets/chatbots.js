// ===== Interactive: toy tokenizer (chapter 14) =====
(function() {
  const input = document.getElementById('tk-input');
  if (!input) return;

  const WORDS = ['the','a','an','and','me','my','you','is','tell','story','about','dragon','pizza','who','like','cat','dog','robot','school','play','time','big','small','happy','eat','fly'];
  const CHUNKS = ['ing','ed','er','est','ly','un','re','th','sh','ch','oo','ee','ea','ou'];
  const LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('');
  const PUNCT = ['.',',','!','?',"'"];
  const VOCAB = [];
  const ID = {};
  WORDS.concat(CHUNKS, LETTERS, PUNCT).forEach(t => {
    if (ID[t] === undefined) { ID[t] = VOCAB.length; VOCAB.push(t); }
  });
  const MAXLEN = Math.max.apply(null, VOCAB.map(t => t.length));

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  // random-but-fixed toy embeddings (a real model LEARNS these)
  const EMB = VOCAB.map((_, i) => {
    const r = mulberry32(777 + i * 13);
    return Array.from({length: 16}, () => r() * 2 - 1);
  });

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function tokenize(text) {
    const out = [];
    const parts = text.toLowerCase().match(/[a-z]+|[^a-z\s]/gu) || [];
    for (const part of parts) {
      const glue = PUNCT.includes(part);   // punctuation sticks to the word before it
      let i = 0, first = true;
      while (i < part.length) {
        let tok = null;
        for (let L = Math.min(MAXLEN, part.length - i); L >= 1; L--) {
          const s = part.slice(i, i + L);
          if (ID[s] !== undefined) { tok = s; break; }
        }
        if (tok === null) {
          const cp = String.fromCodePoint(part.codePointAt(i));  // keep emoji etc. whole
          out.push({ text: cp, id: null, newWord: first && !glue });
          i += cp.length;
        } else {
          out.push({ text: tok, id: ID[tok], newWord: first && !glue });
          i += tok.length;
        }
        first = false;
      }
    }
    return out;
  }

  function detokenize(toks) {
    return toks.map((t, i) => (t.newWord && i > 0 ? ' ' : '') + t.text).join('');
  }

  function render() {
    const text = input.value;
    const toks = tokenize(text);

    document.getElementById('tk-pills').innerHTML = toks.length
      ? toks.map(t => t.id === null
          ? '<span class="token-pill" style="border-color: var(--bad); color: var(--bad); background: #fee2e2;">' + esc(t.text) + '<span class="token-id">no id!</span></span>'
          : '<span class="token-pill">' + esc(t.text) + '<span class="token-id">id ' + t.id + '</span></span>'
        ).join('')
      : '<span style="font-size: 13px; color: var(--muted);">(type something above)</span>';

    document.getElementById('tk-ids').innerHTML =
      '<span class="cmt">// ' + toks.length + ' token' + (toks.length === 1 ? '' : 's') + ' from ' + text.length + ' characters</span>\n' +
      'input_ids = [' + toks.map(t => t.id === null ? '<span style="color:#fca5a5;">??</span>' : t.id).join(', ') + ']';

    const known = toks.filter(t => t.id !== null).slice(0, 10);
    document.getElementById('tk-embeds').innerHTML = known.map(t => {
      const squares = EMB[t.id].map(v => {
        const col = v >= 0
          ? 'rgba(37, 99, 235, ' + Math.abs(v).toFixed(2) + ')'
          : 'rgba(220, 38, 38, ' + Math.abs(v).toFixed(2) + ')';
        return '<div class="embed-square" style="background:' + col + '"></div>';
      }).join('');
      return '<div class="embed-block"><span class="name">' + esc(t.text) + ' (' + t.id + ')</span><div class="row"><div class="embed-row">' + squares + '</div></div></div>';
    }).join('') + (toks.filter(t => t.id !== null).length > 10
      ? '<p style="font-size: 12px; color: var(--muted); margin: 6px 0 0;">(showing the first 10 rows)</p>' : '');

    document.getElementById('tk-roundtrip').innerHTML =
      'Round trip — reading the ids back through the table: <strong>"' + esc(detokenize(toks)) + '"</strong>';
  }

  function renderRev() {
    const v = parseInt(document.getElementById('tk-rev').value, 10);
    const out = document.getElementById('tk-rev-out');
    if (isNaN(v)) out.innerHTML = '→ type a number';
    else if (v >= 0 && v < VOCAB.length) out.innerHTML = '→ "<strong>' + esc(VOCAB[v]) + '</strong>"';
    else out.innerHTML = '→ no row ' + v + ' — this toy table only goes 0–' + (VOCAB.length - 1);
  }

  document.getElementById('tk-vsize').textContent = VOCAB.length;
  document.getElementById('tk-vocab').innerHTML =
    '<div class="token-row">' + VOCAB.map((t, i) =>
      '<span class="token-pill">' + esc(t) + '<span class="token-id">id ' + i + '</span></span>'
    ).join('') + '</div>';

  input.addEventListener('input', render);
  document.getElementById('tk-rev').addEventListener('input', renderRev);
  document.getElementById('tk-vocab-toggle').addEventListener('click', () => {
    const el = document.getElementById('tk-vocab');
    el.hidden = !el.hidden;
    document.getElementById('tk-vocab-toggle').textContent =
      el.hidden ? 'show the whole lookup table' : 'hide the lookup table';
  });
  render();
  renderRev();
})();


// ===== Interactive: full pipeline =====
(function() {
  const tokens = [
    { text: 'tell',    id: 2842 },
    { text: ' me',     id: 502 },
    { text: ' a',      id: 257 },
    { text: ' story',  id: 1334 },
    { text: ' about',  id: 546 },
    { text: ' a',      id: 257 },
    { text: ' dragon', id: 6675 },
    { text: ' who',    id: 508 },
    { text: ' likes',  id: 7832 },
    { text: ' pizza',  id: 14256 },
  ];
  // attention weights for the last token ("pizza") attending to each token
  const attn = [0.04, 0.02, 0.02, 0.14, 0.04, 0.02, 0.32, 0.05, 0.28, 0.07];
  // candidates for first generated token
  const firstCandidates = [
    [' Once',  0.42],
    [' Sure',  0.15],
    [' In',    0.08],
    [' There', 0.06],
    [' One',   0.04],
  ];

  // deterministic fake embedding (just for visualization)
  function fakeEmbed(id, len) {
    const out = [];
    let seed = (id * 2654435761) >>> 0;
    for (let i = 0; i < len; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      out.push(((seed % 2000) / 1000) - 1);
    }
    return out;
  }
  function esc(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/ /g, '·');
  }

  // Stage 1: tokens
  (function() {
    let html = '';
    for (const t of tokens) {
      html += `<div class="token-pill">"${esc(t.text)}"<span class="token-id">id ${t.id}</span></div>`;
    }
    document.getElementById('pipe-tokens-1').innerHTML = html;
  })();

  // Stage 2: embeddings (32 squares per token)
  (function() {
    let html = '';
    for (const t of tokens) {
      const v = fakeEmbed(t.id, 32);
      let squares = '';
      for (const x of v) {
        const intensity = Math.min(1, Math.abs(x));
        const col = x >= 0
          ? `rgba(37, 99, 235, ${intensity})`
          : `rgba(220, 38, 38, ${intensity})`;
        squares += `<div class="embed-square" style="background: ${col};"></div>`;
      }
      html += `<div class="embed-block"><div class="name">"${esc(t.text)}"</div><div class="row"><div class="embed-row">${squares}</div></div></div>`;
    }
    document.getElementById('pipe-embeds').innerHTML = html;
  })();

  // Stage 3: attention bars
  (function() {
    let html = '<div style="font-size:12px; color: var(--muted); margin-bottom:10px; text-transform: uppercase; letter-spacing: 0.5px;">"pizza" is looking at →</div>';
    for (let i = 0; i < tokens.length; i++) {
      const w = attn[i];
      html += `<div class="attn-line">
        <span class="to">"${esc(tokens[i].text)}"</span>
        <div class="bar-wrap"><div class="bar-fill" style="width: ${(w*100/0.35)}%;"></div></div>
        <span class="val">${w.toFixed(2)}</span>
      </div>`;
    }
    html += '<p style="margin-top: 12px; font-size: 14px; color: #374151;">Notice: "pizza" pays the most attention to <strong>"dragon"</strong> and <strong>"likes"</strong>. That tells the next layers "I am the food being liked by the dragon" — exactly the info needed to predict where the story should go.</p>';
    document.getElementById('pipe-attn').innerHTML = html;
  })();

  // Stage 4: prediction
  (function() {
    let html = '<div style="font-size:12px; color: var(--muted); margin-bottom:10px; text-transform: uppercase; letter-spacing: 0.5px;">Top 5 candidates for the FIRST new token</div>';
    for (const [tok, p] of firstCandidates) {
      html += `<div class="bar-container"><span class="label" style="font-family: ui-monospace, Menlo, monospace; width: 110px; color: var(--accent);">"${esc(tok)}"</span><div class="bar-track" style="max-width: 240px;"><div class="bar" style="width:${p*100}%;"></div></div><span class="pct">${p.toFixed(2)}</span></div>`;
    }
    html += '<p style="margin-top:14px; font-size:14px; color: #374151;">Model picks <strong>"·Once"</strong> (highest probability). Now the prompt becomes "<em>...pizza Once</em>" and the entire pipeline runs again to pick the next token. See the demo below to keep watching the story unfold.</p>';
    document.getElementById('pipe-predict').innerHTML = html;
  })();

  let currentStage = 1;
  const totalStages = 4;
  function show(stage) {
    for (let s = 1; s <= totalStages; s++) {
      document.getElementById('pipe-stage-' + s).hidden = s > stage;
      const pill = document.getElementById('pipe-pill-' + s);
      pill.classList.toggle('active', s === stage);
      pill.classList.toggle('done', s < stage);
    }
    currentStage = stage;
    const btn = document.getElementById('pipe-next');
    btn.textContent = stage === totalStages ? 'done — restart to replay' : 'next stage →';
    btn.disabled = stage === totalStages;
    btn.style.opacity = stage === totalStages ? '0.5' : '1';
  }
  document.getElementById('pipe-next').addEventListener('click', () => {
    if (currentStage < totalStages) show(currentStage + 1);
  });
  document.getElementById('pipe-reset').addEventListener('click', () => show(1));
  show(1);
})();


// ===== Interactive 7: token-by-token story =====
(function() {
  const prompt = 'tell me a story about a dragon who likes pizza';
  const seq = [
    { token: ' Once',     candidates: [[' Once',0.42],[' Sure',0.15],[' In',0.08],[' There',0.06],[' One',0.04]] },
    { token: ' upon',     candidates: [[' upon',0.85],[' there',0.04],[' in',0.03],[' was',0.02],[' a',0.02]] },
    { token: ' a',        candidates: [[' a',0.91],[' the',0.03],[' an',0.02],[' some',0.01],[' my',0.01]] },
    { token: ' time',     candidates: [[' time',0.96],[' day',0.01],[' night',0.01],[' morning',0.01],[' year',0.005]] },
    { token: ',',         candidates: [[',',0.78],[' there',0.08],[' in',0.05],[' a',0.02],[' long',0.02]] },
    { token: ' in',       candidates: [[' in',0.34],[' there',0.30],[' a',0.10],[' deep',0.05],[' on',0.04]] },
    { token: ' a',        candidates: [[' a',0.62],[' the',0.15],[' an',0.05],[' some',0.03],[' his',0.02]] },
    { token: ' cave',     candidates: [[' cave',0.31],[' mountain',0.18],[' land',0.09],[' kingdom',0.07],[' forest',0.05]] },
    { token: ' high',     candidates: [[' high',0.21],[' deep',0.18],[' at',0.10],[' on',0.08],[' hidden',0.06]] },
    { token: ' in',       candidates: [[' in',0.45],[' on',0.20],[' atop',0.08],[' above',0.04],[' near',0.03]] },
    { token: ' the',      candidates: [[' the',0.71],[' a',0.10],[' some',0.03],[' my',0.02],[' those',0.02]] },
    { token: ' mountains',candidates: [[' mountains',0.42],[' clouds',0.20],[' hills',0.08],[' peaks',0.05],[' sky',0.04]] },
    { token: ',',         candidates: [[',',0.62],[' there',0.10],[' lived',0.08],[' was',0.04],[' a',0.03]] },
    { token: ' lived',    candidates: [[' lived',0.41],[' there',0.22],[' a',0.06],[' dwelt',0.05],[' sat',0.02]] },
    { token: ' a',        candidates: [[' a',0.79],[' the',0.08],[' an',0.03],[' Pepperoni',0.02],[' his',0.01]] },
    { token: ' dragon',   candidates: [[' dragon',0.84],[' friendly',0.04],[' young',0.03],[' lonely',0.02],[' hungry',0.02]] },
    { token: ' named',    candidates: [[' named',0.62],[' who',0.10],[' called',0.05],[' with',0.03],[' and',0.02]] },
    { token: ' Pepperoni',candidates: [[' Pepperoni',0.18],[' Cheese',0.09],[' Mozzarella',0.06],[' Crust',0.04],[' Pizza',0.04]] },
    { token: '.',         candidates: [['.',0.55],[',',0.20],[' who',0.10],[' .',0.03],[' The',0.02]] },
    { token: ' Pepperoni',candidates: [[' Pepperoni',0.45],[' He',0.18],[' She',0.10],[' Unlike',0.04],[' Most',0.03]] },
    { token: ' loved',    candidates: [[' loved',0.42],[' adored',0.10],[' liked',0.09],[' devoured',0.07],[' was',0.05]] },
    { token: ' pizza',    candidates: [[' pizza',0.91],[' food',0.03],[' people',0.01],[' adventure',0.01],[' gold',0.01]] },
    { token: ' more',     candidates: [[' more',0.42],[' so',0.14],[' above',0.10],[' over',0.06],[' deeply',0.04]] },
    { token: ' than',     candidates: [[' than',0.92],[' .',0.02],[' and',0.01],[' but',0.01],[' yet',0.01]] },
    { token: ' anything', candidates: [[' anything',0.45],[' any',0.20],[' all',0.10],[' gold',0.05],[' treasure',0.04]] },
    { token: ' else',     candidates: [[' else',0.71],[' in',0.18],[' on',0.04],[' he',0.02],[' .',0.01]] },
    { token: '.',         candidates: [['.',0.78],[',',0.10],[' in',0.05],[' that',0.02],[' and',0.02]] },
  ];

  let idx = 0;
  let displayed = prompt;
  let lastNew = '';

  function render() {
    const storyEl = document.getElementById('cg-story');
    if (lastNew) {
      storyEl.innerHTML = escapeHtml(displayed.slice(0, displayed.length - lastNew.length))
        + '<span class="new-token">' + escapeHtml(lastNew) + '</span>';
    } else {
      storyEl.textContent = displayed;
    }
    const candEl = document.getElementById('cg-candidates');
    if (idx >= seq.length) {
      candEl.innerHTML = '<em style="color: var(--muted); font-size: 14px;">Model produced "&lt;end&gt;" token. Story complete!</em>';
      return;
    }
    const { candidates } = seq[idx];
    let html = '<div style="font-size:12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Top 5 candidates for next token</div>';
    for (const [tok, prob] of candidates) {
      const pct = (prob*100).toFixed(0);
      const safeTok = escapeHtml(tok.replace(/ /g, '·'));
      html += `<div class="bar-container"><span class="label" style="font-family: ui-monospace, Menlo, monospace; width: 110px; color: var(--accent);">"${safeTok}"</span><div class="bar-track" style="max-width: 220px;"><div class="bar" style="width:${prob*100}%; background: var(--accent);"></div></div><span class="pct">${prob.toFixed(2)}</span></div>`;
    }
    candEl.innerHTML = html;
  }
  function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }
  function next(n=1) {
    for (let i=0; i<n && idx < seq.length; i++) {
      lastNew = seq[idx].token;
      displayed += lastNew;
      idx++;
    }
    render();
  }
  function reset() { idx = 0; displayed = prompt; lastNew = ''; render(); }

  document.getElementById('cg-next').addEventListener('click', () => next(1));
  document.getElementById('cg-fast').addEventListener('click', () => next(5));
  document.getElementById('cg-reset').addEventListener('click', reset);
  render();
})();

