// --- glossary: dotted-underline terms with hover/tap definitions ---
(function () {
  // canonical term -> short, plain-language definition
  const GLOSSARY = window.GLOSSARY || {};

  // lowercase lookup (so "ReLU"/"relu" both resolve) and a length-sorted alternation
  const byLower = {};
  Object.keys(GLOSSARY).forEach(k => { byLower[k.toLowerCase()] = k; });
  const terms = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  // \b TERM (optional plural) \b  — matches "bias", "biases", "weight", "weights", ...
  const RE = new RegExp('\\b(' + terms.map(esc).join('|') + ')(es|s)?\\b', 'gi');

  // never descend into these
  const SKIP = new Set(['SCRIPT', 'STYLE', 'TEXTAREA', 'CODE', 'PRE', 'NOSCRIPT', 'SVG', 'A', 'BUTTON', 'INPUT', 'LABEL', 'H1', 'H2', 'H3', 'H4']);

  function canonical(matchBase) {
    return byLower[matchBase.toLowerCase()] || null;
  }

  function wrapTextNode(node) {
    const text = node.nodeValue;
    if (!text || !text.trim() || !node.parentNode) return;
    RE.lastIndex = 0;
    let m, last = 0, frag = null;
    while ((m = RE.exec(text))) {
      const term = canonical(m[1]);
      if (!term) continue;
      if (!frag) frag = document.createDocumentFragment();
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const span = document.createElement('span');
      span.className = 'gloss';
      span.dataset.term = term;
      span.textContent = m[0];
      frag.appendChild(span);
      last = m.index + m[0].length;
    }
    if (frag) {
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  // Nodes a widget adds at runtime arrive here on their own, so the skip
  // rules have to look at ancestors too: an HTML <span> wrapped around a word
  // inside an SVG <text> simply makes the word disappear.
  const SKIP_CLASSES = ['gloss', 'terminal', 'CodeMirror', 'toggle', 'token-pill'];
  function skipped(el) {
    return SKIP.has(el.tagName.toUpperCase()) || el.hasAttribute('data-no-gloss') ||
      SKIP_CLASSES.some(c => el.classList.contains(c));
  }
  function insideSkipped(el) {
    for (; el && el.nodeType === 1; el = el.parentElement) {
      if (skipped(el)) return true;
    }
    return false;
  }

  function glossarize(root) {
    if (!root) return;
    if (insideSkipped(root.nodeType === 3 ? root.parentElement : root)) return;
    if (root.nodeType === 3) {
      wrapTextNode(root);
      return;
    }
    if (root.nodeType !== 1) return;
    const textNodes = [];
    (function walk(el) {
      for (let c = el.firstChild; c; c = c.nextSibling) {
        if (c.nodeType === 3) {
          textNodes.push(c);
        } else if (c.nodeType === 1 && !skipped(c)) {
          walk(c);
        }
      }
    })(root);
    textNodes.forEach(wrapTextNode);
  }

  const main = document.querySelector('main');
  if (!main) return;

  // tooltip element
  const tip = document.createElement('div');
  tip.id = 'gloss-tip';
  tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  let pinned = null; // element pinned open via click/tap

  function showTip(el) {
    const term = el.dataset.term;
    const def = GLOSSARY[term];
    if (!def) return;
    tip.innerHTML = '';
    const b = document.createElement('b');
    b.textContent = term;
    tip.appendChild(b);
    tip.appendChild(document.createTextNode(' — ' + def));
    tip.classList.add('show');
    const r = el.getBoundingClientRect();
    const tr = tip.getBoundingClientRect();
    let top = r.top - tr.height - 8;
    if (top < 8) top = r.bottom + 8;
    let left = r.left + r.width / 2 - tr.width / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tr.width - 8));
    tip.style.top = top + 'px';
    tip.style.left = left + 'px';
  }
  function hideTip() { tip.classList.remove('show'); }

  main.addEventListener('mouseover', e => {
    const el = e.target.closest('.gloss');
    if (el && !pinned) showTip(el);
  });
  main.addEventListener('mouseout', e => {
    if (!pinned && e.target.closest('.gloss')) hideTip();
  });
  main.addEventListener('focusin', e => {
    const el = e.target.closest('.gloss');
    if (el) showTip(el);
  });
  main.addEventListener('focusout', e => {
    if (e.target.closest('.gloss') && !pinned) hideTip();
  });
  // tap to pin (touch / click)
  main.addEventListener('click', e => {
    const el = e.target.closest('.gloss');
    if (el) {
      if (pinned === el) { pinned = null; hideTip(); }
      else { pinned = el; showTip(el); }
    } else if (pinned) {
      pinned = null; hideTip();
    }
  });
  window.addEventListener('scroll', () => { pinned = null; hideTip(); }, { passive: true });

  // initial pass over the static prose
  glossarize(main);

  // dynamic panels (live calculations, vote tables, etc.) re-render via innerHTML —
  // glossarize the freshly added nodes so terms in calculations get underlined too.
  const obs = new MutationObserver(muts => {
    obs.disconnect();
    for (const mu of muts) {
      mu.addedNodes.forEach(n => {
        if (n.nodeType === 1 || n.nodeType === 3) glossarize(n);
      });
    }
    obs.observe(main, { childList: true, subtree: true });
  });
  obs.observe(main, { childList: true, subtree: true });
})();
