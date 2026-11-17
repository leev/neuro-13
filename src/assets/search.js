(function bookSearch() {
  const MAX_RESULTS = 12;
  const version = document.currentScript ? new URL(document.currentScript.src, location.href).search : '';
  const book = window.BOOK || { chapters: [], parts: [] };
  const glossary = window.GLOSSARY || {};
  let scrim, input, list, note, status;
  let index = null, loading = null, results = [], active = -1, opener = null;

  const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const escRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const ICON_SEARCH = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L20 20"/></svg>';
  const ICON_CLOSE = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  const ICON_WORD = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 5.5h5a3 3 0 0 1 3 3v11a2.5 2.5 0 0 0-2.5-2.5H4zM20 5.5h-5a3 3 0 0 0-3 3v11a2.5 2.5 0 0 1 2.5-2.5H20z"/></svg>';
  const ICON_PAGE = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 3.5h7l4 4V20.5H7z"/><path d="M14 3.5v4h4"/></svg>';

  function build() {
    scrim = document.createElement('div');
    scrim.className = 'search-scrim';
    scrim.hidden = true;
    scrim.innerHTML = `
      <div class="search-box" role="dialog" aria-modal="true" aria-label="Search the book">
        <div class="search-field">
          ${ICON_SEARCH}
          <input type="search" id="search-input" placeholder="Search every chapter" aria-label="Search every chapter" autocomplete="off" autocapitalize="off" spellcheck="false" role="combobox" aria-expanded="false" aria-controls="search-results" aria-autocomplete="list">
          <button type="button" class="icon-btn" data-search-close aria-label="Close search">${ICON_CLOSE}</button>
        </div>
        <p class="search-note"></p>
        <ol class="search-results" id="search-results" role="listbox" aria-label="Search results"></ol>
        <p class="sr-only" role="status" aria-live="polite"></p>
        <div class="search-foot" aria-hidden="true"><span><kbd>↑</kbd> <kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span><span><kbd>Esc</kbd> close</span></div>
      </div>`;
    document.body.appendChild(scrim);
    input = scrim.querySelector('input');
    list = scrim.querySelector('.search-results');
    note = scrim.querySelector('.search-note');
    status = scrim.querySelector('[role="status"]');

    scrim.addEventListener('mousedown', e => { if (e.target === scrim) close(); });
    scrim.querySelector('[data-search-close]').addEventListener('click', close);
    input.addEventListener('input', run);
    list.addEventListener('click', e => { if (e.target.closest('a')) close(false); });
    list.addEventListener('mousemove', e => {
      const item = e.target.closest('li');
      if (item) select([...list.children].indexOf(item), false);
    });
    scrim.addEventListener('keydown', onKey);
  }

  function load() {
    if (index) return Promise.resolve();
    if (!loading) {
      loading = fetch('assets/search-index.json' + version)
        .then(response => {
          if (!response.ok) throw new Error(response.status);
          return response.json();
        })
        .then(entries => {
          index = entries.map(entry => ({
            ...entry,
            titleL: entry.title.toLowerCase(),
            hookL: entry.hook.toLowerCase(),
            headingsL: entry.headings.map(h => h.toLowerCase()),
            textL: entry.text.toLowerCase(),
          }));
        })
        .catch(err => {
          loading = null;
          throw err;
        });
    }
    return loading;
  }

  function fieldScore(text, term, atWordStart, anywhere) {
    if (term.start.test(text)) return atWordStart;
    return text.includes(term.text) ? anywhere : 0;
  }

  function countIn(text, needle, cap) {
    let n = 0, at = text.indexOf(needle);
    while (at !== -1 && n < cap) {
      n++;
      at = text.indexOf(needle, at + needle.length);
    }
    return n;
  }

  function mark(text, terms) {
    const re = new RegExp('(' + terms.map(t => escRe(t.text)).sort((a, b) => b.length - a.length).join('|') + ')', 'gi');
    return text.split(re).map((piece, i) => (i % 2 ? `<mark>${esc(piece)}</mark>` : esc(piece))).join('');
  }

  function excerpt(entry, query, terms) {
    let at = entry.textL.indexOf(query);
    if (at === -1) {
      for (const term of terms) {
        const hit = term.start.exec(entry.textL);
        at = hit ? hit.index + hit[1].length : entry.textL.indexOf(term.text);
        if (at !== -1) break;
      }
    }
    if (at === -1) return null;
    let from = Math.max(0, at - 56), to = Math.min(entry.text.length, at + 104);
    if (from > 0) from = entry.text.indexOf(' ', from) + 1;
    if (to < entry.text.length) to = entry.text.lastIndexOf(' ', to);
    const shown = entry.text.slice(from, to).replace(/\s+([,.;:!?)\]])/g, '$1').replace(/([(\[])\s+/g, '$1');
    return {
      html: (from > 0 ? '… ' : '') + mark(shown, terms) + (to < entry.text.length ? ' …' : ''),
      jump: jumpText(entry.text, at),
    };
  }

  function jumpText(text, at) {
    const words = [];
    for (const raw of text.slice(text.lastIndexOf(' ', at) + 1, at + 80).split(' ')) {
      const word = raw.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
      if (!word) break;
      words.push(word);
      if (word !== raw || words.length === 5) break;
    }
    return words.join(' ');
  }

  const fragment = text => '#:~:text=' + encodeURIComponent(text).replace(/-/g, '%2D');

  function search(raw) {
    const query = raw.trim().toLowerCase().replace(/\s+/g, ' ');
    if (query.length < 2) return [];
    const terms = [...new Set(query.split(' '))].map(text => ({ text, start: new RegExp('(^|[^a-z0-9])' + escRe(text)) }));
    const found = [];

    const singular = query.replace(/(es|s)$/, '');
    Object.keys(glossary)
      .map(term => ({ term, lower: term.toLowerCase() }))
      .filter(g => g.lower === query || g.lower === singular || (query.length >= 3 && g.lower.startsWith(query)))
      .sort((a, b) => (a.lower === query ? -1 : b.lower === query ? 1 : a.lower.length - b.lower.length))
      .slice(0, 2)
      .forEach(g => found.push({
        score: 1000,
        href: 'glossary.html#' + encodeURIComponent(g.lower.replace(/ /g, '-')),
        badge: ICON_WORD,
        title: 'Glossary: ' + mark(g.term, terms),
        snippet: esc(glossary[g.term]),
      }));

    for (const entry of index) {
      let score = 0, inHeading = null, matchedAll = true;
      for (const term of terms) {
        const title = fieldScore(entry.titleL, term, 100, 60);
        let heading = 0;
        entry.headingsL.forEach((h, i) => {
          const s = fieldScore(h, term, 40, 25);
          if (s > heading) { heading = s; if (!inHeading) inHeading = entry.headings[i]; }
        });
        const hook = fieldScore(entry.hookL, term, 30, 18);
        const hits = countIn(entry.textL, term.text, 8);
        const total = title + heading + hook + (hits ? 6 + 2 * hits : 0);
        if (!total) { matchedAll = false; break; }
        score += total;
      }
      if (!matchedAll) continue;
      if (terms.length > 1) {
        if (entry.titleL.includes(query)) score += 150;
        if (entry.headingsL.some(h => h.includes(query))) score += 60;
        if (entry.textL.includes(query)) score += 25;
      }
      const chapter = book.chapters.find(c => c.slug === entry.slug);
      const part = chapter && book.parts.find(p => p.id === chapter.part);
      const cut = excerpt(entry, query, terms);
      found.push({
        score,
        number: entry.number,
        href: entry.slug + '.html' + (cut && cut.jump ? fragment(cut.jump) : inHeading ? fragment(inHeading) : ''),
        badge: String(entry.number),
        label: `Chapter ${entry.number}: `,
        part,
        title: mark(entry.title, terms),
        snippet: cut ? cut.html : inHeading ? 'Section: ' + mark(inHeading, terms) : mark(entry.hook, terms),
      });
    }

    document.querySelectorAll('.nav-extra a').forEach(link => {
      const title = link.textContent.trim();
      if (!terms.every(term => title.toLowerCase().includes(term.text))) return;
      found.push({ score: 90, href: link.getAttribute('href'), badge: ICON_PAGE, title: mark(title, terms), snippet: 'A page of this book' });
    });

    return found.sort((a, b) => b.score - a.score || (a.number || 0) - (b.number || 0)).slice(0, MAX_RESULTS);
  }

  function say(text) {
    note.hidden = !text;
    note.textContent = text;
  }

  function render() {
    list.innerHTML = results.map((r, i) => `
      <li role="option" id="search-option-${i}" aria-selected="false">
        <a href="${esc(r.href)}" tabindex="-1"${r.part ? ` style="--part:${esc(r.part.color)};--part-tint:${esc(r.part.tint)}"` : ''}>
          <span class="search-num" aria-hidden="true">${r.badge}</span>
          <span><span class="search-title">${r.label ? `<span class="sr-only">${r.label}</span>` : ''}${r.title}</span><span class="search-snippet">${r.snippet}</span></span>
        </a>
      </li>`).join('');
    input.setAttribute('aria-expanded', String(results.length > 0));
    select(results.length ? 0 : -1, false);
  }

  function select(i, scroll) {
    active = i;
    [...list.children].forEach((item, n) => item.setAttribute('aria-selected', String(n === i)));
    if (i >= 0) {
      input.setAttribute('aria-activedescendant', 'search-option-' + i);
      if (scroll) list.children[i].scrollIntoView({ block: 'nearest' });
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  function run() {
    const query = input.value;
    active = -1;
    if (query.trim().length < 2) {
      results = [];
      render();
      say('Type a word or two, like "bias" or "overfitting".');
      status.textContent = '';
      return;
    }
    if (!index) {
      results = [];
      render();
      say('Loading the book…');
      load().then(run, () => say('Search could not load. Check your connection, then try again.'));
      return;
    }
    results = search(query);
    render();
    say(results.length ? '' : `Nothing in the book matches "${query.trim()}". Try a shorter or simpler word.`);
    status.textContent = results.length === 1 ? '1 result' : `${results.length} results`;
  }

  function onKey(e) {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (results.length) select((active + (e.key === 'ArrowDown' ? 1 : results.length - 1)) % results.length, true);
    } else if (e.key === 'Enter' && e.target === input) {
      e.preventDefault();
      const link = list.children[Math.max(0, active)] && list.children[Math.max(0, active)].querySelector('a');
      if (link) link.click();
    } else if (e.key === 'Tab') {
      const stops = [input, scrim.querySelector('[data-search-close]')];
      e.preventDefault();
      stops[(stops.indexOf(document.activeElement) + 1) % stops.length].focus();
    }
  }

  function open() {
    if (!scrim) build();
    if (!scrim.hidden) return;
    opener = document.activeElement;
    scrim.hidden = false;
    document.documentElement.classList.add('search-open');
    document.body.classList.remove('menu-open');
    document.querySelectorAll('[data-menu]').forEach(b => b.setAttribute('aria-expanded', 'false'));
    input.focus();
    input.select();
    run();
    load().catch(() => {});
  }

  function close(restoreFocus) {
    if (!scrim || scrim.hidden) return;
    scrim.hidden = true;
    document.documentElement.classList.remove('search-open');
    if (restoreFocus !== false && opener && opener.focus) opener.focus();
  }

  document.addEventListener('click', e => {
    if (e.target.closest('[data-search]')) open();
  });

  document.addEventListener('keydown', e => {
    const typing = e.target.closest && e.target.closest('input, textarea, select, [contenteditable], .CodeMirror');
    if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === 'k') {
      if (e.ctrlKey && !e.metaKey && e.target.closest && e.target.closest('.CodeMirror')) return;
      e.preventDefault();
      if (scrim && !scrim.hidden) close(); else open();
    } else if (e.key === '/' && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      open();
    }
  });
})();
