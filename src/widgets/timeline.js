(function timelineWidget() {
  const root = document.getElementById('tl');
  if (!root) return;
  const byId = id => document.getElementById(id);
  const book = window.Book || {};
  const chapters = book.chapters || [];
  const parts = book.parts || [];
  const done = book.done && typeof book.done.has === 'function' ? book.done : null;
  const rail = byId('tl-rail'), list = byId('tl-list'), chips = byId('tl-chips');
  const FIRST = 1940, LAST = 2030;
  const place = year => (100 * (year - FIRST) / (LAST - FIRST)) + '%';

  const entries = [...list.querySelectorAll('.tl-item')].map((el, i, all) => {
    const year = +el.dataset.year;
    const end = +(el.dataset.end || year);
    const slugs = (el.dataset.chapters || '').split(' ').filter(Boolean);
    const first = chapters.find(c => c.slug === slugs[0]);
    const part = first && parts.find(p => p.id === first.part);
    if (part) {
      el.style.setProperty('--c', part.color);
      el.style.setProperty('--tint', part.tint);
    }
    const before = i ? all[i - 1].dataset : null;
    const gap = before ? Math.max(0, year - +(before.end || before.year)) : null;
    const gapEl = document.createElement('span');
    gapEl.className = 'tl-gap';
    gapEl.textContent = gap === null ? 'the line starts here' : gap === 0 ? 'the same year' : `+${gap} year${gap === 1 ? '' : 's'}`;
    el.querySelector('.tl-when').appendChild(gapEl);
    const finished = !!done && slugs.length > 0 && slugs.every(s => done.has(s));
    if (finished) {
      const check = document.createElement('span');
      check.className = 'tl-check';
      check.textContent = '✓ finished';
      el.querySelector('.tl-built').appendChild(check);
    }
    return { el, year, end, slugs, part, finished, label: el.querySelector('.tl-when').firstChild.textContent.trim(), tag: el.dataset.tag };
  });

  rail.innerHTML = '<div class="tl-axis"></div>';
  for (let y = FIRST; y <= LAST; y += 10) {
    const tick = document.createElement('div');
    tick.className = 'tl-tick';
    tick.style.left = place(y);
    if (y % 20 === 0) tick.innerHTML = `<span>${y}</span>`;
    rail.appendChild(tick);
  }
  const paint = (node, e) => { if (e.part) node.style.setProperty('--c', e.part.color); };
  entries.forEach(e => {
    if (e.end > e.year) {
      const span = document.createElement('div');
      span.className = 'tl-span';
      span.style.left = place(e.year);
      span.style.width = (100 * (e.end - e.year) / (LAST - FIRST)) + '%';
      paint(span, e);
      rail.appendChild(span);
    }
    e.dot = document.createElement('div');
    e.dot.className = 'tl-dot';
    e.dot.style.left = place((e.year + e.end) / 2);
    paint(e.dot, e);
    rail.appendChild(e.dot);

    e.chip = document.createElement('button');
    e.chip.type = 'button';
    e.chip.className = 'tl-chip';
    e.chip.innerHTML = `${e.year}<small>${e.tag}</small>`;
    paint(e.chip, e);
    if (e.part) e.chip.style.setProperty('--tint', e.part.tint);
    chips.appendChild(e.chip);
  });
  const flag = document.createElement('div');
  flag.className = 'tl-flag';
  rail.appendChild(flag);

  const legend = document.createElement('p');
  legend.className = 'tl-legend';
  legend.setAttribute('data-no-gloss', '');
  const used = parts.filter(p => entries.some(e => e.part === p));
  legend.innerHTML = used.map(p => `<span style="--c:${p.color}"><i></i>Part ${p.number}: ${p.title}</span>`).join('') +
    '<span><i></i>no small version in this book</span>';
  rail.after(legend);

  const built = entries.filter(e => e.slugs.length);
  byId('tl-total').textContent = entries.length;
  byId('tl-built').textContent = built.length;
  if (done) {
    byId('tl-done').textContent = `${built.filter(e => e.finished).length} of ${built.length}`;
    byId('tl-done-wrap').hidden = false;
  }

  let current = 0, showAll = false;

  function holdHeight() {
    if (showAll) { list.style.minHeight = ''; return; }
    entries.forEach(e => { e.el.hidden = false; });
    const tallest = Math.max(...entries.map(e => e.el.offsetHeight));
    entries.forEach((e, i) => { e.el.hidden = i !== current; });
    list.style.minHeight = tallest + 'px';
  }

  function show(i, scrollChip) {
    current = Math.max(0, Math.min(entries.length - 1, i));
    entries.forEach((e, k) => {
      const on = k === current;
      e.el.hidden = !showAll && !on;
      e.dot.classList.toggle('on', on);
      e.chip.classList.toggle('on', on);
      e.chip.setAttribute('aria-pressed', on);
    });
    const e = entries[current];
    const at = 100 * ((e.year + e.end) / 2 - FIRST) / (LAST - FIRST);
    flag.textContent = e.label;
    flag.style.left = Math.max(7, Math.min(93, at)) + '%';
    byId('tl-pos').textContent = `${current + 1} of ${entries.length}`;
    byId('tl-prev').disabled = current === 0;
    byId('tl-next').disabled = current === entries.length - 1;
    if (scrollChip) chips.scrollTo({ left: e.chip.offsetLeft - (chips.clientWidth - e.chip.offsetWidth) / 2, behavior: 'smooth' });
  }

  byId('tl-prev').addEventListener('click', () => show(current - 1, true));
  byId('tl-next').addEventListener('click', () => show(current + 1, true));
  entries.forEach((e, i) => e.chip.addEventListener('click', () => show(i, true)));
  rail.addEventListener('click', ev => {
    const box = rail.getBoundingClientRect();
    const year = FIRST + (LAST - FIRST) * (ev.clientX - box.left) / box.width;
    let best = 0;
    entries.forEach((e, i) => {
      if (Math.abs((e.year + e.end) / 2 - year) < Math.abs((entries[best].year + entries[best].end) / 2 - year)) best = i;
    });
    show(best, true);
  });
  root.addEventListener('keydown', ev => {
    if (showAll || ev.altKey || ev.ctrlKey || ev.metaKey) return;
    if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
    ev.preventDefault();
    // shell.js turns the page on arrow keys unless a button has focus; a link inside a card would do both.
    ev.stopPropagation();
    show(current + (ev.key === 'ArrowRight' ? 1 : -1), true);
  });
  const allBtn = byId('tl-all');
  allBtn.addEventListener('click', () => {
    showAll = !showAll;
    root.classList.toggle('tl-showall', showAll);
    allBtn.textContent = showAll ? 'back to one at a time' : 'show the whole line at once';
    allBtn.setAttribute('aria-pressed', showAll);
    show(current, false);
    holdHeight();
  });

  let resizing = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizing);
    resizing = setTimeout(holdHeight, 150);
  });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(holdHeight);
  show(0, false);
  holdHeight();
})();
