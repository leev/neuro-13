window.Book = (function () {
  const BOOK = window.BOOK || { chapters: [], parts: [] };
  const slug = document.body.dataset.slug;

  const store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem('neuro13.' + key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch (err) {
        return fallback;
      }
    },
    set(key, value) {
      try { localStorage.setItem('neuro13.' + key, JSON.stringify(value)); } catch (err) { /* private mode */ }
    },
  };

  const done = new Set(store.get('done', []));
  const quizzes = store.get('quiz', {});
  const root = document.documentElement;
  const fonts = getComputedStyle(root);
  const bodyFont = fonts.getPropertyValue('--font-body').trim() || 'system-ui, sans-serif';

  BOOK.parts.forEach((part, i) => {
    root.style.setProperty(`--p${i + 1}`, part.color);
    root.style.setProperty(`--t${i + 1}`, part.tint);
  });

  function paintProgress() {
    document.querySelectorAll('a[data-slug]').forEach(a => {
      a.classList.toggle('done', done.has(a.dataset.slug));
      const q = quizzes[a.dataset.slug];
      a.classList.toggle('starred', !!q && q.right === q.total);
    });
    const finished = BOOK.chapters.filter(c => done.has(c.slug)).length;
    document.querySelectorAll('[data-progress-count]').forEach(el => { el.textContent = finished; });
    root.style.setProperty('--progress', BOOK.chapters.length ? finished / BOOK.chapters.length : 0);
    for (const part of BOOK.parts) {
      const chapters = BOOK.chapters.filter(c => c.part === part.id);
      const n = chapters.filter(c => done.has(c.slug)).length;
      const complete = n === chapters.length;
      document.querySelectorAll(`[data-part-count="${part.id}"]`).forEach(el => {
        el.textContent = n ? `${n}/${chapters.length}` : '';
        const group = el.closest('details');
        if (group) group.classList.toggle('complete', complete);
      });
      document.querySelectorAll(`[data-part-bar="${part.id}"]`).forEach(el => { el.style.width = (100 * n / chapters.length) + '%'; });
      const card = document.getElementById('part-' + part.id);
      if (card) card.classList.toggle('complete', complete);
    }
    document.dispatchEvent(new CustomEvent('book:progress'));
  }

  function markDone(chapterSlug) {
    if (done.has(chapterSlug)) return;
    done.add(chapterSlug);
    store.set('done', [...done]);
    paintProgress();
  }

  function resetProgress() {
    done.clear();
    for (const key of Object.keys(quizzes)) delete quizzes[key];
    store.set('done', []);
    store.set('quiz', {});
    store.set('last', null);
    paintProgress();
  }

  function toast(text) {
    let el = document.getElementById('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      el.setAttribute('role', 'status');
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.remove('show'), 2200);
  }

  function rng(seed) {
    let a = seed | 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hidpi(canvas) {
    if (!canvas.dataset.w) {
      canvas.dataset.w = canvas.getAttribute('width');
      canvas.dataset.h = canvas.getAttribute('height');
    }
    const w = +canvas.dataset.w, h = +canvas.dataset.h;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.aspectRatio = w + ' / ' + h;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h, dpr };
  }

  function pointer(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const p = event.touches ? event.touches[0] : event;
    return {
      x: (p.clientX - rect.left) * (+canvas.dataset.w || canvas.width) / rect.width,
      y: (p.clientY - rect.top) * (+canvas.dataset.h || canvas.height) / rect.height,
    };
  }

  function lineChart(canvas, opts) {
    const { ctx, w, h } = hidpi(canvas);
    const series = opts.series.filter(s => s.data.length);
    const left = 46, right = 12, top = 14, bottom = opts.xLabel ? 34 : 22;
    const longest = Math.max(1, ...series.map(s => s.data.length));
    const xMax = opts.xMax || longest;
    const all = series.flatMap(s => s.data);
    const yMin = opts.yMin !== undefined ? opts.yMin : Math.min(0, ...all);
    const yMax = opts.yMax !== undefined ? opts.yMax : Math.max(1e-9, ...all) * 1.05;
    const fmt = opts.yFormat || (v => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)));
    const X = i => left + (xMax <= 1 ? 0 : (i / (xMax - 1)) * (w - left - right));
    const Y = v => top + (1 - (v - yMin) / (yMax - yMin)) * (h - top - bottom);

    ctx.clearRect(0, 0, w, h);
    ctx.font = '12px ' + bodyFont;
    ctx.textBaseline = 'middle';
    for (let t = 0; t <= 4; t++) {
      const v = yMin + (yMax - yMin) * t / 4;
      ctx.strokeStyle = t === 0 ? '#cfc5ae' : '#ece6d8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, Math.round(Y(v)) + 0.5);
      ctx.lineTo(w - right, Math.round(Y(v)) + 0.5);
      ctx.stroke();
      ctx.fillStyle = '#5c6274';
      ctx.textAlign = 'right';
      ctx.fillText(fmt(v), left - 6, Y(v));
    }
    if (opts.xLabel) {
      ctx.textAlign = 'center';
      ctx.fillStyle = '#5c6274';
      ctx.fillText(opts.xLabel, (left + w - right) / 2, h - 10);
    }
    const markers = opts.markers || [];
    const markerTop = opts.series.some(s => s.label) ? top + 16 : top;
    for (const mk of markers) {
      const x = X(mk.x);
      ctx.save();
      ctx.strokeStyle = mk.color || '#5c6274';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, markerTop);
      ctx.lineTo(x, h - bottom);
      ctx.stroke();
      ctx.restore();
    }
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2.25;
      ctx.beginPath();
      s.data.forEach((v, i) => {
        const y = Y(Math.max(yMin, Math.min(yMax, v)));
        if (i) ctx.lineTo(X(i), y); else ctx.moveTo(X(i), y);
      });
      ctx.stroke();
      const last = s.data.length - 1;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(X(last), Y(Math.max(yMin, Math.min(yMax, s.data[last]))), 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    let lx = left + 8;
    ctx.textAlign = 'left';
    for (const s of opts.series) {
      if (!s.label) continue;
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, top + 2, 12, 3);
      ctx.fillStyle = '#3b4157';
      ctx.fillText(s.label, lx + 17, top + 4);
      lx += 17 + ctx.measureText(s.label).width + 16;
    }
    for (const mk of markers) {
      if (!mk.label) continue;
      ctx.save();
      ctx.font = '600 12px ' + bodyFont;
      const width = ctx.measureText(mk.label).width;
      const labelX = Math.max(left + 4, Math.min(X(mk.x) + 7, w - right - width - 4));
      const labelY = h - bottom - 12;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(labelX - 4, labelY - 8, width + 8, 16);
      ctx.fillStyle = mk.color || '#3b4157';
      ctx.fillText(mk.label, labelX, labelY);
      ctx.restore();
    }
    return { ctx, w, h, left, right, top, bottom, X, Y };
  }

  function scoreQuiz(quiz) {
    if (!quiz) return;
    const questions = [...quiz.querySelectorAll('.q')];
    if (!questions.every(q => q.classList.contains('answered'))) return;
    const right = questions.filter(q => q.classList.contains('correct')).length;
    let result = quiz.querySelector('.quiz-result');
    if (!result) {
      result = document.createElement('div');
      result.className = 'quiz-result';
      quiz.appendChild(result);
    }
    const perfect = right === questions.length;
    result.classList.toggle('perfect', perfect);
    result.innerHTML = `<strong>${right} of ${questions.length}</strong> ${perfect ? '— nailed it.' : '— have another go?'} `;
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'secondary';
    retry.textContent = 'try again';
    retry.addEventListener('click', () => {
      questions.forEach(q => {
        q.classList.remove('answered', 'correct', 'incorrect');
        q.querySelectorAll('button[data-opt]').forEach(b => { b.disabled = false; b.classList.remove('picked', 'right', 'wrong'); });
      });
      result.remove();
    });
    if (!perfect) result.appendChild(retry);
    const best = quizzes[slug];
    if (!best || right > best.right) {
      quizzes[slug] = { right, total: questions.length };
      store.set('quiz', quizzes);
      paintProgress();
    }
  }

  function setMenu(open) {
    const was = document.body.classList.contains('menu-open');
    document.body.classList.toggle('menu-open', open);
    document.querySelectorAll('[data-menu]').forEach(b => b.setAttribute('aria-expanded', String(open)));
    if (open === was) return;
    const target = open
      ? document.querySelector('.sidebar a.active, .sidebar [data-menu-close]')
      : document.querySelector('[data-menu]');
    if (target && target.offsetParent !== null) target.focus({ preventScroll: !open });
  }

  document.addEventListener('click', e => {
    const btn = e.target.closest('.q button[data-opt], .predict button[data-opt]');
    if (btn) {
      const box = btn.closest('.q, .predict');
      if (box.classList.contains('answered')) return;
      const answer = box.dataset.answer;
      box.querySelectorAll('button[data-opt]').forEach(b => {
        b.disabled = true;
        if (answer && b.dataset.opt === answer) b.classList.add('right');
      });
      btn.classList.add('picked');
      if (answer && btn.dataset.opt !== answer) btn.classList.add('wrong');
      box.classList.add('answered');
      if (answer) box.classList.add(btn.dataset.opt === answer ? 'correct' : 'incorrect');
      scoreQuiz(box.closest('.quiz'));
      return;
    }
    if (e.target.closest('[data-menu]')) {
      setMenu(!document.body.classList.contains('menu-open'));
      return;
    }
    if (e.target.closest('[data-menu-close]')) {
      setMenu(false);
      return;
    }
    if (e.target.closest('[data-share]')) {
      const data = { title: document.title, url: location.href.split('#')[0] };
      if (navigator.share) navigator.share(data).catch(() => {});
      else if (navigator.clipboard) navigator.clipboard.writeText(data.url).then(() => toast('Link copied — paste it to a friend'));
      return;
    }
    if (e.target.closest('[data-reset-progress]')) {
      resetProgress();
      toast('Progress cleared');
    }
  });

  function makeTogglesReachable(root) {
    root.querySelectorAll('.toggle:not([tabindex])').forEach(el => {
      if (el.matches('button, a, input, label')) return;
      el.tabIndex = 0;
      el.setAttribute('role', 'button');
    });
  }
  makeTogglesReachable(document);
  if ('MutationObserver' in window) {
    new MutationObserver(() => makeTogglesReachable(document)).observe(document.body, { childList: true, subtree: true });
  }
  document.addEventListener('keydown', e => {
    if (e.defaultPrevented) return;
    if ((e.key === 'Enter' || e.key === ' ') && e.target.matches && e.target.matches('.toggle[role="button"]')) {
      e.preventDefault();
      e.target.click();
    }
  });

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') setMenu(false);
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.target.closest('input, textarea, select, button, [contenteditable], [role="button"], [role="dialog"], .CodeMirror, canvas, svg')) return;
    const rel = e.key === 'ArrowRight' ? 'next' : e.key === 'ArrowLeft' ? 'prev' : null;
    const link = rel && document.querySelector(`.pagers a[rel="${rel}"]`);
    if (link) location.href = link.href;
  });

  const chapter = BOOK.chapters.find(c => c.slug === slug);
  if (chapter) {
    store.set('last', slug);
    const end = document.querySelector('.chapter-end');
    if (end && 'IntersectionObserver' in window) {
      new IntersectionObserver((entries, obs) => {
        if (!entries.some(en => en.isIntersecting)) return;
        markDone(slug);
        const note = end.querySelector('.chapter-done');
        if (note) note.hidden = false;
        obs.disconnect();
      }, { threshold: 0.6 }).observe(end);
    }
    const active = document.querySelector('.sidebar a.active');
    if (active) active.scrollIntoView({ block: 'center' });
  }

  const last = store.get('last', null);
  const lastChapter = BOOK.chapters.find(c => c.slug === last);
  document.querySelectorAll('[data-continue]').forEach(a => {
    if (!lastChapter || !done.size) return;
    const i = BOOK.chapters.indexOf(lastChapter);
    const target = done.has(last) && BOOK.chapters[i + 1] ? BOOK.chapters[i + 1] : lastChapter;
    a.href = target.slug + '.html';
    a.textContent = `Continue: ${target.number}. ${target.title}`;
  });

  const readBar = document.querySelector('.read-progress');
  if (chapter && readBar) {
    let queued = false;
    const measure = () => {
      queued = false;
      const room = root.scrollHeight - window.innerHeight;
      readBar.style.setProperty('--read', room > 0 ? Math.min(1, window.scrollY / room).toFixed(4) : 0);
    };
    window.addEventListener('scroll', () => {
      if (!queued) { queued = true; requestAnimationFrame(measure); }
    }, { passive: true });
    measure();
  }

  paintProgress();

  return { chapters: BOOK.chapters, parts: BOOK.parts, slug, store, done, quizzes, markDone, resetProgress, toast, rng, hidpi, pointer, lineChart };
})();
