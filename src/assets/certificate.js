(function certificatePage() {
  const app = document.getElementById('cert-app');
  if (!app || !window.Book) return;
  const byId = id => document.getElementById(id);
  const W = 2000, H = 1414, INK = '#1d2238', SOFT = '#545a70', PAPER = '#fbf7ef';
  const css = getComputedStyle(document.documentElement);
  const DISPLAY = css.getPropertyValue('--font-display').trim() || 'sans-serif';
  const BODY = css.getPropertyValue('--font-body').trim() || 'sans-serif';
  const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const siteMeta = document.querySelector('meta[property="og:url"]');
  const site = siteMeta ? new URL(siteMeta.content).host : location.host;
  const bookName = (document.querySelector('.brand-name') || {}).textContent || 'Neuro 13';
  let imageUrl = null;

  function tally() {
    const parts = Book.parts.map(part => {
      const chapters = Book.chapters.filter(c => c.part === part.id);
      const done = chapters.filter(c => Book.done.has(c.slug));
      return { ...part, chapters, done: done.length, complete: done.length === chapters.length };
    });
    const total = Book.chapters.length;
    const done = parts.reduce((n, p) => n + p.done, 0);
    const stars = Book.chapters.filter(c => {
      const q = Book.quizzes[c.slug];
      return q && q.right === q.total;
    }).length;
    return { parts, total, done, stars, complete: total > 0 && done === total };
  }

  function wrap(ctx, text, maxWidth) {
    const lines = [];
    let line = '';
    for (const word of text.split(' ')) {
      const trial = line ? line + ' ' + word : word;
      if (line && ctx.measureText(trial).width > maxWidth) {
        lines.push(line);
        line = word;
      } else {
        line = trial;
      }
    }
    if (line) lines.push(line);
    return lines;
  }

  function roundedPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function borderPoints(inset, r, spacing, groups) {
    const w = W - 2 * inset, h = H - 2 * inset, quarter = Math.PI * r / 2;
    const runs = [
      { len: w - 2 * r, at: d => [inset + r + d, inset] },
      { len: quarter, at: d => [W - inset - r + r * Math.sin(d / r), inset + r - r * Math.cos(d / r)] },
      { len: h - 2 * r, at: d => [W - inset, inset + r + d] },
      { len: quarter, at: d => [W - inset - r + r * Math.cos(d / r), H - inset - r + r * Math.sin(d / r)] },
      { len: w - 2 * r, at: d => [W - inset - r - d, H - inset] },
      { len: quarter, at: d => [inset + r - r * Math.sin(d / r), H - inset - r + r * Math.cos(d / r)] },
      { len: h - 2 * r, at: d => [inset, H - inset - r - d] },
      { len: quarter, at: d => [inset + r - r * Math.cos(d / r), inset + r - r * Math.sin(d / r)] },
    ];
    const length = runs.reduce((sum, run) => sum + run.len, 0);
    const count = Math.round(length / spacing / groups) * groups;
    const points = [];
    for (let i = 0; i < count; i++) {
      let d = i * length / count;
      for (const run of runs) {
        if (d <= run.len) { points.push(run.at(d)); break; }
        d -= run.len;
      }
    }
    return points;
  }

  function drawMark(ctx, cx, cy, size, colors) {
    const u = size / 2;
    ctx.lineWidth = size * 0.085;
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.strokeStyle = INK;
    ctx.beginPath();
    ctx.moveTo(cx - u, cy + u);
    ctx.lineTo(cx - u, cy - u);
    ctx.lineTo(cx + u, cy + u);
    ctx.lineTo(cx + u, cy - u);
    ctx.stroke();
    const spots = [[-u, u], [-u, -u], [0, 0], [u, u], [u, -u]];
    spots.forEach(([dx, dy], i) => {
      ctx.fillStyle = colors[i % colors.length];
      ctx.beginPath();
      ctx.arc(cx + dx, cy + dy, size * (i === 2 ? 0.115 : 0.14), 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  function drawCertificate(canvas, state, name, dateText) {
    const ctx = canvas.getContext('2d');
    const colors = state.parts.map(p => p.color);
    const sample = !state.complete;
    canvas.width = W;
    canvas.height = H;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(29, 34, 56, 0.13)';
    for (let y = 30; y < H; y += 40) {
      for (let x = 20; x < W; x += 40) {
        ctx.beginPath();
        ctx.arc(x, y, 2.2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    ctx.fillStyle = '#fffdf8';
    roundedPath(ctx, 96, 96, W - 192, H - 192, 30);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = INK;
    ctx.stroke();
    borderPoints(60, 44, 36, colors.length).forEach(([x, y], i) => {
      ctx.fillStyle = colors[i % colors.length];
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    drawMark(ctx, W / 2, 232, 92, colors);
    ctx.fillStyle = INK;
    ctx.font = `800 54px ${DISPLAY}`;
    ctx.fillText('Certificate of completion', W / 2, 372);
    ctx.fillStyle = SOFT;
    ctx.font = `500 32px ${BODY}`;
    ctx.fillText('awarded to', W / 2, 438);

    let size = 150;
    ctx.font = `800 ${size}px ${DISPLAY}`;
    const widest = W - 480;
    const measured = ctx.measureText(name).width;
    if (measured > widest) size = Math.max(54, Math.floor(size * widest / measured));
    ctx.font = `800 ${size}px ${DISPLAY}`;
    ctx.fillStyle = sample ? 'rgba(29, 34, 56, 0.3)' : INK;
    ctx.fillText(name, W / 2, 600);

    const lineW = 1080, x0 = (W - lineW) / 2, seg = lineW / colors.length;
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    colors.forEach((color, i) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x0 + i * seg + 10, 652);
      ctx.lineTo(x0 + (i + 1) * seg - 10, 652);
      ctx.stroke();
    });

    ctx.fillStyle = '#3b4157';
    ctx.font = `400 35px ${BODY}`;
    const body = `for finishing all ${state.total} chapters of ${bookName} and training real neural networks along the way, from one tiny neuron to models that read, write, see and draw.`;
    wrap(ctx, body, 1280).forEach((line, i) => ctx.fillText(line, W / 2, 738 + i * 52));

    const gap = (W - 440) / state.parts.length;
    state.parts.forEach((part, i) => {
      const cx = 220 + gap * (i + 0.5), cy = 960, r = 52;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
      if (part.complete) {
        ctx.fillStyle = part.color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 10;
        ctx.lineCap = ctx.lineJoin = 'round';
        ctx.beginPath();
        ctx.moveTo(cx - 22, cy + 2);
        ctx.lineTo(cx - 6, cy + 18);
        ctx.lineTo(cx + 24, cy - 16);
        ctx.stroke();
      } else {
        ctx.fillStyle = '#fff';
        ctx.fill();
        ctx.strokeStyle = part.color;
        ctx.lineWidth = 6;
        ctx.setLineDash([10, 10]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = SOFT;
        ctx.font = `700 30px ${BODY}`;
        ctx.fillText(`${part.done}/${part.chapters.length}`, cx, cy + 11);
      }
      ctx.fillStyle = INK;
      ctx.font = `700 29px ${DISPLAY}`;
      const title = wrap(ctx, part.title, gap - 36);
      title.forEach((line, n) => ctx.fillText(line, cx, 1058 + n * 34));
      ctx.fillStyle = SOFT;
      ctx.font = `500 24px ${BODY}`;
      ctx.fillText(`${part.chapters.length} chapters`, cx, 1066 + title.length * 34);
    });

    ctx.font = `600 28px ${BODY}`;
    ctx.fillStyle = INK;
    ctx.textAlign = 'left';
    ctx.fillText(dateText, 170, 1262);
    ctx.textAlign = 'right';
    ctx.fillText(site, W - 170, 1262);
    if (state.stars > 0) {
      ctx.textAlign = 'center';
      ctx.fillStyle = SOFT;
      ctx.fillText(`${state.stars} perfect quiz ${state.stars === 1 ? 'score' : 'scores'}`, W / 2, 1262);
    }
    if (sample) {
      ctx.textAlign = 'right';
      ctx.font = `700 26px ${BODY}`;
      ctx.fillStyle = SOFT;
      ctx.fillText('Preview', W - 170, 190);
    }
  }

  function loadFonts(name) {
    if (!document.fonts || !document.fonts.load) return Promise.resolve();
    const wanted = [`800 150px ${DISPLAY}`, `700 29px ${DISPLAY}`, `400 35px ${BODY}`, `500 32px ${BODY}`, `600 28px ${BODY}`, `700 30px ${BODY}`];
    const all = Promise.all(wanted.map(font => document.fonts.load(font, name + ' Certificate 0123456789'))).catch(() => {});
    return Promise.race([all, new Promise(resolve => setTimeout(resolve, 2500))]);
  }

  function showCertificate(state, name) {
    const stored = Book.store.get('finished', null);
    const when = state.complete && stored ? new Date(stored) : new Date();
    if (state.complete && !stored) Book.store.set('finished', when.toISOString());
    const dateText = when.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const shown = state.complete ? name : 'Your name here';
    return loadFonts(shown).then(() => new Promise(resolve => {
      const canvas = document.createElement('canvas');
      drawCertificate(canvas, state, shown, dateText);
      canvas.toBlob(blob => {
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        imageUrl = URL.createObjectURL(blob);
        const img = byId('cert-image');
        img.onload = resolve;
        img.src = imageUrl;
        img.alt = state.complete
          ? `Certificate of completion awarded to ${name} for finishing all ${state.total} chapters of ${bookName}, dated ${dateText}.`
          : `A preview of the certificate with a blank space for your name. ${state.done} of ${state.total} chapters are finished so far.`;
        byId('cert-caption').textContent = state.complete
          ? ''
          : `A preview. The circles fill in as you finish each part: ${state.total - state.done} ${state.total - state.done === 1 ? 'chapter' : 'chapters'} to go.`;
        byId('cert-sheet').hidden = false;
        byId('cert-save').hidden = !state.complete;
      }, 'image/png');
    }));
  }

  function renderProgress(state) {
    byId('cert-parts').innerHTML = state.parts.map(part => {
      const dots = part.chapters.map(c => {
        const done = Book.done.has(c.slug), q = Book.quizzes[c.slug], star = !!q && q.right === q.total;
        const label = `Chapter ${c.number}: ${c.title}${done ? ', finished' : ', not finished yet'}${star ? ', every quiz question right' : ''}`;
        return `<li><a class="cert-dot${done ? ' done' : ''}${star ? ' starred' : ''}" href="${esc(c.slug)}.html" title="${esc(label)}" aria-label="${esc(label)}">${c.number}</a></li>`;
      }).join('');
      const left = part.chapters.filter(c => !Book.done.has(c.slug));
      const todo = left.length
        ? `<ul class="cert-left">${left.map(c => `<li><a href="${esc(c.slug)}.html">${c.number}. ${esc(c.title)}</a></li>`).join('')}</ul>`
        : '';
      return `<li class="cert-part" style="--part:${esc(part.color)};--part-tint:${esc(part.tint)}">
        <div class="cert-part-head"><h3>Part ${part.number}: ${esc(part.title)}</h3><span class="count">${part.complete ? 'all done' : `${part.done} of ${part.chapters.length} done`}</span></div>
        <ol class="cert-dots">${dots}</ol>${todo}
      </li>`;
    }).join('');
    byId('cert-progress').hidden = false;
  }

  function renderOverview(state) {
    let angle = 0;
    const stops = [];
    state.parts.forEach(part => {
      const share = 360 * part.chapters.length / state.total, filled = share * part.done / part.chapters.length;
      stops.push(`${part.color} ${angle}deg ${angle + filled}deg`, `${part.tint} ${angle + filled}deg ${angle + share}deg`);
      angle += share;
    });
    byId('cert-ring').style.background = `conic-gradient(${stops.join(', ')})`;
    byId('cert-count').textContent = `${state.done}/${state.total}`;
    const left = state.total - state.done;
    byId('cert-headline').textContent = state.complete ? 'You finished the whole book.' : state.done ? `${left} ${left === 1 ? 'chapter' : 'chapters'} to go` : 'Your certificate is waiting';
    const quiz = state.stars ? ` You have ${state.stars} perfect quiz ${state.stars === 1 ? 'score' : 'scores'} so far.` : '';
    byId('cert-summary').textContent = state.complete
      ? `All ${state.total} chapters, start to finish.${quiz} Type the name you want on your certificate.`
      : `A chapter counts as finished when you reach the end of it.${quiz} Finish all ${state.total} and your certificate unlocks here.`;
    byId('cert-lede').hidden = state.complete;
    byId('cert-overview').hidden = false;
  }

  function refresh() {
    const state = tally();
    renderOverview(state);
    renderProgress(state);
    byId('cert-make').hidden = !state.complete;
    const name = String(Book.store.get('name', '') || '');
    if (!state.complete) return showCertificate(state, '');
    byId('cert-name').value = name;
    byId('cert-submit').textContent = name ? 'Update my certificate' : 'Make my certificate';
    byId('cert-progress-title').textContent = 'Every chapter, finished';
    if (name) return showCertificate(state, name);
    byId('cert-sheet').hidden = true;
    byId('cert-save').hidden = true;
    return Promise.resolve();
  }

  byId('cert-form').addEventListener('submit', e => {
    e.preventDefault();
    const name = byId('cert-name').value.replace(/\s+/g, ' ').trim().slice(0, 40);
    if (!name) return;
    Book.store.set('name', name);
    refresh().then(() => {
      byId('cert-sheet').scrollIntoView({ block: 'nearest' });
      Book.toast('Certificate ready');
    });
  });

  byId('cert-download').addEventListener('click', () => {
    if (!imageUrl) return;
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = 'neuro-13-certificate.png';
    document.body.appendChild(link);
    link.click();
    link.remove();
  });

  byId('cert-print').addEventListener('click', () => window.print());

  refresh();
})();
