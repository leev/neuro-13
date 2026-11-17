(function dictionaryWidget() {
  const BASE = 256;
  const CHUNK = / ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu;
  const utf8 = new TextEncoder();
  const pairKey = (a, b) => a * 65536 + b;

  function mergePair(ids, a, b, id) {
    let i = 0;
    while (i + 1 < ids.length && !(ids[i] === a && ids[i + 1] === b)) i++;
    if (i + 1 >= ids.length) return ids;
    const out = ids.slice(0, i);
    for (; i < ids.length; i++) {
      if (ids[i] === a && ids[i + 1] === b) { out.push(id); i++; }
      else out.push(ids[i]);
    }
    return out;
  }

  // Ties go to the pair that appears first in the text: a later pair only moves ahead
  // of an earlier one when its count is strictly higher.
  function trainBPE(text, maxMerges) {
    const seen = new Map();
    for (const chunk of text.match(CHUNK) || []) seen.set(chunk, (seen.get(chunk) || 0) + 1);
    const words = [...seen].map(([chunk, n]) => ({ ids: [...utf8.encode(chunk)], n }));
    const merges = [], boards = [];
    const size = BASE + maxMerges;
    const counts = new Int32Array(size * size);
    for (;;) {
      const firstSeen = [];
      for (const w of words) {
        for (let i = 0; i + 1 < w.ids.length; i++) {
          const key = w.ids[i] * size + w.ids[i + 1];
          if (counts[key] === 0) firstSeen.push(key);
          counts[key] += w.n;
        }
      }
      const top = [];
      for (const key of firstSeen) {
        const count = counts[key];
        counts[key] = 0;
        let at = top.length;
        while (at > 0 && top[at - 1].count < count) at--;
        if (at < 5) {
          top.splice(at, 0, { a: Math.floor(key / size), b: key % size, count });
          if (top.length > 5) top.pop();
        }
      }
      boards.push(top);
      const best = top[0];
      if (!best || best.count < 2 || merges.length >= maxMerges) break;
      const id = BASE + merges.length;
      merges.push(best);
      for (const w of words) w.ids = mergePair(w.ids, best.a, best.b, id);
    }
    const rank = new Map(merges.map((m, i) => [pairKey(m.a, m.b), i]));
    const bytes = [];
    for (let id = 0; id < BASE; id++) bytes.push([id]);
    for (const m of merges) bytes.push(bytes[m.a].concat(bytes[m.b]));
    return { merges, boards, rank, bytes };
  }

  function encodeBPE(dict, text, limit) {
    const out = [];
    for (const chunk of text.match(CHUNK) || []) {
      let ids = [...utf8.encode(chunk)];
      while (ids.length > 1) {
        let first = Infinity;
        for (let i = 0; i + 1 < ids.length; i++) {
          const r = dict.rank.get(pairKey(ids[i], ids[i + 1]));
          if (r !== undefined && r < limit && r < first) first = r;
        }
        if (first === Infinity) break;
        ids = mergePair(ids, dict.merges[first].a, dict.merges[first].b, BASE + first);
      }
      for (const id of ids) out.push(id);
    }
    return out;
  }

  function decodeBPE(dict, ids) {
    return new TextDecoder().decode(new Uint8Array(ids.flatMap(id => dict.bytes[id] || [])));
  }

  const $ = id => document.getElementById(id);
  if (!$('bpe-once')) return;

  const STORY = [
    'Once upon a time, in a cave high in the mountains, there lived a small green dragon named Ember. The other dragons liked to guard their gold and scare the sheep. Ember did not care about gold, and she thought the sheep were rather nice. The thing Ember loved more than anything else in the whole world was pizza.',
    'Every morning she flew down to the village with a basket in her claws. The baker gave her flour, the farmer gave her tomatoes, and the cheese maker gave her the biggest wheel of cheese in the shop. In return, Ember lit their ovens and their lanterns with one careful puff of fire.',
    'Back in the cave, Ember mixed the dough and rolled it flat with her tail. She spread the tomato sauce and sprinkled the cheese, and then came the best part. She took a deep breath and blew a thin, steady flame over the pizza until the cheese bubbled and the crust turned golden brown. Nothing in the kingdom could bake a pizza faster than a dragon.',
    'One spring evening the farmer came running up the mountain path. "Ember," he said, "a frost is coming tonight, and it will ruin every strawberry in my field. Can you help me?" Ember had never thought much about strawberries. Still, a friend was a friend. All night long she walked up and down the rows, breathing warm air over the little plants. In the morning the frost was gone, and every single strawberry was safe.',
    'The farmer was so happy that he gave her ten baskets of strawberries. Ember looked at the strawberries, and then she looked at her pizza, and then she had the strangest idea that any dragon has ever had. She put the strawberries on the pizza.',
    'The baker laughed. The cheese maker shook his head. The farmer said it was the silliest thing he had ever seen. Then each of them took a bite, and nobody said anything for a long time, because they were all too busy eating.',
    'After that, people walked for days to visit the cave in the mountains. They came for the mushroom pizza and the pineapple pizza, and they came for the blueberry pizza and the raspberry pizza. Most of all they came for the strawberry pizza. Ember gave everyone a slice, a paper straw and a cup of cold berry juice, and she never asked for a single coin.',
    'When the other dragons heard about it, they flew over the mountains to see for themselves. They were sure that Ember had gone mad. A dragon was meant to be frightening, they said, and there was nothing frightening about a dragon in an apron. Ember did not argue with them. She handed each of them a slice and waited.',
    'The biggest dragon sniffed it. Then he tasted it. Then he asked, in a very small voice, whether there was any more. By the end of the week the other dragons were chopping mushrooms, picking berries and carrying water from the river. By the end of the month there was a long wooden table outside the cave, and the dragons and the villagers were sitting at it together.',
    'That is how the mountain got the best pizza shop in the kingdom. The villagers bring the flour, the cheese and the tomatoes. The farmer brings the strawberries. The dragons bring the fire. And when the first frost of the winter is coming, everyone in the village knows who is going to be walking up and down the strawberry field all night long, keeping the little plants warm.',
    'If you ever visit, remember three things. Bring something to share. Say thank you to the dragon who is doing the baking. And try the strawberry pizza before you say anything about it.',
  ].join('\n\n');
  const MAX_MERGES = 600;
  const TINTS = 5;
  const ROWS_SHOWN = 8;
  const NEWEST_MARKED = 10;
  const strict = new TextDecoder('utf-8', { fatal: true });
  const slider = $('bpe-n');

  let dict, text, n = 0, freshFrom = Infinity, timer = null;

  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const hex = id => dict.bytes[id].map(b => '0x' + b.toString(16).toUpperCase().padStart(2, '0')).join(' ');
  const fmt = x => x.toLocaleString('en-US');
  const isFresh = id => id >= freshFrom && id < BASE + n;

  function pieceText(id) {
    try {
      const s = strict.decode(new Uint8Array(dict.bytes[id]));
      return /[\u0000-\u0008\u000b-\u001f\u007f]/.test(s) ? null : s;
    } catch (err) {
      return null;
    }
  }

  function label(id) {
    const s = pieceText(id);
    return s === null ? hex(id) : s.replace(/ /g, '·').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
  }

  function pill(id, withId, markFresh = true) {
    const cls = 'token-pill' + (pieceText(id) === null ? ' raw' : '') + (markFresh && isFresh(id) ? ' fresh' : '') + (withId ? '' : ' mini');
    return `<code class="${cls}">${esc(label(id))}${withId ? `<span class="token-id">id ${id}</span>` : ''}</code>`;
  }

  function renderBoard() {
    const board = dict.boards[n];
    const finished = n === dict.merges.length;
    const scale = dict.boards[0].length ? dict.boards[0][0].count : 1;
    $('bpe-board').innerHTML = board.length ? board.map((p, i) =>
      `<div class="bpe-pair${i === 0 && !finished ? ' next' : ''}">` +
      `<span class="bpe-parts">${pill(p.a)}<span class="bpe-plus">+</span>${pill(p.b)}</span>` +
      `<div class="bar-track"><div class="bar" style="width:${Math.max(1, 100 * p.count / scale)}%"></div></div>` +
      `<span class="pct">${fmt(p.count)}×</span></div>`).join('')
      : '<p class="bpe-note">There is no text to count pairs in.</p>';

    let note;
    if (n === 0) {
      note = 'Nothing merged yet. Every character is its own token, and the dictionary is just the 256 possible bytes.';
    } else {
      const m = dict.merges[n - 1];
      note = `Merge ${n}: ${pill(m.a)} + ${pill(m.b)} became ${pill(BASE + n - 1)} with the new id <strong>${BASE + n - 1}</strong>. That pair was sitting side by side ${fmt(m.count)} times.`;
    }
    if (finished && n > 0) {
      note += n >= MAX_MERGES
        ? ` This widget stops at ${MAX_MERGES} merges.`
        : ' <strong>Finished:</strong> no pair of neighbors appears twice any more, so there is nothing left worth gluing.';
    }
    $('bpe-last').innerHTML = note;
  }

  function renderStory() {
    const ids = encodeBPE(dict, text, n);
    const pre = document.createElement('pre');
    pre.innerHTML = ids.map((id, i) => {
      const s = pieceText(id);
      return `<span class="t${i % TINTS}${isFresh(id) ? ' fresh' : ''}${s === null ? ' raw' : ''}">${esc(s === null ? hex(id) : s)}</span>`;
    }).join('');
    $('bpe-story').replaceChildren(pre);
    $('bpe-story-chars').textContent = fmt([...text].length);
    $('bpe-story-tokens').textContent = fmt(ids.length);
  }

  function renderVocab() {
    $('bpe-vocab-count').textContent = fmt(n);
    if (!$('bpe-vocab-box').open) return;
    let html = '';
    for (let id = BASE + n - 1; id >= BASE; id--) html += pill(id, true);
    $('bpe-vocab-list').innerHTML = html || '<span class="bpe-note">No learned pieces yet. Press “merge once”.</span>';
  }

  function describe(id) {
    if (id >= BASE) return `“<strong>${esc(label(id))}</strong>”`;
    if (id === 32) return 'a space (drawn here as <strong>·</strong>)';
    if (id === 10) return 'a line break';
    if (id === 9) return 'a tab';
    if (id < 32 || id === 127) return `a control code (byte ${hex(id)}), not something you can see`;
    if (id >= 128) return `byte ${hex(id)}: one part of a bigger character such as é or an emoji`;
    return `“<strong>${esc(label(id))}</strong>”`;
  }

  function renderReverse() {
    const v = parseInt($('tk-rev').value, 10);
    const top = BASE + n - 1;
    let msg;
    if (isNaN(v)) msg = '→ type a number';
    else if (v >= 0 && v <= top) msg = '→ ' + describe(v);
    else if (v > top && v < BASE + dict.merges.length) msg = `→ no row ${v} yet. Your dictionary stops at ${top}: merge some more above.`;
    else msg = `→ no row ${v}. Your dictionary only goes from 0 to ${top}.`;
    $('tk-rev-out').innerHTML = msg;
  }

  function renderSentence() {
    const s = $('bpe-input').value;
    const ids = encodeBPE(dict, s, n);
    const chars = [...s].length;
    const empty = '<span class="bpe-note">(type something)</span>';
    $('bpe-pills').innerHTML = ids.map(id => pill(id, true)).join('') || empty;
    $('tk-pills').innerHTML = ids.map(id => pill(id, true, false)).join('') || empty;
    $('bpe-chars').textContent = fmt(chars);
    $('bpe-tokens').textContent = fmt(ids.length);

    const firstWord = (s.match(CHUNK) || [])[0] || '';
    let hint = '';
    if (firstWord && !/^\s/.test(firstWord)) {
      const bare = encodeBPE(dict, firstWord, n).length;
      const spaced = encodeBPE(dict, ' ' + firstWord, n).length;
      if (spaced < bare) {
        hint = `Your first word has no space in front of it, so it is chopped as a different string: “${firstWord}” takes ${bare} tokens here, but “·${firstWord}” in the middle of a sentence would take ${spaced}.`;
      }
    }
    $('bpe-hint').textContent = hint;

    $('tk-size').textContent = fmt(BASE + n);
    $('tk-ids').innerHTML = `<span class="cmt">${fmt(ids.length)} token${ids.length === 1 ? '' : 's'} from ${fmt(chars)} character${chars === 1 ? '' : 's'}</span>\n` +
      `input_ids = [${ids.join(', ')}]`;

    const rows = ids.slice(0, ROWS_SHOWN).map(id => {
      const random = Book.rng(9001 + id * 7919);
      let squares = '';
      for (let k = 0; k < 16; k++) {
        const v = random() * 2 - 1;
        squares += `<div class="embed-square" style="background:rgba(${v >= 0 ? '37, 99, 235' : '220, 38, 38'}, ${Math.abs(v).toFixed(2)})"></div>`;
      }
      return `<div class="embed-block"><code class="name">${esc(label(id))} <span class="tk-rowid">row ${id}</span></code><div class="row"><div class="embed-row">${squares}</div></div></div>`;
    }).join('');
    $('tk-embeds').innerHTML = rows + (ids.length > ROWS_SHOWN
      ? `<p class="bpe-note">(showing the first ${ROWS_SHOWN} of ${ids.length} rows)</p>` : '');

    const back = decodeBPE(dict, ids);
    $('tk-roundtrip').innerHTML = ids.length
      ? `Read every id back through the dictionary and glue the pieces together: <strong>“${esc(back)}”</strong>` +
        (back === s ? ', exactly what you typed.' : '')
      : '';
  }

  function render() {
    const max = dict.merges.length;
    slider.value = n;
    $('bpe-n-v').textContent = n;
    $('bpe-vocab').textContent = fmt(BASE + n);
    $('bpe-learned').textContent = fmt(n);
    for (const id of ['bpe-once', 'bpe-ten', 'bpe-hundred']) $(id).disabled = n >= max;
    $('bpe-play').disabled = n >= max && !timer;
    $('bpe-reset').disabled = n === 0;
    renderBoard();
    renderStory();
    renderVocab();
    renderSentence();
    renderReverse();
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    $('bpe-play').textContent = '▶ play';
  }

  function setMerges(target) {
    const next = Math.max(0, Math.min(dict.merges.length, target));
    freshFrom = next > n ? BASE + Math.max(n, next - NEWEST_MARKED) : Infinity;
    n = next;
    if (timer && n >= dict.merges.length) stop();
    render();
  }

  function learn(raw) {
    stop();
    text = raw.trim().replace(/\s+/g, ' ');
    dict = trainBPE(text, MAX_MERGES);
    n = 0;
    freshFrom = Infinity;
    slider.max = Math.max(1, dict.merges.length);
    slider.disabled = dict.merges.length === 0;
    $('bpe-max').textContent = fmt(dict.merges.length);
    $('bpe-words').textContent = fmt((text.match(/\S+/g) || []).length);
    render();
  }

  function setSentence(value, from) {
    if (from !== $('bpe-input')) $('bpe-input').value = value;
    if (from !== $('tk-input')) $('tk-input').value = value;
    renderSentence();
  }

  $('bpe-once').addEventListener('click', () => { stop(); setMerges(n + 1); });
  $('bpe-ten').addEventListener('click', () => { stop(); setMerges(n + 10); });
  $('bpe-hundred').addEventListener('click', () => { stop(); setMerges(n + 100); });
  $('bpe-reset').addEventListener('click', () => { stop(); setMerges(0); });
  $('bpe-play').addEventListener('click', () => {
    if (timer) { stop(); render(); return; }
    $('bpe-play').textContent = '❚❚ pause';
    timer = setInterval(() => setMerges(n + 1), 70);
  });
  slider.addEventListener('input', () => { stop(); setMerges(+slider.value); });
  $('bpe-input').addEventListener('input', e => setSentence(e.target.value, e.target));
  $('tk-input').addEventListener('input', e => setSentence(e.target.value, e.target));
  $('bpe-examples').addEventListener('click', e => {
    const btn = e.target.closest('[data-text]');
    if (btn) setSentence(btn.getAttribute('data-text'), null);
  });
  $('tk-rev').addEventListener('input', renderReverse);
  $('bpe-vocab-box').addEventListener('toggle', renderVocab);
  $('bpe-relearn').addEventListener('click', () => {
    learn($('bpe-corpus').value);
    Book.toast(dict.merges.length ? `Learned ${dict.merges.length} merges from your text` : 'Not enough text to learn from');
  });
  $('bpe-restore').addEventListener('click', () => {
    $('bpe-corpus').value = STORY;
    learn(STORY);
  });

  $('bpe-corpus').value = STORY;
  learn(STORY);
})();
