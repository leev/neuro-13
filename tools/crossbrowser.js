// Drive every widget the same way in Chromium, Firefox and WebKit and check that the text they show is identical.
//   node tools/crossbrowser.js [site-dir] [--native] [--only slug,slug] [--engines chromium,webkit] [--save file.json]
// --native blocks pmath.js, which shows how far apart the browsers are without it. Setup is in the README.
const fs = require('fs'), path = require('path'), http = require('http');
const pw = require('playwright-core');

const args = process.argv.slice(2);
const flag = name => { const i = args.indexOf(name); return i < 0 ? null : args.splice(i, 2)[1]; };
const only = (flag('--only') || '').split(',').filter(Boolean), save = flag('--save');
const native = args.includes('--native');
const engines = (flag('--engines') || 'chromium,firefox,webkit').split(',');
const dir = path.resolve(args.find(a => !a.startsWith('--')) || 'public');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.woff2': 'font/woff2', '.svg': 'image/svg+xml' };

const server = http.createServer((req, res) => {
  const file = path.join(dir, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(file, (err, data) => {
    res.writeHead(err ? 404 : 200, { 'content-type': TYPES[path.extname(file)] || 'application/octet-stream' });
    res.end(err ? '' : data);
  });
});

// How long a snapshot takes to settle differs between browsers, and so does how many timer ticks draw a random
// number meanwhile. Restarting the sequence at every press keeps what follows a press comparable.
function prepare() {
  let a = 0;
  window.__reseed = () => { a = 20241118; };
  window.__textOf = box => {
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT), parts = [];
    while (walker.nextNode()) {
      if (!walker.currentNode.parentElement.closest('select, script, style')) parts.push(walker.currentNode.nodeValue.trim());
    }
    return parts.filter(Boolean).join(' ').replace(/\d+\.\d s\b/g, 'N s');
  };
  window.__reseed();
  Math.random = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function countControls() {
  const skip = '[data-py], [data-py-reset], .quiz *, .predict *, .CodeMirror *, a';
  window.__controls = [...document.querySelectorAll('.interactive button, .interactive .toggle')].filter(el => !el.matches(skip));
  return window.__controls.length;
}

function press(i) {
  const el = window.__controls[i];
  if (!el.isConnected || el.disabled) return null;
  window.__reseed();
  el.click();
  return true;
}

function read(i) {
  const box = window.__controls[i].closest('.interactive');
  return box ? window.__textOf(box) : '';
}

function readAll() {
  return [...document.querySelectorAll('.interactive')].map(window.__textOf).join('\n');
}

const QUIET_MS = 2500, GIVE_UP_MS = 60000;

// Timers run on a fake clock, but downloads and time-paced training loops take real time, so a snapshot
// waits until no widget on the page has changed its text for QUIET_MS. Returns null if that never happens.
async function settled(page, i) {
  const began = Date.now();
  let seen = null, since = began;
  while (Date.now() - began < GIVE_UP_MS) {
    await page.clock.runFor(1000);
    await page.waitForTimeout(150);
    const now = await page.evaluate(readAll);
    if (now !== seen) {
      seen = now;
      since = Date.now();
    } else if (Date.now() - since >= QUIET_MS) {
      return page.evaluate(read, i);
    }
  }
  return null;
}

async function drive(engine, port, pages) {
  const browser = await pw[engine].launch(), result = {};
  for (const slug of pages) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    await context.addInitScript(prepare);
    if (native) await context.route('**/pmath.js*', route => route.fulfill({ contentType: 'text/javascript', body: '' }));
    const page = await context.newPage(), errors = [], shots = [];
    page.on('pageerror', e => errors.push(String(e.message)));
    await page.clock.install({ time: new Date('2026-01-01T12:00:00Z') });
    const open = async () => {
      await page.goto(`http://localhost:${port}/${slug}.html`);
      await page.clock.runFor(2000);
      await page.waitForTimeout(300);
      return Math.min(await page.evaluate(countControls), 60);
    };
    const n = await open();
    for (let i = 0; i < n; i++) {
      if (!(await page.evaluate(press, i))) {
        shots.push('(disabled)');
        continue;
      }
      const text = await settled(page, i);
      shots.push(text === null ? '(never settles)' : text);
      // An animation that never stops would leave each browser at a different frame, so start the page again.
      if (text === null) await open();
    }
    result[slug] = { shots, errors };
    await context.close();
    process.stderr.write(`${engine} ${slug}: ${shots.length} controls${errors.length ? ', ERRORS ' + errors.join(' | ') : ''}\n`);
  }
  await browser.close();
  return result;
}

server.listen(0, async () => {
  const port = server.address().port;
  const pages = fs.readdirSync(dir).filter(f => f.endsWith('.html')).map(f => f.slice(0, -5)).filter(s => !only.length || only.includes(s));
  const results = {};
  await Promise.all(engines.map(async engine => { results[engine] = await drive(engine, port, pages); }));
  if (save) fs.writeFileSync(save, JSON.stringify(results));
  let failed = 0;
  for (const slug of pages) {
    const base = results[engines[0]][slug], problems = [];
    for (const engine of engines) {
      const r = results[engine][slug];
      if (r.errors.length) problems.push(`${engine} threw: ${r.errors[0]}`);
      const at = r.shots.findIndex((s, i) => s !== base.shots[i]);
      if (at >= 0 || r.shots.length !== base.shots.length) {
        const a = base.shots[at] || '', b = r.shots[at] || '';
        let k = 0;
        while (k < a.length && a[k] === b[k]) k++;
        problems.push(`${engine} differs from ${engines[0]} after control ${at}: "${a.slice(Math.max(0, k - 40), k + 40)}" vs "${b.slice(Math.max(0, k - 40), k + 40)}"`);
      }
    }
    if (problems.length) failed++;
    const endless = base.shots.filter(s => s === '(never settles)').length;
    console.log(`${problems.length ? 'FAIL' : 'ok  '} ${slug}${endless ? `  (${endless} endless animations not compared)` : ''}${problems.map(p => '\n       ' + p).join('')}`);
  }
  server.close();
  process.exit(failed ? 1 : 0);
});
