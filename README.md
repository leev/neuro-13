# Neuro 13

A hands-on book that teaches a curious 13-year-old how neural networks actually work. Every idea comes with something to play with, and every model in the book is really trained, in the reader's browser, on real numbers.

Live at https://neuro-13.leev.net (deployed from `main` by GitHub Actions to a Cloudflare Worker; see `wrangler.toml`).

## Commands

```
make build     build the site into public/
make serve     build, then serve it at http://localhost:8000
make smoke     build, then load every page in headless Chrome and report JavaScript errors
make run-all   run every Python exercise that needs only numpy / scikit-learn
make deploy    build and deploy (needs wrangler login)
```

Preview through `make serve`, not by opening files: widgets and the in-browser Python fetch datasets.

## Layout

```
src/book.json             parts, chapters and extra pages, in reading order
src/chapters/<slug>.html  one chapter: prose and widget markup (a fragment, no <html>)
src/widgets/<slug>.js     that chapter's interactive code (optional)
src/widgets/<slug>.css    that chapter's extra styles (optional)
src/pages/<slug>.html     unnumbered pages (certificate, grown-ups)
src/assets/               shell.css, shell.js, widgets.css, shared.js, terminal.js, glossary.js
src/glossary.json         term -> definition; src/glossary.d/*.json are merged in
src/*.html                page templates and partials
py/                       Python exercises and the datasets they read
tools/                    dataset builders and the smoke test
build.py                  src/ + py/ -> public/
```

`python3 build.py --draft --out DIR` builds to another directory and tolerates chapters that do not exist yet.

## Writing a chapter

### Who it is for

A bright 13-year-old who has never seen this before, reading on their own. Write to "you". Short paragraphs, one idea each. Concrete before abstract: start from something in their life, then name the idea. American spelling.

- **Bold a new term the first time it appears and define it in the same sentence.** Add it to the glossary.
- **Every number in the prose must come from running the code.** If a widget's result varies between runs, say so and give the range.
- **Label anything illustrative.** If numbers are hand-written or a model is pre-trained rather than trained live, the chapter says so plainly.
- **Say when you are simplifying.** Use a short "Honesty note" paragraph rather than a quiet half-truth.
- No hype and no doom. Explain what the thing does and let that be impressive.
- For every widget: say what to do and what to watch for, then afterwards explain what happened and why.

### Anatomy

The build supplies the chapter number, title and hook from `book.json`, so a chapter file starts straight in with prose. Sections use `<h3>`. A typical chapter runs: opening hook → sections → a "predict first" prompt → the widget → what just happened → the big idea → a challenge → a three-question quiz.

Refer to other chapters with `{{ch:slug}}` (renders as a link reading "chapter 7"), `{{Ch:slug}}` to start a sentence, or `{{title:slug}}` to link by title. Never type a chapter number.

### Components

```html
<div class="predict" data-answer="b">
  <p>Before you press train: what will the shallow network do with the spiral?</p>
  <button data-opt="a">Solve it, just more slowly</button>
  <button data-opt="b">Get stuck around a coin flip</button>
  <p class="why">Shown once they pick. Omit data-answer when there is no wrong guess.</p>
</div>

<div class="interactive">
  <div class="interactive-title">▸ try it</div>
  <h3>Name of the widget</h3>
  <p>What to do and what to watch.</p>
  <!-- controls: <button>, <button class="secondary">, .toggle-row > .toggle(.on),
       .slider-row > label + input[type=range] + span.value, .stat-row > span > strong,
       .math-panel > h4, .bar-container > .label + .bar-track > .bar + .pct,
       .token-row > .token-pill > .token-id, pre.model-dump -->
</div>

<div class="bigidea"><p>One or two sentences a reader should still know next month.</p></div>

<div class="challenge"><strong>Challenge</strong><p>Something to try that has no button.</p></div>

<div class="quiz">
  <div class="q" data-answer="c">
    <p>Question?</p>
    <button data-opt="a">…</button><button data-opt="b">…</button><button data-opt="c">…</button>
    <p class="why">Why the right answer is right.</p>
  </div>
</div>

<div class="callout">A warning or surprising fact.</div>
```

Quiz questions test the idea, not recall of a number. Wrong options should be things a reader might genuinely think.

### Widgets

`src/widgets/<slug>.js` is loaded after `shell.js` and `shared.js` on that chapter's page only.

- One named IIFE per widget: `(function nameWidget() { … })();`. No globals. Prefix every element id with a short tag unique to the widget (`dm-…`).
- Train for real, in the browser, from random weights. Keep a run under about 15 seconds on a laptop, and yield to the page between epochs (`await new Promise(r => setTimeout(r, 0))`) so the display updates and the tab never freezes.
- Use `Book.rng(seed)` when a demo must reproduce exactly; plain `Math.random()` when variety between runs is part of the lesson.
- Canvases: declare logical size with `width`/`height` attributes, then `const { ctx, w, h } = Book.hidpi(canvas)` and draw in those logical units. Map pointer events with `Book.pointer(canvas, event)`.
- Line charts: `Book.lineChart(canvas, { series: [{ label, color, data }], xMax, yMin, yMax, yFormat, xLabel })`. It returns `{ ctx, w, h, left, right, top, bottom, X, Y }` so you can draw markers on top without copying its margins.
- `Book.toast(text)` for a brief message. `shared.js` provides `sigmoid`, `makeModelDump`, `makeSingleTrainer`, `makeSeparator`.
- Fetch datasets from `py/<file>`. Handle the failure case with a visible message.
- Widget colors: `var(--accent)` blue, `var(--good)`, `var(--bad)`, `var(--warn)`; `var(--part)` is the current part's color.
- Test the numerical core outside the browser first (node or numpy), including a numerical gradient check for any hand-written backward pass.

### Python exercises

Scripts live in `py/` and must run both locally (`cd py && python3 name.py`) and in the browser through Pyodide, using only numpy (scikit-learn is available but slow to load). A script is teaching material: a docstring that says what it is and what is new, then code a 13-year-old can follow top to bottom. Print progress as it trains.

```html
<div class="interactive">
  <div class="interactive-title">▸ run real python</div>
  <h3>Run it</h3>
  <p>What this script is and what to look for in the output.</p>
  <div class="code-editor-wrap">
    <div class="code-header">
      <span class="code-filename">name.py</span>
      <span class="code-status" id="py-status-name">Python loads on first run</span>
    </div>
    <textarea class="code-editor" id="py-code-name" spellcheck="false" rows="30" data-src="name.py" data-files="data.csv"></textarea>
  </div>
  <div style="margin: 10px 0;">
    <button data-py="name">▶ run</button>
    <button class="secondary" data-py-reset="name">reset code</button>
  </div>
  <div class="terminal" id="py-out-name"><span class="dim">(click "run" to execute)</span></div>
</div>
```

The build fills the textarea from `py/<data-src>`. `data-files` lists datasets (in `py/`) to make available before the script runs.

### Glossary

Add terms in `src/glossary.d/<slug>.json` as `{"term": "one plain sentence"}`. The reader sees them as dotted-underlined words with a hover definition, and on the glossary page.

Underlines are added to prose automatically, including text a widget writes at runtime. They are never added inside SVG, code, buttons, links, headings, the terminal or the code editor. Put `data-no-gloss` on any other container whose text must be left alone.

### Before you call it done

```
python3 build.py --draft --out /tmp/site && tools/smoke.sh /tmp/site
```

Then open the page and use every control. A chapter is not finished until its widget has been watched doing what the prose says it does.

## Datasets

- `py/headlines.csv`: 10,000 labelled news stories, a cleaned and balanced sample of the AG News research collection (2004). Rebuild with `tools/make_headlines.py`.
- `py/digits.bin`: 8,000 training and 2,000 test handwritten digits sampled from MNIST (Yann LeCun, Corinna Cortes, Christopher J.C. Burges; CC BY-SA 3.0). Rebuild with `python3 tools/make_digits.py data/MNIST/raw py/digits.bin` (the raw MNIST files are not in the repo; `py/neuro9.py` or `py/convnet.py` will download them into `data/`).
- `py/names.txt`: 8,020 first names, the 4,300 most common per sex from the names file in karpathy/makemore (MIT-licensed repo; derived from public-domain US Social Security Administration baby-name records). Rebuild with `tools/make_names.py`.
- `py/dinosaurs.txt`: 1,389 dinosaur genus names from Wikipedia's "List of dinosaur genera" (CC BY-SA 4.0, fetched 2026-11-17). Also built by `tools/make_names.py`.
- `py/convnet-weights.json`: both networks used in the convolution chapter, trained on MNIST by `py/convnet.py` (needs PyTorch). The file embeds ten MNIST test digits. If you retrain, update the measured tables in `src/chapters/convolution.html`; `node tools/check_convnet.js` checks the JavaScript forward pass against PyTorch.
- `py/diffusion-weights.json`: the noise-predicting network used in the diffusion chapter, trained by `py/diffusion.py` on shapes the script draws itself. No outside data.

Figures quoted in the text that come from outside sources are named where they appear (for example Meta's Llama 3.1 model card and the US Energy Information Administration's 2022 household electricity figure in the scale chapter).

## Design assets

- Fonts in `src/assets/fonts/` are self-hosted under the SIL Open Font License (`OFL.txt`): Bricolage Grotesque, Figtree and Atkinson Hyperlegible Mono.
- `src/assets/og.png` (the link-preview image) is rendered from `tools/og.html` by `tools/make_og.sh`.
- The five part colors live in `src/book.json` and are mirrored in `src/assets/shell.css`, `src/assets/favicon.svg` and `tools/og.html`. If you change them, update all four and rerun `tools/make_og.sh`.
- `--accent` (blue) is reserved for widgets so they look the same in every part; `--good` and `--warn` are dark enough to pass AA contrast as text.
