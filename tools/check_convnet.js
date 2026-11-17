#!/usr/bin/env node
/*
Check the JavaScript that runs the networks in the chapter "How networks see"
(src/widgets/convolution.js) against PyTorch, using py/convnet-weights.json.

    node tools/check_convnet.js [--all] [folder with the MNIST test files]

Always: the scores for the ten sample digits must match the ones PyTorch
wrote into the weights file.
With the MNIST test files (default data/MNIST/raw): the counts in the shift
test must match PyTorch's, and digits blown up to drawing-pad size must come
back from tidy() close to the originals. The convolutional network is slow in
plain JavaScript, so without --all it is checked on the chapter's table
(pushed right, 0 to 6 pixels) plus 4 pixels in the other three directions.
*/
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const root = path.resolve(__dirname, '..');
const net = require(path.join(root, 'src/widgets/convolution.js'));
const file = JSON.parse(fs.readFileSync(path.join(root, 'py/convnet-weights.json'), 'utf8'));
const model = net.load(file);
const run = { plain: x => net.runPlain(model.plain, x), conv: x => net.runConv(model.conv, x) };
const argmax = a => a.indexOf(Math.max(...a));
let failed = false;

function report(ok, text) {
  if (!ok) failed = true;
  console.log((ok ? 'ok    ' : 'FAIL  ') + text);
}

let worst = 0;
for (const kind of ['plain', 'conv']) {
  for (const [digit, expected] of Object.entries(file.check[kind])) {
    const scores = run[kind](model.samples[digit]).scores;
    expected.forEach((want, i) => { worst = Math.max(worst, Math.abs(want - scores[i])); });
  }
}
report(worst < 1e-3, `sample digits: largest difference between a JavaScript score and a PyTorch score = ${worst.toExponential(2)}`);

const dir = process.argv.slice(2).find(arg => arg !== '--all') || path.join(root, 'data/MNIST/raw');
const imagesPath = path.join(dir, 't10k-images-idx3-ubyte.gz');
if (!fs.existsSync(imagesPath)) {
  console.log(`skipped the full test set (no ${imagesPath})`);
  process.exit(failed ? 1 : 0);
}
const pixels = zlib.gunzipSync(fs.readFileSync(imagesPath)).subarray(16);
const labels = zlib.gunzipSync(fs.readFileSync(path.join(dir, 't10k-labels-idx1-ubyte.gz'))).subarray(8);
const digits = Array.from(labels, (_, i) => Float32Array.from(pixels.subarray(i * 784, (i + 1) * 784), v => v / 255));

const DIRECTIONS = { right: [1, 0], left: [-1, 0], down: [0, 1], up: [0, -1] };
const everything = process.argv.includes('--all');
for (const kind of ['plain', 'conv']) {
  const mismatches = [];
  let checked = 0;
  for (const [name, [dx, dy]] of Object.entries(DIRECTIONS)) {
    file.measured[kind][name].forEach((want, k) => {
      if (!everything && kind === 'conv' && name !== 'right' && k !== 4) return;
      let right = 0;
      digits.forEach((x, i) => { if (argmax(run[kind](net.shift(x, dx * k, dy * k)).probs) === labels[i]) right++; });
      if (right !== want) mismatches.push(`${name} ${k}px: JavaScript ${right}, PyTorch ${want}`);
      checked++;
    });
  }
  const unshifted = file.measured[kind].right[0];
  report(!mismatches.length, `${kind}: ${checked} of the 28 shift-test counts checked against PyTorch, ${checked - mismatches.length} identical (unshifted: ${unshifted} of ${labels.length})` +
    (mismatches.length ? '\n        ' + mismatches.join('\n        ') : ''));
}

const SCALE = 10, BIG = 28 * SCALE;
function blowUp(x) {
  const big = new Float32Array(BIG * BIG);
  for (let r = 0; r < BIG; r++) for (let c = 0; c < BIG; c++) big[r * BIG + c] = x[Math.floor(r / SCALE) * 28 + Math.floor(c / SCALE)];
  return big;
}
const SOME = 2000;
let drift = 0;
const right = { plain: 0, conv: 0 }, original = { plain: 0, conv: 0 };
for (let i = 0; i < SOME; i++) {
  const again = net.tidy(blowUp(digits[i]), BIG);
  let diff = 0;
  for (let p = 0; p < 784; p++) diff += Math.abs(again[p] - digits[i][p]);
  drift += diff / 784;
  for (const kind of ['plain', 'conv']) {
    if (argmax(run[kind](again).probs) === labels[i]) right[kind]++;
    if (argmax(run[kind](digits[i]).probs) === labels[i]) original[kind]++;
  }
}
const close = kind => Math.abs(right[kind] - original[kind]) <= SOME * 0.01;
report(close('plain') && close('conv'),
  `tidy(): ${SOME} test digits blown up ${SCALE}x and tidied again: mean pixel difference ${(drift / SOME).toFixed(4)}; ` +
  `plain ${right.plain} right (originals ${original.plain}), conv ${right.conv} right (originals ${original.conv})`);

process.exit(failed ? 1 : 0);
