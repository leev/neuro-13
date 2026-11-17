"""
Pictures from noise: a tiny diffusion model.

This is the idea behind AI image generators, shrunk until you can read all
of it. Our "pictures" are clouds of dots that draw a shape: a smiley, a
heart, a spiral, a star and the word HI.

  1. RUIN     Take a dot from a real picture and mix in random static:
                  noisy dot = keep * real position + noise * static
              There are 50 noise levels. At level 1 the dot barely moves.
              At level 50 almost nothing but static is left.
  2. LEARN    Show the network a ruined dot, its noise level and which shape
              it came from. Ask one question: what static was mixed in?
  3. UN-RUIN  Start from pure static. 50 times over, ask the network for its
              guess and take a little of the guessed static back out.
              A shape appears, made of dots that were never in the training
              pictures.

What is new compared with the earlier scripts:
  - The network's answer is two numbers (the static in x and the static in
    y), not a choice from a list. Its loss is the squared distance between
    its guess and the real static.
  - Each dot's x and y also go in as sin and cos waves at four zoom levels.
    That helps a small network draw thin, sharp lines.
  - The weights are updated with Adam: gradient descent that remembers each
    weight's recent gradients and gives every weight a step size that suits it.

Run:
    python3 diffusion.py           the full run: five shapes, a few minutes.
                                   Writes diffusion-weights.json, the file
                                   the book's web page loads.
    python3 diffusion.py --quick   two shapes and a smaller network. This is
                                   what runs when you press "run" in the book.
"""
import json
import sys

import numpy as np

QUICK = sys.platform == "emscripten" or "--quick" in sys.argv

SHAPES = ["heart", "star"] if QUICK else ["smiley", "heart", "spiral", "star", "hi"]
HIDDEN = 48 if QUICK else 128
TRAIN_STEPS = 4000 if QUICK else 30000
BATCH = 128 if QUICK else 1024
DOTS = 2000
LEVELS = 50
LINE_WIDTH = 0.03
LEARNING_RATE = 0.008 if QUICK else 0.002
WAVES = np.array([1, 2, 4, 8])

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@` on
# matrices this size, even though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")
rng = np.random.default_rng(13)


def circle(cx, cy, r, start=0.0, end=1.0):
    a = np.linspace(start, end, 100) * 2 * np.pi
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def heart_outline():
    t = np.linspace(0, 2 * np.pi, 400)
    return np.stack([16 * np.sin(t) ** 3,
                     13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)], axis=1)


def spiral_arm():
    a = np.linspace(0.35, 5.2, 600) * np.pi
    return np.stack([a * np.cos(a), a * np.sin(a)], axis=1)


def star_outline():
    corner = np.arange(11)
    a = np.pi / 2 + corner * np.pi / 5
    r = np.where(corner % 2 == 0, 1.0, 0.42)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


# A shape is a list of pen strokes. A stroke is a list of (x, y) corners
# joined by straight lines. Any size will do: make_dots rescales it.
PEN_STROKES = {
    "smiley": [circle(0, 0, 1), circle(-0.36, 0.3, 0.13), circle(0.36, 0.3, 0.13),
               circle(0, -0.05, 0.58, 0.575, 0.925)],
    "heart": [heart_outline()],
    "spiral": [spiral_arm()],
    "star": [star_outline()],
    "hi": [[(-1, -0.8), (-1, 0.8)], [(-0.15, -0.8), (-0.15, 0.8)], [(-1, 0), (-0.15, 0)],
           [(0.65, -0.8), (0.65, 0.8)], [(0.3, 0.8), (1, 0.8)], [(0.3, -0.8), (1, -0.8)]],
}


def make_dots(name):
    """Sprinkle DOTS dots evenly along a shape's pen strokes."""
    strokes = [np.asarray(s, dtype=float) for s in PEN_STROKES[name]]
    starts = np.concatenate([s[:-1] for s in strokes])
    ends = np.concatenate([s[1:] for s in strokes])
    lengths = np.linalg.norm(ends - starts, axis=1)
    piece = rng.choice(len(starts), DOTS, p=lengths / lengths.sum())
    p = starts[piece] + rng.uniform(size=(DOTS, 1)) * (ends[piece] - starts[piece])
    p -= (p.max(axis=0) + p.min(axis=0)) / 2
    # Static has a spread of 1. Giving the picture a spread of 1 too keeps a
    # ruined picture the same size while it dissolves.
    p /= np.sqrt((p ** 2).mean())
    return np.round(p + rng.normal(0, LINE_WIDTH, p.shape), 3)


def show(points, title):
    """Print a cloud of dots as a picture made of text."""
    cols, rows, reach = 44, 22, 2.2
    c = np.floor((points[:, 0] + reach) / (2 * reach) * cols).astype(int)
    r = np.floor((reach - points[:, 1]) / (2 * reach) * rows).astype(int)
    inside = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
    grid = np.zeros((rows, cols), dtype=int)
    np.add.at(grid, (r[inside], c[inside]), 1)
    print(f"\n{title}")
    for row in grid:
        print("   " + "".join(" .o#"[v] for v in np.digitize(row, [1, 3, 8])).rstrip())


dots = np.stack([make_dots(name) for name in SHAPES])
print(f"{len(SHAPES)} training pictures of {DOTS} dots each: {', '.join(SHAPES)}")
show(dots[0], f'a training picture: "{SHAPES[0]}"')

# keep falls from 1.00 to 0.03 while noise climbs from 0.00 to 1.00. The
# power of 1.5 makes the first levels gentle, so a picture fades gradually.
angle = np.linspace(0, 1, LEVELS + 1) ** 1.5 * 0.98 * np.pi / 2
keep, noise = np.cos(angle), np.sin(angle)

for level in (20, 35, 50):
    static = rng.normal(size=dots[0].shape)
    show(keep[level] * dots[0] + noise[level] * static,
         f"ruined to level {level}: {keep[level]:.2f} x picture + {noise[level]:.2f} x static")


def features(x, level, label):
    """Everything the network is told about each dot: where it is (plain and
    as waves), how noisy things are, and which shape is wanted."""
    waves = (x[:, :, None] * WAVES).reshape(len(x), -1)
    switches = np.eye(len(SHAPES))[label]
    return np.hstack([x, np.sin(waves), np.cos(waves),
                      noise[level][:, None], keep[level][:, None], switches])


N_INPUTS = 2 + 4 * len(WAVES) + 2 + len(SHAPES)
sizes = [N_INPUTS, HIDDEN, HIDDEN, HIDDEN, 2]
weights = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)) for n_in, n_out in zip(sizes, sizes[1:])]
biases = [np.zeros(n_out) for n_out in sizes[1:]]
params = weights + biases
print(f"\nnetwork: {' -> '.join(map(str, sizes))}, {sum(p.size for p in params):,} weights")


def forward(inputs):
    """Returns the output of every layer; the last one is the guessed static."""
    outputs = [inputs]
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = outputs[-1] @ W + b
        outputs.append(z if i == len(weights) - 1 else np.maximum(0, z))
    return outputs


def backward(outputs, d):
    """Backpropagation. d arrives as the gradient at the network's output."""
    d_weights, d_biases = [], []
    for i in reversed(range(len(weights))):
        d_weights.insert(0, outputs[i].T @ d)
        d_biases.insert(0, d.sum(axis=0))
        if i > 0:
            d = (d @ weights[i].T) * (outputs[i] > 0)
    return d_weights + d_biases


average_grad = [np.zeros_like(p) for p in params]
average_square = [np.zeros_like(p) for p in params]


def adam(grads, step, lr):
    for p, g, m, v in zip(params, grads, average_grad, average_square):
        m += 0.1 * (g - m)
        v += 0.001 * (g * g - v)
        p -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)


print(f"training for {TRAIN_STEPS:,} steps of {BATCH} dots (guessing zero every time would score a loss of 1.0)")
recent = []
for step in range(1, TRAIN_STEPS + 1):
    label = rng.integers(0, len(SHAPES), BATCH)
    real = dots[label, rng.integers(0, DOTS, BATCH)]
    level = rng.integers(1, LEVELS + 1, BATCH)
    static = rng.normal(size=real.shape)
    noisy = keep[level][:, None] * real + noise[level][:, None] * static

    outputs = forward(features(noisy, level, label))
    miss = outputs[-1] - static
    recent.append((miss ** 2).mean())
    adam(backward(outputs, 2 * miss / miss.size), step, LEARNING_RATE * (1 - step / (TRAIN_STEPS + 1)))

    if step % (TRAIN_STEPS // 10) == 0:
        print(f"  step {step:6,}   loss {np.mean(recent):.3f}")
        recent = []


def generate(label, n=DOTS):
    """Un-ruin n dots of pure static into the shape numbered `label`."""
    x = rng.normal(size=(n, 2))
    for level in range(LEVELS, 0, -1):
        guess = forward(features(x, np.full(n, level), np.full(n, label)))[-1]
        clean = (x - noise[level] * guess) / keep[level]
        x = keep[level - 1] * clean + noise[level - 1] * guess
    return x


def gap_to_training(points, k):
    """Average distance from each point to the nearest training dot of shape k."""
    return np.sqrt(((points[:, None, :] - dots[k][None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()


print("\nDid it copy its training dots? A copy would sit 0.000 away from one.")
for k, name in enumerate(SHAPES):
    made = generate(k)
    show(made, f'generated from pure static, asked for "{name}"')
    print(f"   average gap to the nearest training dot: {gap_to_training(made, k):.3f}"
          f"   (for brand-new real dots it is {gap_to_training(make_dots(name), k):.3f})")

if not QUICK:
    def whole(numbers):
        return np.round(np.asarray(numbers) * 1000).astype(int).ravel().tolist()

    with open("diffusion-weights.json", "w") as f:
        json.dump({
            "about": "Made by diffusion.py. Weights, biases and dots are stored x1000 as whole numbers.",
            "shapes": SHAPES, "levels": LEVELS, "waves": WAVES.tolist(), "sizes": sizes,
            "keep": keep.tolist(), "noise": noise.tolist(),
            "weights_x1000": [whole(W) for W in weights],
            "biases_x1000": [whole(b) for b in biases],
            "dots_x1000": [whole(d) for d in dots],
        }, f, separators=(",", ":"))
    print("\nsaved diffusion-weights.json")
