"""
A language model you can train. It reads a list of first names and learns
to invent new ones, one letter at a time.

Data: names.txt, 8,020 first names (lowercase, one per line, already
shuffled). The first 90% train the model. The last 10% are held out to
test it on names it has never seen.

What is new compared with the earlier scripts:
  * Nobody labelled anything. The training examples are cut out of the
    names themselves: slide a three-letter window along each name and ask
    "what comes next?"
  * Letters go in as id numbers, and each id picks its row of an
    embedding table.
  * The model answers with 27 probabilities: one for each letter, plus
    one for "." which means "the name ends here".
  * To write, it runs in a loop: predict, pick a symbol by its
    probability, stick it on the end, ask again.

The whole model:
    C   27 rows of 2 numbers       the embedding table, one row per symbol
    W1  6 numbers -> 64 neurons    three rows side by side, then tanh
    W2  64 neurons -> 27 scores    softmax turns the scores into probabilities

Run:
    python3 names.py
"""
import numpy as np

SYMBOLS = ".abcdefghijklmnopqrstuvwxyz"
CONTEXT = 3
EMBED = 2
HIDDEN = 64
STEPS = 20000
BATCH = 32
LR_START = 0.3
LR_END = 0.01

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@`, even
# though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

words = open("names.txt").read().split()
n_train = int(len(words) * 0.9)
train_words, held_words = words[:n_train], words[n_train:]


def make_examples(word_list):
    windows, answers = [], []
    for word in word_list:
        window = [0] * CONTEXT
        for symbol in word + ".":
            answer = SYMBOLS.index(symbol)
            windows.append(window)
            answers.append(answer)
            window = window[1:] + [answer]
    return np.array(windows), np.array(answers)


X_train, y_train = make_examples(train_words)
X_held, y_held = make_examples(held_words)
print(f"{len(words)} names: {len(train_words)} to train on, {len(held_words)} held out")
print(f"sliding a {CONTEXT}-letter window along the training names makes {len(y_train)} examples")
print(f"the examples cut from '{train_words[0]}':")
for window, answer in zip(X_train[:len(train_words[0]) + 1], y_train):
    print("   ", "".join(SYMBOLS[i] for i in window), "->", SYMBOLS[answer])

counts = np.zeros((27, 27))
np.add.at(counts, (X_train[:, -1], y_train), 1)
table = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)
counting_loss = -np.log(table[X_held[:, -1], y_held]).mean()
print(f"\na model that knows nothing would score a loss of {np.log(27):.3f}")
print(f"the counting table (one letter back) scores {counting_loss:.3f} on the held-out names")

rng = np.random.default_rng(13)
C = rng.normal(0, 0.3, (27, EMBED))
W1 = rng.normal(0, (5 / 3) / np.sqrt(CONTEXT * EMBED), (CONTEXT * EMBED, HIDDEN))
b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, 0.01, (HIDDEN, 27))
b2 = np.zeros(27)
print(f"the network has {C.size + W1.size + b1.size + W2.size + b2.size:,} weights, all random to begin with\n")


def forward(windows):
    x = C[windows].reshape(len(windows), -1)
    h = np.tanh(x @ W1 + b1)
    scores = h @ W2 + b2
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    return x, h, p / p.sum(axis=1, keepdims=True)


def loss(windows, answers):
    p = forward(windows)[2]
    return -np.log(p[np.arange(len(answers)), answers]).mean()


def invent(temperature=1.0):
    window, name = [0] * CONTEXT, ""
    while len(name) < 24:
        p = forward(np.array([window]))[2][0] ** (1 / temperature)
        pick = rng.choice(27, p=p / p.sum())
        if pick == 0:
            break
        name += SYMBOLS[pick]
        window = window[1:] + [pick]
    return name


def report(step):
    print(f"step {step:5d}   training loss {loss(X_train, y_train):.3f}   "
          f"held-out loss {loss(X_held, y_held):.3f}   it writes: {invent()}")


report(0)
for step in range(STEPS):
    lr = LR_START + (LR_END - LR_START) * step / STEPS
    batch = rng.integers(0, len(y_train), BATCH)
    x, h, p = forward(X_train[batch])

    d_scores = p
    d_scores[np.arange(BATCH), y_train[batch]] -= 1
    d_scores /= BATCH
    d_h = d_scores @ W2.T
    d_sum = d_h * (1 - h * h)
    d_x = d_sum @ W1.T

    W2 -= lr * (h.T @ d_scores)
    b2 -= lr * d_scores.sum(axis=0)
    W1 -= lr * (x.T @ d_sum)
    b1 -= lr * d_sum.sum(axis=0)
    np.add.at(C, X_train[batch], -lr * d_x.reshape(BATCH, CONTEXT, EMBED))

    if (step + 1) % 2000 == 0:
        report(step + 1)

held_loss = loss(X_held, y_held)
print(f"\nheld-out loss {held_loss:.3f}: about as unsure as choosing among {np.exp(held_loss):.1f} symbols")
print(f"(the counting table: {np.exp(counting_loss):.1f} symbols; knowing nothing: 27)")

print("\ntwenty names it made up:")
in_training = set(train_words)
for _ in range(20):
    name = invent()
    print(f"    {name:26s}{'copied from the training list' if name in in_training else 'new'}")

q, u = SYMBOLS.index("q"), SYMBOLS.index("u")
after_q = forward(np.array([[0] * (CONTEXT - 1) + [q]]))[2][0]
print(f"\nwhat follows a 'q' at the start of a name? the counting table says 'u' {counts[q, u] / counts[q].sum():.0%} of the time;")
print(f"the network, with {EMBED} numbers per letter, gives 'u' {after_q[u]:.0%}.")

if EMBED == 2:
    print("\nthe letter map: every symbol placed by the two numbers in its row of C (VOWELS in capitals)")
    width, height = 61, 21
    grid = [[" "] * width for _ in range(height)]
    low, high = C.min(axis=0), C.max(axis=0)
    for symbol, (across, up) in zip(SYMBOLS, (C - low) / (high - low)):
        grid[round((1 - up) * (height - 1))][round(across * (width - 1))] = symbol.upper() if symbol in "aeiou" else symbol
    print("\n".join("    " + "".join(row).rstrip() for row in grid))
