"""
A classifier for your own messages.

How to use it: replace the lines in MESSAGES with your own. Every line is a
kind, then a short message. Use two or more kinds (homework / fun, excited /
grumpy, me / my sister) and about ten messages of each. Then put sentences
you want sorted in TRY. The starter messages are made up.

What is new: nothing! This is the decision model from neuro10.py, shrunk to
fit a handful of messages:

    words -> ids -> rows of E -> average them -> @ W -> softmax -> decision

A few messages, picked at random, are held back and never trained on. They
are the honest test. The dictionary is built from the training messages
only, so a word that appears nowhere in them has no id and no row of E: the
model cannot see it. The output marks those words with [brackets].

Things to try once it runs: change SEED, change TRY, add messages.

Run:
    python3 my_messages.py
"""
import re

import numpy as np

MESSAGES = [
    ("homework", "do we have math homework tonight"),
    ("homework", "the science project is due on friday"),
    ("homework", "i forgot my math book at school"),
    ("homework", "can you help me study for the spelling test"),
    ("homework", "what pages do we have to read for english"),
    ("homework", "the history test is on monday"),
    ("homework", "i still have to finish my science homework"),
    ("homework", "is the book report due tomorrow"),
    ("homework", "we have a math test on friday"),
    ("homework", "did you finish the reading for english"),
    ("fun", "want to play the new game after school"),
    ("fun", "lets go to the park on saturday"),
    ("fun", "movie night at my house bring snacks"),
    ("fun", "we are going to the beach this weekend"),
    ("fun", "come over and play video games"),
    ("fun", "the pool party is on saturday"),
    ("fun", "i got a new bike lets ride to the park"),
    ("fun", "pizza and a movie tonight"),
    ("fun", "do you want to come to my birthday party"),
    ("fun", "lets play soccer at the park after school"),
]
TRY = [
    "can you come and play",
    "we have pizza",
    "homework is fun",
    "fantastic news everybody",
]

HOLD_BACK = 5
EMBED = 8
EPOCHS = 200
LR = 1.0
SEED = 2

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@` on
# small matrices, even though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")
rng = np.random.default_rng(SEED)

kinds = sorted({kind for kind, _ in MESSAGES})
texts = [text for _, text in MESSAGES]
labels = np.array([kinds.index(kind) for kind, _ in MESSAGES])

test_rows = sorted(int(i) for i in rng.permutation(len(MESSAGES))[:HOLD_BACK])
train_rows = [i for i in range(len(MESSAGES)) if i not in test_rows]
print(f"{len(MESSAGES)} messages: {len(train_rows)} to train on, {len(test_rows)} held back for the test")


def words(text):
    return re.findall(r"[a-z]+", text.lower())


vocab = {w: i + 1 for i, w in enumerate(sorted({w for i in train_rows for w in words(texts[i])}))}
V = len(vocab) + 1
print(f"dictionary: {len(vocab)} words")


def to_ids(text):
    return [vocab[w] for w in words(text) if w in vocab]


def padded(id_lists):
    """Rows of ids, filled out with 0 (blank) so they are all the same length."""
    rows = np.zeros((len(id_lists), max(1, max(len(ids) for ids in id_lists))), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        rows[i, :len(ids)] = ids
    return rows, np.maximum((rows > 0).sum(axis=1, keepdims=True), 1)


def with_brackets(text):
    return " ".join(w if w in vocab else f"[{w}]" for w in words(text))


sample = texts[train_rows[0]]
print(f"\nexample: {sample!r}\n  words: {words(sample)}\n  ids:   {to_ids(sample)}\n")

E = rng.normal(0, 0.1, (V, EMBED))
E[0] = 0
W = rng.normal(0, 0.1, (EMBED, len(kinds)))
b = np.zeros(len(kinds))
print(f"model: {E.size + W.size + b.size} weights ({V} x {EMBED} embedding table + a {EMBED} -> {len(kinds)} layer)\n")


def forward(id_rows, word_counts):
    h = E[id_rows].sum(axis=1) / word_counts
    z = h @ W + b
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return h, p / p.sum(axis=1, keepdims=True)


def probabilities(text):
    ids = to_ids(text)
    if not ids:
        return np.full(len(kinds), 1 / len(kinds))
    return forward(*padded([ids]))[1][0]


def count_right(rows):
    return sum(int(probabilities(texts[i]).argmax() == labels[i]) for i in rows)


train_ids, train_counts = padded([to_ids(texts[i]) for i in train_rows])
train_labels = labels[train_rows]

for epoch in range(EPOCHS + 1):
    h, p = forward(train_ids, train_counts)
    if epoch % max(1, EPOCHS // 4) == 0:
        loss = -np.mean(np.log(p[np.arange(len(train_rows)), train_labels]))
        print(f"epoch {epoch:4d}: loss {loss:.3f}   training messages right {count_right(train_rows)}/{len(train_rows)}"
              f"   held-back right {count_right(test_rows)}/{len(test_rows)}")
    if epoch == EPOCHS:
        break

    # softmax + cross-entropy: the gradient at the scores is (p - truth)
    d_scores = p
    d_scores[np.arange(len(train_rows)), train_labels] -= 1
    d_scores /= len(train_rows)
    d_h = d_scores @ W.T

    W -= LR * (h.T @ d_scores)
    b -= LR * d_scores.sum(axis=0)
    # h was an average of word rows, so each word in the message gets an
    # equal share of the blame.
    np.add.at(E, train_ids, -LR * (d_h / train_counts)[:, None, :])
    E[0] = 0

print(f"\nheld-back messages (never trained on{', first 10 shown' if len(test_rows) > 10 else ''}):")
for i in test_rows[:10]:
    p = probabilities(texts[i])
    guess = kinds[int(p.argmax())]
    mark = "right" if guess == kinds[labels[i]] else "WRONG"
    print(f"  {with_brackets(texts[i])}\n      -> {guess} ({p.max():.0%} sure)   truth: {kinds[labels[i]]}   {mark}")

commonest = int(np.bincount(train_labels, minlength=len(kinds)).argmax())
lazy_right = int((labels[test_rows] == commonest).sum())
print(f"\ntraining messages:  {count_right(train_rows)} of {len(train_rows)} right")
print(f"held-back messages: {count_right(test_rows)} of {len(test_rows)} right")
print(f'baseline:           always answering "{kinds[commonest]}" would get {lazy_right} of {len(test_rows)} held-back messages right')

id_to_word = {i: w for w, i in vocab.items()}
pull = E @ W
print("\nthe words it leans on hardest (each word's row of E, sent through W):")
for k, kind in enumerate(kinds):
    lean = pull[:, k] - np.delete(pull, k, axis=1).max(axis=1)
    strongest = [id_to_word[i] for i in np.argsort(-lean) if i > 0][:6]
    print(f"  toward {kind:10s}: {', '.join(strongest)}")

print("\nyour sentences:")
for text in TRY:
    answer = "   ".join(f"{kind} {share:.0%}" for kind, share in zip(kinds, probabilities(text)))
    shrug = "" if to_ids(text) else "   (it can see none of these words, so it shrugs)"
    print(f"  {with_brackets(text)}\n      -> {answer}{shrug}")
