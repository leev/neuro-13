"""
Step 10: a tiny "decision model". It never writes a sentence. It reads a
news headline and returns a typed answer with a probability:

    choice : which of four topics is this?  -> {"choice": "sports", "confidence": 0.97}
    yes/no : is this about sports?          -> {"sports": 0.97}

Data: headlines.csv, 10,000 real headlines (2,500 per topic) from the
AG News collection. First 8,000 rows train the model, last 2,000 test it.

The whole model is two grids of numbers:
    E : one row of 16 numbers per known word   (the embedding table, chapter 14)
    W : 16 numbers in -> 4 topic scores out    (the snack picker, chapter 9)

    words -> ids -> rows of E -> average them -> @ W -> softmax -> decision

Run:
    python3 neuro10.py
"""
import csv
import json
import re

import numpy as np

TOPICS = ["world", "sports", "business", "scitech"]
EMBED = 16
EPOCHS = 15
BATCH = 64
LR = 1.0

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@` on
# matrices this size, even though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

rows = list(csv.DictReader(open("headlines.csv", encoding="utf-8")))
texts = [r["text"] for r in rows]
labels = np.array([TOPICS.index(r["topic"]) for r in rows])
N_TRAIN = 8000
print(f"{len(rows)} headlines: {N_TRAIN} to train on, {len(rows) - N_TRAIN} held back for the test")


def words(text):
    return re.findall(r"[a-z]+", text.lower())


# The dictionary is built from the TRAINING rows only. A word has to show up
# at least 3 times to earn an id. Id 0 is reserved for "blank".
counts = {}
for t in texts[:N_TRAIN]:
    for w in words(t):
        counts[w] = counts.get(w, 0) + 1
vocab = {w: i + 1 for i, w in enumerate(sorted(w for w, c in counts.items() if c >= 3))}
V = len(vocab) + 1
print(f"dictionary: {len(vocab)} words")


def to_ids(text):
    return [vocab[w] for w in words(text) if w in vocab]


sample = texts[0][:60]
print(f"\nexample: {sample!r}\n  words: {words(sample)}\n  ids:   {to_ids(sample)}\n")

# Every headline becomes one row of ids, padded with 0 (blank) so the rows
# are all the same length and numpy can process a whole batch at once.
id_lists = [to_ids(t) for t in texts]
LONGEST = max(len(ids) for ids in id_lists)
ids = np.zeros((len(rows), LONGEST), dtype=np.int64)
for i, row in enumerate(id_lists):
    ids[i, :len(row)] = row
n_words = np.maximum((ids > 0).sum(axis=1, keepdims=True), 1)

rng = np.random.default_rng(13)
E = rng.normal(0, 0.1, (V, EMBED))
E[0] = 0
W = rng.normal(0, 0.1, (EMBED, len(TOPICS)))
b = np.zeros(len(TOPICS))
print(f"model: {E.size + W.size + b.size:,} weights ({V} x {EMBED} embedding table + a {EMBED} -> 4 layer)\n")


def forward(id_rows, word_counts):
    h = E[id_rows].sum(axis=1) / word_counts
    z = h @ W + b
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return h, p / p.sum(axis=1, keepdims=True)


def accuracy(lo, hi):
    _, p = forward(ids[lo:hi], n_words[lo:hi])
    return (p.argmax(axis=1) == labels[lo:hi]).mean()


for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(N_TRAIN)
    for start in range(0, N_TRAIN, BATCH):
        batch = order[start:start + BATCH]
        h, p = forward(ids[batch], n_words[batch])

        # softmax + cross-entropy: the gradient at the scores is (p - truth)
        d_scores = p
        d_scores[np.arange(len(batch)), labels[batch]] -= 1
        d_scores /= len(batch)
        d_h = d_scores @ W.T

        W -= LR * (h.T @ d_scores)
        b -= LR * d_scores.sum(axis=0)
        # h was an average of word rows, so each word in the headline gets
        # an equal share of the blame.
        np.add.at(E, ids[batch], -LR * (d_h / n_words[batch])[:, None, :])
        E[0] = 0
    print(f"epoch {epoch:2d}   train accuracy {accuracy(0, N_TRAIN):.3f}   test accuracy {accuracy(N_TRAIN, len(rows)):.3f}")

_, p_test = forward(ids[N_TRAIN:], n_words[N_TRAIN:])
confidence = p_test.max(axis=1)
correct = p_test.argmax(axis=1) == labels[N_TRAIN:]
print("\nhow honest is its confidence? (2,000 test headlines)")
for lo, hi in [(0.25, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
    mask = (confidence >= lo) & (confidence < hi)
    print(f"  says {lo:4.0%} to {min(hi, 1):4.0%} sure: {mask.sum():4d} headlines, actually right {correct[mask].mean():.0%}")


def probabilities(text):
    row = to_ids(text)
    if not row:
        return np.full(len(TOPICS), 1 / len(TOPICS))
    return forward(np.array([row]), np.array([[len(row)]]))[1][0]


def choice(text):
    p = probabilities(text)
    return {"choice": TOPICS[p.argmax()], "confidence": round(float(p.max()), 2)}


def yes_no(text, topic):
    return {topic: round(float(probabilities(text)[TOPICS.index(topic)]), 2)}


print("\ndecisions:")
for text in [
    "Striker scores twice as United win the cup final",
    "New phone chip makes laptops faster",
    "Oil prices fall as markets slide",
    "My dog ate my homework",
]:
    print(f"  {text}")
    print(f"    choice -> {json.dumps(choice(text))}")
    print(f"    is it sports? -> {json.dumps(yes_no(text, 'sports'))}")

# Nobody told the model which words are related. Words that push toward the
# same answers end up with similar rows in E.
id_to_word = {i: w for w, i in vocab.items()}
unit = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
print("\nwords the model filed next to each other:")
for w in ["football", "software", "president", "profit"]:
    nearest = np.argsort(-(unit @ unit[vocab[w]]))[1:6]
    print(f"  {w:10s} ~ {', '.join(id_to_word[i] for i in nearest)}")
