"""
Your own dataset: teach a small network a table that you wrote.

How to use it: replace the rows in MY_TABLE with your own. Every row is a
few clues followed by the answer, all written as words. About 20 rows is
enough to start. Then change ASK to a set of clues you want a prediction
for. The starter rows are made up: they are one imaginary person's taste
in games.

What is new: nothing! This is the meal guesser's network (one-hot inputs,
a ReLU hidden layer, softmax outputs, cross-entropy loss) plus the first
habit of honest training: a few rows, picked at random, are held back as a
test set and never trained on. Compare the two scores at the end. High on
both means the network found a pattern. High on the training rows but low
on the test rows means it only memorized them.

Things to try once it runs: set SCRAMBLE = True, change SEED, raise EPOCHS.

Run:
    python3 my_data.py
"""
import numpy as np

COLUMNS = ("kind", "players", "pace", "like it?")
MY_TABLE = [
    ("racing",    "solo",    "calm",    "yes"),
    ("racing",    "solo",    "frantic", "no"),
    ("racing",    "friends", "calm",    "yes"),
    ("racing",    "friends", "frantic", "yes"),
    ("puzzle",    "solo",    "calm",    "yes"),
    ("puzzle",    "solo",    "frantic", "no"),
    ("puzzle",    "friends", "calm",    "no"),
    ("puzzle",    "friends", "frantic", "no"),
    ("sports",    "solo",    "calm",    "yes"),
    ("sports",    "solo",    "frantic", "no"),
    ("sports",    "friends", "calm",    "yes"),
    ("sports",    "friends", "frantic", "yes"),
    ("building",  "solo",    "calm",    "yes"),
    ("building",  "solo",    "frantic", "no"),
    ("building",  "friends", "calm",    "yes"),
    ("building",  "friends", "frantic", "yes"),
    ("adventure", "solo",    "calm",    "yes"),
    ("adventure", "solo",    "frantic", "no"),
    ("adventure", "friends", "calm",    "yes"),
    ("adventure", "friends", "frantic", "yes"),
]
ASK = ("puzzle", "friends", "frantic")

TEST_ROWS = 5
HIDDEN = 8
EPOCHS = 500
LR = 0.1
SEED = 12
SCRAMBLE = False

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@` on
# small matrices, even though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")
np.random.seed(SEED)

for row in MY_TABLE:
    if len(row) != len(COLUMNS):
        raise ValueError(f"every row needs {len(COLUMNS)} words ({', '.join(COLUMNS)}) but this one has {len(row)}: {row}")

clues = [row[:-1] for row in MY_TABLE]
answers = [row[-1] for row in MY_TABLE]

test_rows = sorted(int(i) for i in np.random.permutation(len(MY_TABLE))[:TEST_ROWS])
train_rows = [i for i in range(len(MY_TABLE)) if i not in test_rows]

if SCRAMBLE:
    answers = [str(a) for a in np.random.permutation(answers)]
    print("SCRAMBLE is on: the answers were shuffled, so there is no pattern left to find.\n")

classes = sorted(set(answers))
switches = sorted({(column, word) for i in train_rows for column, word in enumerate(clues[i])})


def to_inputs(words):
    """One yes/no input for every (column, word) pair seen in the training rows."""
    return [1.0 if words[column] == word else 0.0 for column, word in switches]


def to_label(answer):
    return [1.0 if answer == c else 0.0 for c in classes]


x_train = np.array([to_inputs(clues[i]) for i in train_rows])
y_train = np.array([to_label(answers[i]) for i in train_rows])
x_test = np.array([to_inputs(clues[i]) for i in test_rows])
y_test = np.array([to_label(answers[i]) for i in test_rows])

print(f"{len(MY_TABLE)} rows: {len(train_rows)} to train on, {len(test_rows)} held back for the test")
print(f"{len(switches)} inputs (one per clue word), {HIDDEN} hidden neurons, {len(classes)} outputs ({', '.join(classes)})")
for words in [clues[i] for i in test_rows] + [ASK]:
    for column, word in enumerate(words):
        if (column, word) not in switches:
            print(f'note: {COLUMNS[column]} = "{word}" is not in any training row, so the network knows nothing about it')
print()


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


w1 = np.random.randn(len(switches), HIDDEN) * np.sqrt(2 / len(switches))
b1 = np.zeros((1, HIDDEN))
w2 = np.random.randn(HIDDEN, len(classes)) * np.sqrt(2 / HIDDEN)
b2 = np.zeros((1, len(classes)))


def predict(x):
    hidden = np.maximum(0, x @ w1 + b1)
    return softmax(hidden @ w2 + b2)


def count_right(x, y):
    return int((predict(x).argmax(axis=1) == y.argmax(axis=1)).sum())


for epoch in range(EPOCHS + 1):
    hidden_raw = x_train @ w1 + b1
    hidden = np.maximum(0, hidden_raw)
    probs = softmax(hidden @ w2 + b2)

    if epoch % max(1, EPOCHS // 5) == 0:
        loss = -np.mean(np.log((probs * y_train).sum(axis=1) + 1e-9))
        print(f"epoch {epoch:5d}: loss {loss:.3f}   training rows right {count_right(x_train, y_train)}/{len(train_rows)}"
              f"   test rows right {count_right(x_test, y_test)}/{len(test_rows)}")
    if epoch == EPOCHS:
        break

    d_out = (probs - y_train) / len(train_rows)
    d_hidden = (d_out @ w2.T) * (hidden_raw > 0)
    w2 -= LR * hidden.T @ d_out
    b2 -= LR * d_out.sum(axis=0, keepdims=True)
    w1 -= LR * x_train.T @ d_hidden
    b1 -= LR * d_hidden.sum(axis=0, keepdims=True)

print()
print("The test rows (never trained on):")
for i, p in zip(test_rows, predict(x_test)):
    guess = classes[int(p.argmax())]
    mark = "right" if guess == answers[i] else "WRONG"
    print(f"  {', '.join(clues[i]):32s} -> {guess} ({p.max():.0%} sure)   truth: {answers[i]}   {mark}")

train_right = count_right(x_train, y_train)
test_right = count_right(x_test, y_test)
train_answers = [answers[i] for i in train_rows]
commonest = max(classes, key=train_answers.count)
lazy_right = sum(answers[i] == commonest for i in test_rows)

print()
print(f"training rows: {train_right} of {len(train_rows)} right")
print(f"test rows:     {test_right} of {len(test_rows)} right")
print(f'baseline:      always answering "{commonest}" would get {lazy_right} of {len(test_rows)} test rows right')
print()
if test_right > lazy_right:
    print("Verdict: it beat the baseline on rows it never saw, so it found a real pattern.")
elif train_right > 0.8 * len(train_rows):
    print("Verdict: good on the training rows, no better than the baseline on the test rows.")
    print("That is memorizing, not learning.")
else:
    print("Verdict: it has not even learned the training rows. Try more EPOCHS or more HIDDEN neurons.")
print(f"(With only {len(test_rows)} test rows, each one counts for a lot. More rows give a steadier score.)")

print()
print(f"You asked about: {', '.join(ASK)}")
for c, p in sorted(zip(classes, predict(np.array([to_inputs(ASK)]))[0]), key=lambda pair: -pair[1]):
    print(f"  {c:>10s}: {p:.0%}")
