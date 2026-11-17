"""
Read my handwriting: train a network that reads handwritten digits.

Data: digits.bin holds 10,000 real handwritten digits from the MNIST
collection (by Yann LeCun, Corinna Cortes and Christopher Burges). 8,000 are
for training and 2,000 are held back for an honest test.

What is new: the input is a picture. A picture is a grid of numbers, one per
pixel, saying how much ink is there (0 = blank paper, 255 = full ink). We
unroll the 28 x 28 grid into one long row of 784 numbers and feed that in.

    784 pixels -> 64 hidden neurons (ReLU) -> 10 scores -> softmax

After training, the script slides every test digit sideways to show the
network's blind spot. Then try the experiment at the bottom of the chapter:
set SCRAMBLE = True and run it again.

Run:
    python3 digits.py
"""
import numpy as np

HIDDEN = 64
EPOCHS = 10
BATCH = 32
LR = 0.2
SCRAMBLE = False

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@` on
# matrices this size, even though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

raw = np.fromfile("digits.bin", dtype=np.uint8)
n_train, n_test, side = (int(v) for v in raw[4:16].view("<u4"))
n = n_train + n_test
labels = raw[16:16 + n].astype(np.int64)
pictures = raw[16 + n:].reshape(n, side, side)
print(f"{n} digits, each {side} x {side} pixels: {n_train} to train on, {n_test} held back for the test")


def show(picture):
    for row in picture:
        print("    " + "".join(" .:+#@"[int(v) * 6 // 256] * 2 for v in row))


print(f"\nthe first one is labelled {labels[0]}. Here is its grid of numbers, drawn with letters:")
show(pictures[0])

rng = np.random.default_rng(13)

if SCRAMBLE:
    # One fixed shuffle of the 784 pixel positions, used for every picture.
    shuffle = rng.permutation(side * side)
    pictures = pictures.reshape(n, -1)[:, shuffle].reshape(n, side, side)
    print("\nSCRAMBLE is on. The same digit now looks like this to you:")
    show(pictures[0])

x = pictures.reshape(n, side * side).astype(np.float32) / 255
x_train, y_train = x[:n_train], labels[:n_train]
x_test, y_test = x[n_train:], labels[n_train:]

W1 = rng.normal(0, 0.01, (x.shape[1], HIDDEN)).astype(np.float32)
b1 = np.zeros(HIDDEN, dtype=np.float32)
W2 = rng.normal(0, 0.1, (HIDDEN, 10)).astype(np.float32)
b2 = np.zeros(10, dtype=np.float32)
print(f"\nmodel: {W1.size + b1.size + W2.size + b2.size:,} weights "
      f"({x.shape[1]} x {HIDDEN} + {HIDDEN} biases, then {HIDDEN} x 10 + 10 biases)\n")


def forward(rows):
    hidden = np.maximum(rows @ W1 + b1, 0)
    scores = hidden @ W2 + b2
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    return hidden, p / p.sum(axis=1, keepdims=True)


def accuracy(rows, answers):
    return (forward(rows)[1].argmax(axis=1) == answers).mean()


print(f"before training   train accuracy {accuracy(x_train, y_train):.3f}   test accuracy {accuracy(x_test, y_test):.3f}")
for epoch in range(1, EPOCHS + 1):
    order = rng.permutation(n_train)
    for start in range(0, n_train, BATCH):
        batch = order[start:start + BATCH]
        hidden, p = forward(x_train[batch])

        # softmax + cross-entropy: the gradient at the scores is (p - truth)
        d_scores = p
        d_scores[np.arange(len(batch)), y_train[batch]] -= 1
        d_scores /= len(batch)
        # ReLU passes blame back only through the neurons that fired
        d_hidden = (d_scores @ W2.T) * (hidden > 0)

        W2 -= LR * (hidden.T @ d_scores)
        b2 -= LR * d_scores.sum(axis=0)
        W1 -= LR * (x_train[batch].T @ d_hidden)
        b1 -= LR * d_hidden.sum(axis=0)
    print(f"epoch {epoch:2d}          train accuracy {accuracy(x_train, y_train):.3f}   test accuracy {accuracy(x_test, y_test):.3f}")

p_test = forward(x_test)[1]
guesses = p_test.argmax(axis=1)
wrong = np.flatnonzero(guesses != y_test)
print(f"\nit got {len(wrong)} of the {n_test} test digits wrong. The first three:")
for i in wrong[:3]:
    print(f"  the label says {y_test[i]}, the network said {guesses[i]} and was {p_test[i].max():.0%} sure")
if not SCRAMBLE:
    print("\nhere is the first of those. Would you have got it right?")
    show(pictures[n_train + wrong[0]])

    print("\nthe blind spot: slide every test digit sideways and test again")
    for shift in [1, 2, 3, 4]:
        moved = np.zeros_like(pictures[n_train:])
        moved[:, :, shift:] = pictures[n_train:, :, :-shift]
        moved = moved.reshape(n_test, -1).astype(np.float32) / 255
        print(f"  {shift} pixel{'s' if shift > 1 else ' '} to the right: test accuracy {accuracy(moved, y_test):.3f}")
else:
    print("\nevery picture was scrambled the same way. Compare the test accuracy with an")
    print("unscrambled run. If it is about the same, the network never knew which pixels")
    print("were next to each other.")
