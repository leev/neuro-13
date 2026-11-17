"""
Learning vs. memorizing: a network that aces the homework and fails the test.

The data is dots on a square of graph paper. A gently wavy line is the true
rule: a dot above the line is 1 ("orange"), a dot below it is 0 ("blue").
Then 15% of the TRAINING dots are given the wrong label on purpose, like
typos in an answer key. The 300 test dots are never trained on, and every
one of their labels is correct.

So a network that learned the true rule perfectly would score 85% on the
training dots (it disagrees with every typo) and 100% on the test. Any
training score far above 85% means it has started memorizing typos.

Three experiments, the same ones as the widget in this chapter:
    A  big network,   80 dots   -> memorizes
    B  big network,  400 dots   -> cure 1: more data
    C  small network, 80 dots   -> cure 2: a simpler model
Every run also remembers the epoch where its test score peaked. Stopping
there is cure 3: early stopping.

What is new compared with the earlier scripts: tanh as the hidden squisher
(sigmoid's cousin, it outputs -1 to +1), and momentum (each step keeps 90%
of the previous step, like a ball that keeps rolling).

Run:
    python3 overfit.py
"""
import numpy as np

WAVE = 0.3
WRONG_SHARE = 0.15
EPOCHS = 1000
LR = 0.2
MOMENTUM = 0.9

# numpy 2.0 on Apple Silicon prints bogus overflow warnings for `@` on
# small matrices, even though every value is finite.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

rng = np.random.default_rng(13)


def true_rule(points):
    x, y = points[:, 0], points[:, 1]
    return (y > WAVE * np.sin(np.pi * x)).astype(float)


def make_dots(count, wrong_share):
    points = rng.uniform(-1, 1, (count, 2))
    labels = true_rule(points)
    typos = rng.choice(count, round(count * wrong_share), replace=False)
    labels[typos] = 1 - labels[typos]
    return points, labels


def make_network(hidden_sizes):
    sizes = [2, *hidden_sizes, 1]
    weights = [rng.uniform(-1, 1, (n_in, n_out)) / np.sqrt(n_in) for n_in, n_out in zip(sizes, sizes[1:])]
    biases = [np.zeros(n_out) for n_out in sizes[1:]]
    return weights, biases


def forward(weights, biases, points):
    layers = [points]
    for w, b in zip(weights[:-1], biases[:-1]):
        layers.append(np.tanh(layers[-1] @ w + b))
    score = layers[-1] @ weights[-1] + biases[-1]
    guess = 1 / (1 + np.exp(-score[:, 0]))
    return layers, guess


def gradients(weights, biases, points, labels):
    layers, guess = forward(weights, biases, points)
    # sigmoid + cross-entropy: the blame at the output is simply (guess - truth)
    blame = (guess - labels)[:, None] / len(points)
    grad_w, grad_b = [], []
    for i in reversed(range(len(weights))):
        grad_w.insert(0, layers[i].T @ blame)
        grad_b.insert(0, blame.sum(axis=0))
        if i > 0:
            # pass the blame back one layer; tanh's slope is 1 - output squared
            blame = (blame @ weights[i].T) * (1 - layers[i] ** 2)
    return grad_w, grad_b


def accuracy(weights, biases, points, labels):
    _, guess = forward(weights, biases, points)
    return ((guess > 0.5) == (labels == 1)).mean()


def run(name, hidden_sizes, train_dots):
    points, labels = train_dots
    weights, biases = make_network(hidden_sizes)
    speed_w = [np.zeros_like(w) for w in weights]
    speed_b = [np.zeros_like(b) for b in biases]
    stored = sum(w.size for w in weights) + sum(b.size for b in biases)
    typos = round(len(points) * WRONG_SHARE)
    print(f"\n{name}: {stored} stored numbers, {len(points)} training dots ({typos} mislabeled)")

    best_test, best_epoch, train_at_best = 0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        grad_w, grad_b = gradients(weights, biases, points, labels)
        for i in range(len(weights)):
            speed_w[i] = MOMENTUM * speed_w[i] - LR * grad_w[i]
            speed_b[i] = MOMENTUM * speed_b[i] - LR * grad_b[i]
            weights[i] += speed_w[i]
            biases[i] += speed_b[i]

        if epoch % 5 == 0:
            train = accuracy(weights, biases, points, labels)
            test = accuracy(weights, biases, *test_dots)
            if test > best_test:
                best_test, best_epoch, train_at_best = test, epoch, train
            if epoch % 100 == 0:
                print(f"  epoch {epoch:4d}   training {train:4.0%}   test {test:4.0%}")
    print(f"  best test score: {best_test:.0%} at epoch {best_epoch} (training was {train_at_best:.0%} then)")
    return name, train, test, best_test, best_epoch


few_dots = make_dots(80, WRONG_SHARE)
many_dots = make_dots(400, WRONG_SHARE)
test_dots = make_dots(300, 0)

results = [
    run("A  big network, 80 dots", [32, 32], few_dots),
    run("B  big network, 400 dots", [32, 32], many_dots),
    run("C  small network, 80 dots", [4], few_dots),
]

print(f"\nafter {EPOCHS} epochs:")
print("                             training   test   best test (early stopping)")
for name, train, test, best_test, best_epoch in results:
    print(f"  {name:26s}  {train:5.0%}   {test:5.0%}   {best_test:5.0%} at epoch {best_epoch}")
best_homework = max(results, key=lambda r: r[1])
best_on_test = max(results, key=lambda r: r[2])
print(f"\nbest homework score: {best_homework[0]}")
print(f"best test score:     {best_on_test[0]}")
