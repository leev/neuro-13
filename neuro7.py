# Requires: pip install numpy scikit-learn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Step 7: real dataset, still hand-rolled backprop in numpy.
# Iris flowers: 150 samples, 4 numeric features, 3 species.
#
# New concepts vs the toy examples:
#   - Real-valued inputs (not just 0/1) -> need feature scaling
#   - Train/test split -> measure generalization, not memorization
#   - Mini-batches -> update weights more often, smoother training
#   - Accuracy metric -> easier to interpret than raw loss
#
# Dataset details:
#   x_raw: shape (150, 4). Columns = [sepal_len, sepal_wid, petal_len, petal_wid] in cm.
#   y_raw: shape (150,).   Values = 0 (setosa), 1 (versicolor), 2 (virginica).
#
# After preprocessing:
#   x_train: (120, 4) standardized features (mean 0, std 1).
#   y_train: (120, 3) one-hot labels.
#   x_test:  (30, 4)  held-out samples we NEVER train on.
#   y_test:  (30, 3)  held-out one-hot labels.

iris = load_iris()
x_raw = iris.data            # (150, 4)
y_raw = iris.target          # (150,)

# One-hot encode labels: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1].
y_onehot = np.zeros((y_raw.shape[0], 3))
y_onehot[np.arange(y_raw.shape[0]), y_raw] = 1

# Train/test split. Hold out 20% of samples to measure generalization.
x_train, x_test, y_train, y_test = train_test_split(
    x_raw, y_onehot, test_size=0.2, random_state=1, stratify=y_raw,
)

# Feature scaling. Each feature now mean=0, std=1.
# Compute stats on TRAIN only, apply to both, so we don't leak test info.
mean = x_train.mean(axis=0)
std  = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std

# Silence a cosmetic numpy warning. On Apple Silicon / Accelerate, the
# small `d_o @ w2.T` matmul in the backward pass can trigger spurious
# "divide/overflow/invalid" FP flags from inside SIMD code even though
# every input and output is finite and training converges fine.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

def relu(x):  return np.maximum(0, x)
def drelu(x): return (x > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def accuracy(x, y):
    h = relu(x @ w1 + b1)
    o = softmax(h @ w2 + b2)
    return (o.argmax(axis=1) == y.argmax(axis=1)).mean()

np.random.seed(1)
# Wider hidden layer: 4 inputs -> 16 hidden -> 3 classes.
# He initialization: scale by sqrt(2/fan_in) for ReLU layers. Keeps
# activations from exploding or collapsing on the very first forward
# pass. Without it, the first batch can NaN out before training settles.
w1 = np.random.randn(4, 16) * np.sqrt(2.0 / 4)
b1 = np.zeros((1, 16))
w2 = np.random.randn(16, 3) * np.sqrt(2.0 / 16)
b2 = np.zeros((1, 3))
lr = 0.02            # ReLU + He init can amplify early gradients; small lr avoids NaN spikes
batch_size = 16
epochs = 200

n = x_train.shape[0]
for epoch in range(epochs):
    # Shuffle indices each epoch so mini-batches differ.
    idx = np.random.permutation(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start+batch_size]
        xb = x_train[batch]
        yb = y_train[batch]

        # forward
        h = relu(xb @ w1 + b1)
        o = softmax(h @ w2 + b2)

        # backward
        d_o = (o - yb) / xb.shape[0]
        d_h = (d_o @ w2.T) * drelu(h)

        # update
        w2 -= lr * h.T @ d_o
        b2 -= lr * d_o.sum(axis=0, keepdims=True)
        w1 -= lr * xb.T @ d_h
        b1 -= lr * d_h.sum(axis=0, keepdims=True)

    if epoch % 20 == 0:
        train_acc = accuracy(x_train, y_train)
        test_acc  = accuracy(x_test,  y_test)
        print(f"epoch {epoch:3d}: train_acc {train_acc:.3f}  test_acc {test_acc:.3f}")

print()
print(f"final train accuracy: {accuracy(x_train, y_train):.3f}")
print(f"final test  accuracy: {accuracy(x_test,  y_test):.3f}")
