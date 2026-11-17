"""
Step 8: same iris problem as neuro7.py, rewritten in PyTorch.

Goal: see what a framework hides. Every line below replaces something
you wrote by hand in neuro7.py.

  numpy version (neuro7.py)            PyTorch version (this file)
  ---------------------------          -----------------------------
  np.maximum(0, x)                ->   nn.ReLU
  softmax + cross-entropy by hand ->   nn.CrossEntropyLoss
  manual forward (h = ..., o = ..)->   model(x)
  manual backward (d_o, d_h)      ->   loss.backward()    (autograd)
  manual weight updates           ->   optimizer.step()
  plain SGD                       ->   torch.optim.Adam   (smarter)
  CPU only                        ->   model.to('cuda') if available

The math is identical. PyTorch just stops you typing it.

Run:
    pip install torch scikit-learn
    python3 neuro8.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ---- data ----
iris = load_iris()
x_raw = iris.data           # (150, 4)
y_raw = iris.target         # (150,)  integer labels 0/1/2

x_train, x_test, y_train, y_test = train_test_split(
    x_raw, y_raw, test_size=0.2, random_state=1, stratify=y_raw,
)

# Standardize features using train stats only.
mean = x_train.mean(axis=0)
std  = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std

# Convert numpy arrays to torch tensors.
# Note: CrossEntropyLoss expects integer class labels, NOT one-hot.
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
x_test_t  = torch.tensor(x_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# ---- model ----
# nn.Sequential = stack of layers, runs them in order.
# nn.Linear(in, out) = weights w (in, out) + bias b (out). Same as
# `x @ w + b` you wrote by hand.
hidden = 16   # try 16, then 4, then 64
model = nn.Sequential(
    nn.Linear(4, hidden),
    nn.ReLU(),
    nn.Linear(hidden, 3),
)

# CrossEntropyLoss applies log-softmax internally, then computes
# cross-entropy. Don't put a softmax in the model. Tensors fed in are
# raw scores ("logits"), labels are integer class indices.
loss_fn = nn.CrossEntropyLoss()

# Adam = SGD with momentum + per-parameter learning rate. Almost
# always the default for new projects.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---- training ----
batch_size = 16
epochs = 200
n = x_train_t.shape[0]

for epoch in range(epochs):
    # Shuffle indices each epoch.
    idx = torch.randperm(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start+batch_size]
        xb = x_train_t[batch]
        yb = y_train_t[batch]

        logits = model(xb)            # forward
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()         # clear leftover grads
        loss.backward()               # autograd: compute every gradient
        optimizer.step()              # apply gradients

    if epoch % 20 == 0:
        with torch.no_grad():
            train_pred = model(x_train_t).argmax(dim=1)
            test_pred  = model(x_test_t).argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()
            test_acc  = (test_pred  == y_test_t).float().mean().item()
            print(f"epoch {epoch:3d}: train_acc {train_acc:.3f}  test_acc {test_acc:.3f}")

# ---- final eval ----
with torch.no_grad():
    train_acc = (model(x_train_t).argmax(dim=1) == y_train_t).float().mean().item()
    test_acc  = (model(x_test_t ).argmax(dim=1) == y_test_t ).float().mean().item()

print()
print(f"final train accuracy: {train_acc:.3f}")
print(f"final test  accuracy: {test_acc:.3f}")
