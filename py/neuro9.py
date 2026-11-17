"""
Step 9: scale up to MNIST. 60k handwritten digits, 28x28 pixels each,
10 classes (0-9). Real-size problem, real benchmark, real dataset.

Same shape as neuro8.py, just bigger:
  - input layer: 784  (28 * 28 flattened pixels)
  - hidden 1:    128
  - hidden 2:    64
  - output:      10   (one logit per digit class)

New compared to neuro8.py:
  - torchvision for dataset download / preprocessing
  - DataLoader for batching + shuffling + parallel loading
  - Two hidden layers (not one) -> our first "deep" net
  - Optional GPU via .to(device)
  - Train accuracy AND test accuracy each epoch -> gap = overfitting signal

This is the canonical "hello world" of deep learning.
Expect ~97-98% test accuracy after a few epochs.

Run:
    pip install torch torchvision
    python3 neuro9.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- device ----
# Picks GPU when available, else CPU. PyTorch moves any tensor or
# module with `.to(device)` to that hardware.
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using device: {device}")

# ---- data ----
# transforms.ToTensor() converts PIL image -> float tensor in [0, 1].
# Normalize subtracts mean and divides by std (MNIST conventional values).
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Downloads to ./data on first run (~12MB). After that it's cached.
train_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader handles batching, shuffling, parallel CPU loading.
# num_workers=0 keeps everything in the main process. On macOS with
# Python 3.9, num_workers>0 spawns subprocesses that re-import this
# file, which can crash. 0 is slower but safe everywhere.
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=0)

# ---- model ----
# nn.Flatten turns (batch, 1, 28, 28) into (batch, 784).
hidden1 = 128   # try 128, then 16
hidden2 = 64
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, hidden1),
    nn.ReLU(),
    nn.Linear(hidden1, hidden2),
    nn.ReLU(),
    nn.Linear(hidden2, 10),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- helpers ----
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total

# ---- training ----
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_acc = evaluate(train_loader)
    test_acc  = evaluate(test_loader)
    avg_loss  = running_loss / len(train_loader)
    print(
        f"epoch {epoch+1}: "
        f"loss {avg_loss:.4f}  "
        f"train_acc {train_acc:.4f}  "
        f"test_acc {test_acc:.4f}"
    )

# ---- inspect a few predictions ----
print("\nsample predictions on test set:")
model.eval()
with torch.no_grad():
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    pred = model(x).argmax(dim=1)
    for i in range(10):
        ok = "OK " if pred[i] == y[i] else "BAD"
        print(f"  {ok}  true={y[i].item()}  predicted={pred[i].item()}")
