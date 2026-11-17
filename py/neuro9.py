"""
Step 9: read handwritten digits with PyTorch, using all of MNIST:
60,000 pictures to learn from and 10,000 held back for the test.

The network is 784 -> 128 -> 64 -> 10: every pixel in, one score per digit out.

What each part is, in words you already know:

    datasets.MNIST, Normalize    load the pictures and scale the pixels
    DataLoader                   shuffle and cut into mini-batches
    nn.Flatten                   a 28 x 28 picture becomes one list of 784 numbers
    nn.Linear, nn.ReLU           x @ w + b, then np.maximum(0, x)
    nn.CrossEntropyLoss          softmax + cross-entropy
    loss.backward()              the whole backward pass
    optimizer.step()             nudge every weight (Adam's smarter nudge)
    .to(device)                  do the math on a graphics chip if there is one

PyTorch does not run in the browser. On your own computer:

    pip install torch torchvision
    python3 neuro9.py

The first run downloads MNIST (about 12 MB) into a folder called "data",
next to the folder this script is in.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 5
BATCH = 128
LEARNING_RATE = 0.001
DATA = Path(__file__).resolve().parent.parent / "data"

torch.manual_seed(13)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"doing the math on: {device}")

# 0.1307 and 0.3081 are the average and spread of every pixel in the MNIST
# training pictures, so this is the same feature scaling as the iris script.
prepare = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_set = datasets.MNIST(root=DATA, train=True, download=True, transform=prepare)
test_set = datasets.MNIST(root=DATA, train=False, download=True, transform=prepare)
train_batches = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_batches = DataLoader(test_set, batch_size=512)
print(f"{len(train_set)} training pictures, {len(test_set)} test pictures")

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"{sum(p.numel() for p in model.parameters()):,} numbers to learn\n")


def accuracy(batches):
    right = 0
    with torch.no_grad():
        for pictures, labels in batches:
            pictures, labels = pictures.to(device), labels.to(device)
            right += (model(pictures).argmax(dim=1) == labels).sum().item()
    return right / len(batches.dataset)


for epoch in range(1, EPOCHS + 1):
    loss_sum = 0.0
    for pictures, labels in train_batches:
        pictures, labels = pictures.to(device), labels.to(device)

        scores = model(pictures)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(
        f"epoch {epoch}: loss {loss_sum / len(train_batches):.4f}   "
        f"train accuracy {accuracy(train_batches):.4f}   test accuracy {accuracy(test_batches):.4f}"
    )

print("\nthe first ten test pictures:")
pictures, labels = next(iter(test_batches))
with torch.no_grad():
    guesses = model(pictures.to(device)).argmax(dim=1).cpu()
for label, guess in zip(labels[:10].tolist(), guesses[:10].tolist()):
    print(f"  really a {label}, network says {guess}   {'ok' if label == guess else 'WRONG'}")
