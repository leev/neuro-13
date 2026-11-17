"""
Step 8: the iris network from neuro7.py again, this time in PyTorch.

Every line here replaces something neuro7.py spelled out by hand:

    neuro7.py (numpy, by hand)            neuro8.py (PyTorch)
    x @ w1 + b1                           nn.Linear(4, 16)
    np.maximum(0, x)                      nn.ReLU()
    softmax, then cross-entropy           nn.CrossEntropyLoss()
    h = ..., o = ...     (forward)        model(x)
    d_o = ..., d_h = ... (backward)       loss.backward()
    w -= lr * gradient                    optimizer.step()
    plain gradient descent                torch.optim.Adam

One line has no partner in neuro7.py: optimizer.zero_grad(). PyTorch adds
new blame on top of old blame, exactly like the += in autograd.py, so the
old blame has to be cleared before every backward pass.

PyTorch does not run in the browser. On your own computer:

    pip install torch scikit-learn
    python3 neuro8.py
"""
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

HIDDEN = 16
EPOCHS = 200
BATCH = 16
LEARNING_RATE = 0.01

torch.manual_seed(1)

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=1, stratify=iris.target,
)

# The average and spread come from the training flowers only, so nothing
# about the test flowers leaks into training.
mean, std = x_train.mean(axis=0), x_train.std(axis=0)
x_train = torch.tensor((x_train - mean) / std, dtype=torch.float32)
x_test = torch.tensor((x_test - mean) / std, dtype=torch.float32)
# CrossEntropyLoss wants each label as a class number (0, 1 or 2), not one-hot.
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(4, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 3),
)
# There is no softmax layer in the model: CrossEntropyLoss does the softmax itself.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"{sum(p.numel() for p in model.parameters())} numbers to learn")


def accuracy(x, y):
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


for epoch in range(EPOCHS):
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), BATCH):
        batch = order[start:start + BATCH]

        scores = model(x_train[batch])
        loss = loss_fn(scores, y_train[batch])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}: train accuracy {accuracy(x_train, y_train):.3f}   test accuracy {accuracy(x_test, y_test):.3f}")

print()
print(f"final train accuracy: {accuracy(x_train, y_train):.3f}")
print(f"final test  accuracy: {accuracy(x_test, y_test):.3f}")
