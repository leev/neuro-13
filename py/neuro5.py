import numpy as np

# Step 5: same snack classifier as neuro4.py, but ReLU hidden
# activation instead of sigmoid. ONE change to forward + ONE change
# to backward. Output layer stays softmax + cross-entropy.
#
# Why this matters:
#   Sigmoid saturates (slope -> 0) when input is very + or very -.
#   In deep nets, sigmoid gradients shrink layer by layer until they
#   vanish. ReLU = max(0, x). Slope is exactly 1 wherever x > 0,
#   so gradients pass through cleanly. Every modern deep net uses
#   ReLU (or a variant: GELU, SiLU, LeakyReLU) in hidden layers.
#
# Side note: ReLU is sensitive to learning rate. Sigmoid was forgiving
# at lr=0.5. With ReLU we drop to lr=0.1 to avoid blowing up the
# weighted sums on the first few steps.
#
# Same data as neuro4.py: "Which snack should I grab?"
#   inputs  = [hungry, after_school, weekend]
#   classes = [fruit, chips, cookie]
x = np.array([
    [0,0,0],[1,0,0],
    [0,1,0],[1,1,0],
    [0,0,1],[1,0,1],
    [0,1,1],[1,1,1],
])
y = np.array([
    [1,0,0],[1,0,0],
    [0,1,0],[0,1,0],
    [0,0,1],[0,0,1],
    [0,0,1],[0,0,1],
])

CLASS_NAMES = ["fruit", "chips", "cookie"]

def relu(x):  return np.maximum(0, x)
def drelu(x): return (x > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

np.random.seed(1)
# He init would be sqrt(2/fan_in) scaling. Skipping it here, the small
# uniform init still converges on this toy problem.
w1 = 2*np.random.random((3,6)) - 1   # 3 inputs -> 6 hidden
b1 = np.zeros((1,6))
w2 = 2*np.random.random((6,3)) - 1   # 6 hidden -> 3 classes
b2 = np.zeros((1,3))
lr = 0.1

for epoch in range(20000):
    # forward
    h = relu(x @ w1 + b1)          # <- was sigmoid in neuro4.py
    o = softmax(h @ w2 + b2)

    # backward
    d_o = (o - y) / x.shape[0]     # softmax + cross-entropy combined gradient
    d_h = (d_o @ w2.T) * drelu(h)  # <- was * dsigmoid(h) in neuro4.py

    # update
    w2 -= lr * h.T @ d_o
    b2 -= lr * d_o.sum(axis=0, keepdims=True)
    w1 -= lr * x.T @ d_h
    b1 -= lr * d_h.sum(axis=0, keepdims=True)

    if epoch % 2000 == 0:
        loss = -np.sum(y * np.log(o + 1e-9)) / x.shape[0]
        print(f"epoch {epoch}: loss {loss:.4f}")

print()
print('Trained — "Which snack should I grab?"  (fruit / chips / cookie)')
header = (
    f'  {"hungry":>6} {"after_school":>13} {"weekend":>8}  '
    f'{"fruit":>6} {"chips":>6} {"cookie":>7}   '
    f'{"predicted":>9}   {"truth":>6}'
)
print(header)
for (hu, sc, we), probs, truth in zip(x, o, y):
    pred_idx  = int(probs.argmax())
    truth_idx = int(truth.argmax())
    pred_name  = CLASS_NAMES[pred_idx]
    truth_name = CLASS_NAMES[truth_idx]
    print(
        f'  {hu:>6} {sc:>13} {we:>8}  '
        f'{probs[0]:>6.2f} {probs[1]:>6.2f} {probs[2]:>7.2f}   '
        f'{pred_name:>9}   {truth_name:>6}'
    )
