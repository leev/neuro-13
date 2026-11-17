import numpy as np

# Step 4: multi-class classifier. 3 binary inputs -> 3 mutually
# exclusive output classes (fruit / chips / cookie).
# New concepts vs step 3: softmax output, cross-entropy loss,
# one-hot labels.
#
# Rule we're teaching: "Which snack should I grab?"
#   weekend                 -> cookie  (weekend treat)
#   after_school (not wknd) -> chips   (after-school energy)
#   otherwise               -> fruit   (boring-day default)
#   (the "hungry" input is intentionally noise — the network must
#    figure out by itself that it doesn't change the decision.)
#
# x: shape (8, 3). 8 samples, 3 binary features per sample.
#   columns = [hungry, after_school, weekend]
#   rows    = all 8 combinations.
#
# y: shape (8, 3). One-hot labels.
#   columns = [fruit, chips, cookie]   (exactly one column is 1 per row)
#
# Row-by-row decoding of (x, y):
#   [0,0,0] plain weekday         -> [1,0,0] fruit
#   [1,0,0] hungry weekday        -> [1,0,0] fruit
#   [0,1,0] after school          -> [0,1,0] chips
#   [1,1,0] hungry, after school  -> [0,1,0] chips
#   [0,0,1] weekend               -> [0,0,1] cookie
#   [1,0,1] hungry weekend        -> [0,0,1] cookie
#   [0,1,1] weekend + after-school-> [0,0,1] cookie  (weekend wins)
#   [1,1,1] hungry, both          -> [0,0,1] cookie
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

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return x*(1-x)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

np.random.seed(1)
w1 = 2*np.random.random((3,6)) - 1   # 3 inputs -> 6 hidden
b1 = np.zeros((1,6))
w2 = 2*np.random.random((6,3)) - 1   # 6 hidden -> 3 classes
b2 = np.zeros((1,3))
lr = 0.5

for epoch in range(20000):
    # forward
    h = sigmoid(x @ w1 + b1)
    o = softmax(h @ w2 + b2)

    # backward
    d_o = (o - y) / x.shape[0]            # softmax + cross-entropy combined gradient
    d_h = (d_o @ w2.T) * dsigmoid(h)

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
