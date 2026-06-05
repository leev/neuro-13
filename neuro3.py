import numpy as np

# Step 3: 2-layer net (input -> hidden -> output). Same XOR rule as
# neuro2.py, but the hidden layer + non-linearity solve what the
# single-layer perceptron couldn't.
#
# Rule: "Secret door" (XOR).
#   The door opens if the left button OR the right button is pressed,
#   but NOT if both are pressed at the same time.
#
# x: shape (4, 2). columns = [left_button, right_button]
# y: shape (4, 1). 1 = door opens, 0 = shut. Pattern is XOR.
x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return x*(1-x)

np.random.seed(1)
hidden = 4   # try 4, then 2, then 1
w1 = 2*np.random.random((2, hidden)) - 1   # 2 inputs -> hidden
b1 = np.zeros((1, hidden))
w2 = 2*np.random.random((hidden, 1)) - 1   # hidden -> 1 out
b2 = np.zeros((1,1))
lr = 0.5

for epoch in range(20000):
    # forward
    h = sigmoid(x @ w1 + b1)
    o = sigmoid(h @ w2 + b2)

    # backward
    err   = y - o
    d_o   = err * dsigmoid(o)
    d_h   = (d_o @ w2.T) * dsigmoid(h)

    w2 += lr * h.T @ d_o
    b2 += lr * d_o.sum(axis=0, keepdims=True)
    w1 += lr * x.T @ d_h
    b1 += lr * d_h.sum(axis=0, keepdims=True)

print('Trained network — "Secret door (XOR)":')
print(f'  {"left":>7}  {"right":>7}  {"prediction":>10}  guess  truth  verdict')
for (left, right), pred, truth in zip(x, o, y):
    want    = "YES" if truth[0] > 0.5 else "NO "
    guess   = "YES" if pred[0]  > 0.5 else "NO "
    verdict = "OK   " if (pred[0] > 0.5) == (truth[0] > 0.5) else "WRONG"
    print(f'  {left:>7}  {right:>7}  {pred[0]:>10.3f}  {guess:>5}  {want:>5}  {verdict}')
