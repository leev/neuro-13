import numpy as np

# Step 1: single-layer perceptron, OR rule.
# Linearly separable -> one weighted sum + sigmoid is enough.
#
# Rule we're teaching: "Can I have screen time?"
#   Yes if I practiced piano OR I finished my homework.
#
# x: shape (4, 2). 4 samples, 2 binary features.
#   columns = [practiced_piano, finished_homework]
# y: shape (4, 1). 1 = screen time allowed, 0 = no screen time.
x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[1]])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    # NOTE: this is sigmoid's derivative *evaluated at the post-sigmoid value*.
    # If `s = sigmoid(z)` then sigmoid'(z) = s * (1 - s). Below we always call
    # `sigmoid_derivative(outputs)` where outputs is already sigmoid'd, so the
    # math works out. If you ever call this on a raw pre-activation value you
    # will get the wrong answer.
    return x*(1-x)

np.random.seed(1)
weights = 2 * np.random.random((2, 1)) - 1

# NOTE: there's no explicit learning rate on the update below — we're
# effectively using lr = 1. The toy OR data is friendly enough that this
# works. Starting in neuro3.py we write the learning rate out explicitly
# (lr = 0.5) so you can see what it does.
for epoch in range(10000):
    input_layer = x
    weighted_sum = np.dot(input_layer, weights)
    outputs = sigmoid(weighted_sum)

    error = y - outputs
    adjustments = error * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)

print('Trained network — "Can I have screen time?"')
print('  rule learned: screen time allowed if piano OR homework is done')
print()
print(f'  {"piano":>7}  {"homework":>8}  {"prediction":>10}  decision  truth')
for (hi, ho), pred, truth in zip(x, outputs, y):
    decision = "YES" if pred[0] > 0.5 else "NO "
    want     = "YES" if truth[0] > 0.5 else "NO "
    print(f'  {hi:>7}  {ho:>8}  {pred[0]:>10.3f}  {decision:>8}  {want}')
