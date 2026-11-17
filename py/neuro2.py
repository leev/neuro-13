"""
Step 2: the same neuron as neuro1.py, given a rule it cannot learn.

The rule: "Secret door" (XOR). The door opens if the left button OR the right
button is pressed, but NOT if both are pressed at the same time.

x has 4 rows and 2 columns: [left_button, right_button].
y has 4 rows and 1 column: 1 = door opens, 0 = door stays shut.

The two YES rows sit on opposite corners of the graph paper, so no single
straight line separates them from the NO rows, and one neuron can only draw
one line. Watch the error (mse): it stops shrinking. neuro3.py has the fix.

As in neuro1.py, no learning rate is written, which means a learning rate of 1.
"""
import numpy as np

x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

np.random.seed(1)
weights = 2 * np.random.random((2, 1)) - 1
bias = 0.0

for epoch in range(10000):
    input_layer = x
    weighted_sum = np.dot(input_layer, weights) + bias
    outputs = sigmoid(weighted_sum)

    error = y - outputs
    adjustments = error * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)
    bias += adjustments.sum()

    if epoch % 2000 == 0:
        mse = float(np.mean(error**2))
        print(f"epoch {epoch}: mse {mse:.4f}")

print()
print('After 10000 epochs: "Secret door (XOR)"')
print(f'  weights: left {weights[0][0]:.2f}, right {weights[1][0]:.2f}    bias: {bias:.2f}')
print()
print(f'  {"left":>7}  {"right":>7}  {"prediction":>10}  guess  truth  verdict')
for (left, right), pred, truth in zip(x, outputs, y):
    want    = "YES" if truth[0] > 0.5 else "NO "
    guess   = "YES" if pred[0]  > 0.5 else "NO "
    verdict = "OK   " if (pred[0] > 0.5) == (truth[0] > 0.5) else "WRONG"
    print(f'  {left:>7}  {right:>7}  {pred[0]:>10.3f}  {guess:>5}  {want:>5}  {verdict}')

rows_wrong = int(((outputs > 0.5) != (y > 0.5)).sum())
print()
if rows_wrong:
    print(f"  {rows_wrong} of the 4 rows are wrong. If more epochs do not help, no single")
    print("  straight line fits this rule, and one neuron can never learn it.")
    print("  See neuro3.py for the fix: a hidden layer.")
else:
    print("  All 4 rows are right: one straight line fits this rule.")
