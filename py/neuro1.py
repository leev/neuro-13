"""
Step 1: one neuron learns the screen-time rule.

The rule: "Can I have screen time?" Yes if I practiced piano OR finished my
homework. One straight line can split the YES rows from the NO rows, so one
neuron is enough: two weights and a bias, a weighted sum, then sigmoid.

x has 4 rows and 2 columns: [practiced_piano, finished_homework].
y has 4 rows and 1 column: 1 = screen time allowed, 0 = not allowed.

No learning rate is written in the update, which is the same as a learning
rate of 1. This tiny dataset is friendly enough for that to work. From
neuro3.py on, the learning rate is written out (lr = 0.5).
"""
import numpy as np

x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[1]])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    # x must already be a sigmoid output: the slope of sigmoid at z is s * (1 - s), where s = sigmoid(z).
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

print('Trained neuron: "Can I have screen time?"')
print(f'  weights: piano {weights[0][0]:.2f}, homework {weights[1][0]:.2f}    bias: {bias:.2f}')
print()
print(f'  {"piano":>7}  {"homework":>8}  {"prediction":>10}  decision  truth')
for (hi, ho), pred, truth in zip(x, outputs, y):
    decision = "YES" if pred[0] > 0.5 else "NO "
    want     = "YES" if truth[0] > 0.5 else "NO "
    print(f'  {hi:>7}  {ho:>8}  {pred[0]:>10.3f}  {decision:>8}  {want}')

rows_right = int(((outputs > 0.5) == (y > 0.5)).sum())
print()
print(f'  The neuron gets {rows_right} of the 4 rows right.')
