import numpy as np

# Step 2: same single-layer perceptron as neuro1.py, but XOR rule.
# This is INTENDED TO FAIL. The point is to see the failure mode
# and understand why a hidden layer is needed (see neuro3.py).
#
# Rule we're trying to teach: "Secret door" (XOR).
#   The door opens if the left button OR the right button is pressed,
#   but NOT if both are pressed at the same time.
#
# x: shape (4, 2). 4 samples, 2 binary features.
#   columns = [left_button, right_button]
# y: shape (4, 1). 1 = door opens, 0 = door stays shut.
#   Pattern: left_button XOR right_button.
# Diagonal corners share a class -> no single straight line separates
# them. A 1-layer net can only draw a single straight line, so it
# CANNOT solve this. Outputs collapse to ~0.5 for every input and
# error never shrinks.
x = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

np.random.seed(1)
weights = 2 * np.random.random((2, 1)) - 1

# Same effective learning rate (lr = 1) as neuro1.py. From neuro3.py
# onward we write the learning rate out explicitly.
for epoch in range(10000):
    input_layer = x
    weighted_sum = np.dot(input_layer, weights)
    outputs = sigmoid(weighted_sum)

    error = y - outputs
    adjustments = error * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)

    if epoch % 2000 == 0:
        mse = float(np.mean(error**2))
        print(f"epoch {epoch}: mse {mse:.4f}")

print()
print('After 10000 epochs — "Secret door (XOR)":')
print(f'  {"left":>7}  {"right":>7}  {"prediction":>10}  guess  truth  verdict')
for (left, right), pred, truth in zip(x, outputs, y):
    want    = "YES" if truth[0] > 0.5 else "NO "
    guess   = "YES" if pred[0]  > 0.5 else "NO "
    verdict = "OK   " if (pred[0] > 0.5) == (truth[0] > 0.5) else "WRONG"
    print(f'  {left:>7}  {right:>7}  {pred[0]:>10.3f}  {guess:>5}  {want:>5}  {verdict}')

print()
print("  Output is ~0.5 for every input — single-layer perceptron")
print("  cannot solve XOR. This is the failure that motivates hidden")
print("  layers. See neuro3.py for the fix.")
