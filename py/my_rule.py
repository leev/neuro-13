"""
Teach one neuron your own rule.

The neuron from neuro1.py, with its learning rate written out, learns the
four-row table in ANSWERS. That table is the new part, and it is yours to
change. Each row reads (input 1, input 2): answer, with 1 for YES and 0 for
NO. It starts as AND.

    AND   0, 0, 0, 1      NAND  1, 1, 1, 0      SAME  1, 0, 0, 1

Before you run it, predict: can ONE straight line split the YES dots from
the NO dots? If not, the output shows you what being stuck looks like.
"""
import numpy as np

RULE = "sleepover allowed?"
INPUT_1 = "room clean"
INPUT_2 = "homework done"
ANSWERS = {
    (0, 0): 0,
    (1, 0): 0,
    (0, 1): 0,
    (1, 1): 1,
}
ROUNDS = 5000
LEARNING_RATE = 0.5

x = np.array(list(ANSWERS.keys()))
y = np.array(list(ANSWERS.values())).reshape(4, 1)


def sigmoid(total):
    return 1 / (1 + np.exp(-total))


def in_words(score):
    if score > 0.6:
        return "YES"
    if score < 0.4:
        return "NO"
    return "shrug"


np.random.seed(1)
weights = 2 * np.random.random((2, 1)) - 1
bias = 0.0

print(f'Teaching one neuron the rule "{RULE}"')
for round_number in range(ROUNDS + 1):
    prediction = sigmoid(x @ weights + bias)
    error = y - prediction
    if round_number % 1000 == 0:
        print(f"  round {round_number:>4}: average squared error {np.mean(error ** 2):.4f}")
    nudge = LEARNING_RATE * error * prediction * (1 - prediction)
    weights += x.T @ nudge
    bias += nudge.sum()

weight_1, weight_2 = weights[:, 0]
print()
print(f"The whole model: weight 1 = {weight_1:.2f}, weight 2 = {weight_2:.2f}, bias = {bias:.2f}")
print()
print(f"  {INPUT_1:>14}  {INPUT_2:>14}  prediction  neuron says  you said")
rows_right = 0
for (first, second), score, truth in zip(x, prediction[:, 0], y[:, 0]):
    you_said = "YES" if truth == 1 else "NO"
    rows_right += in_words(score) == you_said
    print(f"  {first:>14}  {second:>14}  {score:>10.3f}  {in_words(score):>11}  {you_said:>8}")

print()
print("The neuron's answer all over the graph paper, from 0 (sure NO) to 9 (sure YES).")
print(f"Left to right is {INPUT_1}, bottom to top is {INPUT_2}, and the corners are your four rows:")
print()
for second in (1, 0.75, 0.5, 0.25, 0):
    scores = [sigmoid(weight_1 * first + weight_2 * second + bias) for first in np.linspace(0, 1, 9)]
    print("    " + "  ".join(str(min(9, int(score * 10 + 0.5))) for score in scores))

print()
if rows_right == 4:
    print("LEARNED IT. One straight line splits your YES dots from your NO dots,")
    print("and the neuron found it: look for the edge where the map crosses 5.")
else:
    print(f"STUCK. The neuron matched {rows_right} of your 4 rows. If more ROUNDS do not help,")
    print("no straight line can split this rule's YES dots from its NO dots, and one")
    print("neuron can never learn it. It needs hidden neurons: see neuro3.py.")
