"""
autograd.py: a tiny gradient engine. This is the main trick inside PyTorch.

In every earlier script the backward pass was worked out by hand (d_o, d_h
and friends). Here the computer does that part. Each number is wrapped in a
Value, and a Value keeps notes on how it was made:

    data      the number itself
    parents   the Values it was calculated from
    slopes    one per parent: if that parent moves a tiny bit, this Value
              moves slope times as much
    op        the name of the calculation ("+", "*" or "sigmoid")
    grad      the blame: if this Value moves a tiny bit, the FINAL answer
              moves grad times as much. backward() fills it in.

backward() starts at the final answer and reads the notes in reverse,
handing every parent its share:   parent.grad += slope * grad

Part 1 checks the engine's blame against the wiggle test.
Part 2 trains a 2-4-1 network on XOR with no hand-written gradient code.

Run:
    python3 autograd.py
"""
import math
import random


class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self.parents = ()
        self.slopes = ()
        self.op = ""

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data)
        out.parents = (self, other)
        out.slopes = (1.0, 1.0)
        out.op = "+"
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data)
        out.parents = (self, other)
        out.slopes = (other.data, self.data)
        out.op = "*"
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s)
        out.parents = (self,)
        out.slopes = (s * (1 - s),)
        out.op = "sigmoid"
        return out

    def history(self):
        """Me and my ancestors, oldest first."""
        order, seen = [], set()

        def visit(value):
            if value not in seen:
                seen.add(value)
                for parent in value.parents:
                    visit(parent)
                order.append(value)

        visit(self)
        return order

    def backward(self):
        self.grad = 1.0
        for item in reversed(self.history()):
            notes = zip(item.parents, item.slopes)
            for parent, slope in notes:
                parent.grad += slope * item.grad

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


print("PART 1: is the engine's blame right? Check it by wiggling.")

WIGGLE = 0.000001
TARGET = 1.0


def warm_up(inputs):
    a, b, c = inputs
    return (a * b + c) * a


def one_neuron(inputs):
    x, w, b = inputs
    prediction = (w * x + b).sigmoid()
    miss = prediction + (-TARGET)
    return miss * miss


def show_notes(answer):
    print("  the notes the engine kept, in the order it calculated:")
    for value in answer.history():
        if len(value.parents) == 2:
            first, second = value.parents
            print(f"    {first.data:g} {value.op} {second.data:g} = {value.data:g}")
        elif value.parents:
            print(f"    {value.op}({value.parents[0].data:g}) = {value.data:g}")


def check(title, calculate, names, numbers):
    inputs = [Value(number) for number in numbers]
    answer = calculate(inputs)
    answer.backward()
    print(f"\n{title}")
    print("  " + ", ".join(f"{name} = {number:g}" for name, number in zip(names, numbers)))
    show_notes(answer)
    print("  input   backward() says   wiggling says")
    for name, value in zip(names, inputs):
        value.data += WIGGLE
        wiggled = calculate(inputs)
        value.data -= WIGGLE
        print(f"  {name:5s} {value.grad:15.6f} {(wiggled.data - answer.data) / WIGGLE:15.6f}")


check("warm-up: (a * b + c) * a", warm_up, "abc", [2.0, -3.0, 10.0])
check("one neuron and its loss: (sigmoid(w * x + b) - 1) squared", one_neuron, "xwb", [2.0, 0.5, -1.0])


print("\nPART 2: train a 2-4-1 network on XOR. No gradient formulas anywhere.")

HIDDEN = 4
LEARNING_RATE = 2.0
STEPS = 500
XOR = [((0, 0), 0), ((1, 0), 1), ((0, 1), 1), ((1, 1), 0)]

random.seed(13)
hidden_weights = [[Value(random.uniform(-1, 1)), Value(random.uniform(-1, 1))] for _ in range(HIDDEN)]
hidden_biases = [Value(0.0) for _ in range(HIDDEN)]
output_weights = [Value(random.uniform(-1, 1)) for _ in range(HIDDEN)]
output_bias = Value(0.0)
parameters = [w for pair in hidden_weights for w in pair] + hidden_biases + output_weights + [output_bias]


def predict(left, right):
    total = output_bias
    for (w_left, w_right), bias, w_out in zip(hidden_weights, hidden_biases, output_weights):
        hidden = (w_left * left + w_right * right + bias).sigmoid()
        total = total + hidden * w_out
    return total.sigmoid()


def total_loss():
    loss = Value(0.0)
    for (left, right), target in XOR:
        miss = predict(left, right) + (-target)
        loss = loss + miss * miss
    return loss


print(f"  {len(parameters)} numbers to learn")
print(f"  one pass over the 4 examples makes {len(total_loss().history())} Values, each with its notes\n")

for step in range(1, STEPS + 1):
    loss = total_loss()

    for p in parameters:
        p.grad = 0.0
    loss.backward()
    for p in parameters:
        p.data -= LEARNING_RATE * p.grad

    if step == 1 or step % 50 == 0:
        print(f"  step {step:4d}   loss {loss.data:.4f}")

print("\n  left  right   network says   want")
for (left, right), target in XOR:
    print(f"  {left:4d}  {right:5d}   {predict(left, right).data:12.3f}   {target:4d}")
