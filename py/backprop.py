"""
Backpropagation by hand, on the smallest network that can learn XOR.

The network is the one in the chapter's widget, with the same starting numbers:

    left  button ─┬─> H1 ─┐
                  ╳       ├─> out ─> prediction
    right button ─┴─> H2 ─┘

Two inputs, two hidden neurons, one output neuron, sigmoid everywhere.
Nine numbers to tune: six weights (one per wire) and three biases.

"Blame" is this book's word for the gradient: how fast the loss goes up
when a number goes up by a tiny bit. The script has four parts:

  Part 1  one round on one example, slowly, printing every step
  Part 2  the same steps packed into two short functions
  Part 3  the wiggle check: nudge each number by 0.001, see how far the loss
          moves, and compare with the blame from Part 1
  Part 4  apply the nudges, run 2,000 more rounds, print the four answers

Run:
    python3 backprop.py
"""
import numpy as np

LEARNING_RATE = 2.0
WIGGLE = 0.001
MORE_ROUNDS = 2000
EXAMPLES = [((0, 0), 0), ((1, 0), 1), ((0, 1), 1), ((1, 1), 0)]
FIRST_EXAMPLE = 1

net = {
    "left->H1": 0.8, "right->H1": -0.7, "bias H1": -0.3,
    "left->H2": -0.4, "right->H2": 0.9, "bias H2": 0.2,
    "H1->out": 0.5, "H2->out": -0.6, "bias out": -0.1,
}


def squish(total):
    return 1 / (1 + np.exp(-total))


def show(value, places=4):
    text = f"{value:.{places}f}"
    return text.lstrip("-") if float(text) == 0 else text


(left, right), target = EXAMPLES[FIRST_EXAMPLE]
print(f"PART 1: one round by hand on left={left}, right={right} (target {target})")

print("\nforward pass")
h1_total = left * net["left->H1"] + right * net["right->H1"] + net["bias H1"]
h1 = squish(h1_total)
print(f"  H1 total   = {left} x {show(net['left->H1'], 3)} + {right} x {show(net['right->H1'], 3)} + {show(net['bias H1'], 3)} = {show(h1_total, 3)}")
print(f"  H1 output  = squish({show(h1_total, 3)}) = {show(h1, 3)}")
h2_total = left * net["left->H2"] + right * net["right->H2"] + net["bias H2"]
h2 = squish(h2_total)
print(f"  H2 total   = {left} x {show(net['left->H2'], 3)} + {right} x {show(net['right->H2'], 3)} + {show(net['bias H2'], 3)} = {show(h2_total, 3)}")
print(f"  H2 output  = squish({show(h2_total, 3)}) = {show(h2, 3)}")
out_total = h1 * net["H1->out"] + h2 * net["H2->out"] + net["bias out"]
prediction = squish(out_total)
print(f"  out total  = {show(h1, 3)} x {show(net['H1->out'], 3)} + {show(h2, 3)} x {show(net['H2->out'], 3)} + {show(net['bias out'], 3)} = {show(out_total, 3)}")
print(f"  prediction = squish({show(out_total, 3)}) = {show(prediction, 3)}")

print("\nthe loss")
miss = prediction - target
loss = 0.5 * miss * miss
print(f"  miss = prediction - target = {show(prediction, 3)} - {target} = {show(miss, 3)}")
print(f"  loss = half of miss x miss = {show(loss)}")

print("\nbackward pass: blame at out, then the three numbers feeding out")
out_slope = prediction * (1 - prediction)
blame_at_out = miss * out_slope
print(f"  blame at out = miss x slope of out's squish = {show(miss, 3)} x {show(out_slope, 3)} = {show(blame_at_out)}")
blame = {
    "H1->out": blame_at_out * h1,
    "H2->out": blame_at_out * h2,
    "bias out": blame_at_out,
}
print(f"  H1->out   = blame at out x H1's output = {show(blame_at_out)} x {show(h1, 3)} = {show(blame['H1->out'])}")
print(f"  H2->out   = blame at out x H2's output = {show(blame_at_out)} x {show(h2, 3)} = {show(blame['H2->out'])}")
print(f"  bias out  = blame at out x 1 = {show(blame['bias out'])}")

print("\nhand the blame back: blame arriving x weight on the wire it came along x slope of the squish")
h1_slope = h1 * (1 - h1)
blame_at_h1 = blame_at_out * net["H1->out"] * h1_slope
print(f"  blame at H1 = {show(blame_at_out)} x {show(net['H1->out'], 3)} x {show(h1_slope, 3)} = {show(blame_at_h1)}")
h2_slope = h2 * (1 - h2)
blame_at_h2 = blame_at_out * net["H2->out"] * h2_slope
print(f"  blame at H2 = {show(blame_at_out)} x {show(net['H2->out'], 3)} x {show(h2_slope, 3)} = {show(blame_at_h2)}")

print("\nthe numbers feeding H1 and H2: blame at the neuron x the signal the wire carried")
blame["left->H1"] = blame_at_h1 * left
blame["right->H1"] = blame_at_h1 * right
blame["bias H1"] = blame_at_h1
blame["left->H2"] = blame_at_h2 * left
blame["right->H2"] = blame_at_h2 * right
blame["bias H2"] = blame_at_h2
for name in ["left->H1", "right->H1", "bias H1", "left->H2", "right->H2", "bias H2"]:
    print(f"  {name:10s}= {show(blame[name])}")


print("\nPART 2: the same steps as functions, so they can be repeated (nothing to print)")


def forward(net, left, right):
    h1 = squish(left * net["left->H1"] + right * net["right->H1"] + net["bias H1"])
    h2 = squish(left * net["left->H2"] + right * net["right->H2"] + net["bias H2"])
    prediction = squish(h1 * net["H1->out"] + h2 * net["H2->out"] + net["bias out"])
    return h1, h2, prediction


def backward(net, left, right, target):
    h1, h2, prediction = forward(net, left, right)
    miss = prediction - target
    blame_at_out = miss * (prediction * (1 - prediction))
    blame_at_h1 = blame_at_out * net["H1->out"] * (h1 * (1 - h1))
    blame_at_h2 = blame_at_out * net["H2->out"] * (h2 * (1 - h2))
    return {
        "left->H1": blame_at_h1 * left, "right->H1": blame_at_h1 * right, "bias H1": blame_at_h1,
        "left->H2": blame_at_h2 * left, "right->H2": blame_at_h2 * right, "bias H2": blame_at_h2,
        "H1->out": blame_at_out * h1, "H2->out": blame_at_out * h2, "bias out": blame_at_out,
    }


def loss_on(net, left, right, target):
    miss = forward(net, left, right)[2] - target
    return 0.5 * miss * miss


def average_loss(net):
    return sum(loss_on(net, l, r, t) for (l, r), t in EXAMPLES) / len(EXAMPLES)


print(f"\nPART 3: the wiggle check (add {WIGGLE} to one number, see how far the loss moves)\n")
print("  number       backprop     wiggling")
loss_before = loss_on(net, left, right, target)
forward_passes = 1
biggest_gap = 0
for name in net:
    wiggled = dict(net)
    wiggled[name] += WIGGLE
    measured = (loss_on(wiggled, left, right, target) - loss_before) / WIGGLE
    forward_passes += 1
    biggest_gap = max(biggest_gap, abs(measured - blame[name]))
    print(f"  {name:10s} {show(blame[name], 5):>10}   {show(measured, 5):>10}")
print(f"\n  biggest gap between the two columns: {biggest_gap:.6f}")
print(f"  wiggling needed {forward_passes} forward passes; backprop needed 1 forward and 1 backward")


print(f"\nPART 4: apply the nudges (new = old - {LEARNING_RATE} x blame), then keep going\n")
for name in net:
    old = net[name]
    net[name] = old - LEARNING_RATE * blame[name]
    print(f"  {name:10s} {show(old, 3):>7} -> {show(net[name], 3):>7}")
new_prediction = forward(net, left, right)[2]
print(f"\n  prediction for this example: {show(prediction, 3)} -> {show(new_prediction, 3)}")
print(f"  loss for this example:       {show(loss)} -> {show(loss_on(net, left, right, target))}")

print(f"\n{MORE_ROUNDS:,} more rounds, taking the four examples in turn")
print(f"  after {1:5,} round    average loss {show(average_loss(net))}")
for round_number in range(1, MORE_ROUNDS + 1):
    (left, right), target = EXAMPLES[(round_number - 1) % len(EXAMPLES)]
    blame = backward(net, left, right, target)
    for name in net:
        net[name] -= LEARNING_RATE * blame[name]
    if round_number % 250 == 0:
        print(f"  after {round_number + 1:5,} rounds   average loss {show(average_loss(net))}")

print("\nthe trained network")
for (left, right), target in EXAMPLES:
    prediction = forward(net, left, right)[2]
    verdict = "right" if (prediction > 0.5) == (target > 0.5) else "WRONG"
    print(f"  left={left}, right={right}   prediction {show(prediction, 3)}   target {target}   {verdict}")
print("\nits nine numbers now")
for name in net:
    print(f"  {name:10s} {show(net[name], 3):>7}")
