"""
Attention from scratch: a network that learns where to look.

The task is a locker room. Each example is a row of lockers (owner:item) and
then a question that names one owner. The right answer is that owner's item:

    sam:hat   mia:ball   leo:book   ? mia   ->   ball

Owners and items are shuffled for every example, so the only way to answer
is to find the right locker and read what is inside. That is a lookup, and
this model does it with one attention head. In order, forward() computes:

    lockers   = owner row + item row            16 numbers per locker
    query     = asked owner's row @ W_query     the question
    keys      = lockers @ W_key                 each locker's label
    scores    = keys @ query                    a dot product per locker
    attention = softmax(scores)                 shares that add up to 1
    values    = lockers @ W_value               each locker's payload
    blend     = attention @ values              the weighted blend
    probs     = softmax(blend @ W_out + b_out)  one probability per item

Every grid of numbers in W starts out random. gradients() is backpropagation
written out by hand: the same steps as forward(), walked backwards. Nobody
ever tells the model which locker to look at. It is only told the answer.

It trains on rooms of 3 lockers. At the end it is tried on bigger rooms, and
on a question about somebody who has no locker at all.

Run:
    python3 attention.py
"""
import numpy as np

OWNERS = ["sam", "mia", "leo", "zoe", "ava", "max", "eli", "ivy"]
ITEMS = ["hat", "ball", "book", "kite", "drum", "map", "sock", "frog"]
SIZE = 16
LOCKERS = 3
BATCH = 16
LR = 0.1
STEPS = 1500
TEST_SIZE = 500

# numpy 2 on Apple Silicon prints bogus overflow warnings for small "@" products.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

rng = np.random.default_rng(13)

W = {
    "owner_rows": rng.normal(0, 0.1, (len(OWNERS), SIZE)),
    "item_rows": rng.normal(0, 0.1, (len(ITEMS), SIZE)),
    "W_query": rng.normal(0, 0.1, (SIZE, SIZE)),
    "W_key": rng.normal(0, 0.1, (SIZE, SIZE)),
    "W_value": rng.normal(0, 0.1, (SIZE, SIZE)),
    "W_out": rng.normal(0, 0.1, (SIZE, len(ITEMS))),
    "b_out": np.zeros(len(ITEMS)),
}
print(f"model: {sum(grid.size for grid in W.values()):,} weights, all random to begin with\n")


def new_example(lockers=LOCKERS):
    owners = rng.permutation(len(OWNERS))[:lockers]
    items = rng.permutation(len(ITEMS))[:lockers]
    asked = owners[rng.integers(lockers)]
    return owners, items, asked


def same_room_and_question(example):
    owners, items, asked = example
    return tuple(sorted(zip(owners.tolist(), items.tolist()))), int(asked)


def softmax(scores):
    raised = np.exp(scores - scores.max())
    return raised / raised.sum()


def forward(owners, items, asked):
    lockers = W["owner_rows"][owners] + W["item_rows"][items]
    query = W["owner_rows"][asked] @ W["W_query"]
    keys = lockers @ W["W_key"]
    scores = keys @ query
    attention = softmax(scores)
    values = lockers @ W["W_value"]
    blend = attention @ values
    probs = softmax(blend @ W["W_out"] + W["b_out"])
    return attention, probs, (lockers, query, keys, values, blend)


def gradients(owners, items, asked):
    attention, probs, (lockers, query, keys, values, blend) = forward(owners, items, asked)
    answer = items[owners == asked][0]

    d_out = probs.copy()
    d_out[answer] -= 1
    d_blend = W["W_out"] @ d_out
    d_values = np.outer(attention, d_blend)
    d_attention = values @ d_blend
    d_scores = attention * (d_attention - attention @ d_attention)
    d_query = d_scores @ keys
    d_keys = np.outer(d_scores, query)
    d_lockers = d_keys @ W["W_key"].T + d_values @ W["W_value"].T

    grad = {
        "W_out": np.outer(blend, d_out),
        "b_out": d_out,
        "W_value": lockers.T @ d_values,
        "W_key": lockers.T @ d_keys,
        "W_query": np.outer(W["owner_rows"][asked], d_query),
        "owner_rows": np.zeros_like(W["owner_rows"]),
        "item_rows": np.zeros_like(W["item_rows"]),
    }
    np.add.at(grad["owner_rows"], owners, d_lockers)
    np.add.at(grad["item_rows"], items, d_lockers)
    grad["owner_rows"][asked] += W["W_query"] @ d_query
    return grad


def check(examples):
    right = focus = 0
    for owners, items, asked in examples:
        attention, probs, _ = forward(owners, items, asked)
        right += probs.argmax() == items[owners == asked][0]
        focus += attention[owners == asked][0]
    return (f"answers right {right / len(examples):.3f}   "
            f"attention on the right locker {focus / len(examples):.2f}")


def show(owners, items, asked):
    attention, probs, _ = forward(owners, items, asked)
    row = "   ".join(f"{OWNERS[o]}:{ITEMS[i]} {a:.2f}" for o, i, a in zip(owners, items, attention))
    print(f"  {row}   ? {OWNERS[asked]} -> {ITEMS[probs.argmax()]} ({probs.max():.0%} sure)")


test_set = [new_example() for _ in range(TEST_SIZE)]
held_back = {same_room_and_question(example) for example in test_set}


def new_training_example():
    while True:
        example = new_example()
        if same_room_and_question(example) not in held_back:
            return example


demo = (np.array([0, 1, 2]), np.array([0, 1, 2]), 1)
print("before training, the attention on each locker and the model's answer:")
show(*demo)

print(f"\ntraining on rooms of {LOCKERS} lockers, {BATCH} rooms per step,")
print(f"scored on {TEST_SIZE} test rooms it never trains on")
print(f"step {0:4d}   {check(test_set)}")
for step in range(1, STEPS + 1):
    total = {name: np.zeros_like(grid) for name, grid in W.items()}
    for _ in range(BATCH):
        for name, g in gradients(*new_training_example()).items():
            total[name] += g
    for name in W:
        W[name] -= LR * total[name] / BATCH
    if step % 100 == 0:
        print(f"step {step:4d}   {check(test_set)}")

print("\nafter training, the same question:")
show(*demo)

print(f"\nrooms of other sizes (it only ever trained on {LOCKERS} lockers):")
for lockers in range(2, len(OWNERS) + 1):
    print(f"  {lockers} lockers   {check([new_example(lockers) for _ in range(500)])}")

print("\nasking about max, who has no locker in the room (there is no right answer):")
show(np.array([0, 1, 2]), np.array([0, 1, 2]), OWNERS.index("max"))
