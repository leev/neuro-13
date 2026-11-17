"""
The networks behind the chapter "How networks see".

Two networks learn from the same 55,000 handwritten digits (MNIST):

    plain : 784 pixels -> 64 hidden neurons -> 10 scores
            (the kind of network from the chapter before: one weight per pixel)

    conv  : 8 filters of 5x5 -> ReLU -> 2x2 max pooling
            -> 32 filters of 5x5 -> ReLU
            -> keep only the strongest response in each of the 32 maps
            -> 10 scores

Then come the experiments the chapter talks about:

    the shift test   : push every test digit sideways and see who can still read it
    remember where?  : the same filters, but a last layer with a weight for every position
    scrambled pixels : shuffle the pixels the same way in every picture, then train again
    counting weights : how many numbers each network stores

No network here ever trains on a shifted digit. Whatever happens in the shift
test comes from how the network is wired, not from what it was shown.

Add --save to also write convnet-weights.json, the file the web page loads.
To keep that file small each weight is rounded to one of 255 levels, and the
script measures the rounded networks again, because those are the ones your
browser runs.

Run:
    pip install torch numpy
    python3 convnet.py
The first run downloads MNIST (11 MB). Expect a few minutes; a graphics chip
makes it much quicker. Your numbers can differ from the book's in the last
digit, because different computers add up in a slightly different order.
"""
import base64
import gzip
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 13
EPOCHS = 8
BATCH = 64
LR = 0.002
MAX_SHIFT = 6
DIRECTIONS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
SAMPLES = [10, 5, 1, 32, 27, 52, 11, 0, 177, 12]

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "MNIST" / "raw"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def read_mnist(name):
    path = DATA / (name + ".gz")
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name}.gz")
        urllib.request.urlretrieve(MIRROR + name + ".gz", path)
    raw = gzip.open(path).read()
    if "images" in name:
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28)
    return np.frombuffer(raw, np.uint8, offset=8)


def as_inputs(images):
    return torch.tensor(images.astype(np.float32) / 255).to(DEVICE)


def as_answers(labels):
    return torch.tensor(labels.astype(np.int64)).to(DEVICE)


test_pixels = read_mnist("t10k-images-idx3-ubyte")
all_x, all_y = as_inputs(read_mnist("train-images-idx3-ubyte")), as_answers(read_mnist("train-labels-idx1-ubyte"))
test_x, test_y = as_inputs(test_pixels), as_answers(read_mnist("t10k-labels-idx1-ubyte"))

# The last 5,000 training digits are a practice test for watching progress.
# The real test set is only touched after training has finished.
train_x, train_y = all_x[:55000], all_y[:55000]
practice_x, practice_y = all_x[55000:], all_y[55000:]
print(f"{len(train_x)} digits to learn from, {len(practice_x)} for practice tests, {len(test_x)} for the final test, on {DEVICE}")


class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(28 * 28, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x.flatten(1))))


class ConvNet(nn.Module):
    def __init__(self, remember_where=False):
        super().__init__()
        self.remember_where = remember_where
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, padding=2)
        self.out = nn.Linear(32 * 7 * 7 if remember_where else 32, 10)

    def forward(self, x):
        maps = F.relu(self.conv1(x[:, None]))           # 8 maps of 28x28
        pooled = F.max_pool2d(maps, 2)                  # 8 maps of 14x14
        maps2 = F.relu(self.conv2(pooled))              # 32 maps of 14x14
        if self.remember_where:
            return self.out(F.max_pool2d(maps2, 2).flatten(1))
        return self.out(maps2.amax(dim=(2, 3)))         # 32 numbers: "how strongly, anywhere?"


@torch.no_grad()
def count_right(net, x, y):
    return sum((net(x[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item() for i in range(0, len(x), 1000))


def train(name, build, x=train_x, practice=practice_x):
    torch.manual_seed(SEED)
    net = build().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    step, total_steps = 0, EPOCHS * math.ceil(len(x) / BATCH)
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(x), device=DEVICE)
        for start in range(0, len(x), BATCH):
            batch = order[start:start + BATCH]
            optimizer.param_groups[0]["lr"] = LR * (1 - step / total_steps)   # smaller steps as training goes on
            loss = F.cross_entropy(net(x[batch]), train_y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        print(f"  {name}, epoch {epoch}/{EPOCHS}: {count_right(net, practice, practice_y) / 50:.1f}% of the practice digits right")
    return net


def shifted(x, dx, dy):
    """Move every picture dx pixels right and dy pixels down. What slides off the edge is gone."""
    m = MAX_SHIFT
    return F.pad(x, (m, m, m, m))[:, m - dy:m - dy + 28, m - dx:m - dx + 28]


def shift_test(net):
    return {name: [count_right(net, shifted(test_x, dx * k, dy * k), test_y) for k in range(MAX_SHIFT + 1)]
            for name, (dx, dy) in DIRECTIONS.items()}


def pack(net):
    """Round every weight to a whole number from -127 to 127 times one scale per layer."""
    packed = {}
    for name, tensor in net.state_dict().items():
        values = tensor.cpu().numpy()
        if name.endswith("bias"):
            packed[name] = {"shape": list(values.shape), "values": [round(float(v), 5) for v in values]}
        else:
            scale = float(np.abs(values).max()) / 127
            levels = np.round(values / scale).astype(np.int8)
            packed[name] = {"shape": list(values.shape), "scale": scale, "q": base64.b64encode(levels.tobytes()).decode()}
    return packed


def unpack(net, packed):
    state = {}
    for name, t in packed.items():
        flat = t["values"] if "values" in t else np.frombuffer(base64.b64decode(t["q"]), np.int8) * t["scale"]
        state[name] = torch.tensor(np.asarray(flat, np.float32).reshape(t["shape"]))
    net.load_state_dict(state)
    return net.to(DEVICE)


def size(net):
    return sum(p.numel() for p in net.parameters())


def percent(right):
    return f"{right / 100:5.1f}%"


print("\nTRAINING")
plain = train("plain", PlainNet)
conv = train("conv", ConvNet)
where = train("conv that remembers where", lambda: ConvNet(remember_where=True))

packed = {"plain": pack(plain), "conv": pack(conv)}
before = {"plain": shift_test(plain), "conv": shift_test(conv), "where": shift_test(where)}
plain, conv = unpack(plain, packed["plain"]), unpack(conv, packed["conv"])
measured = {"plain": shift_test(plain), "conv": shift_test(conv)}

print("\nTHE SHIFT TEST (10,000 test digits; nobody trained on shifted digits)")
for direction in DIRECTIONS:
    print(f"\n  pushed {direction:5s}       plain      conv   conv that remembers where")
    for k in range(MAX_SHIFT + 1):
        print(f"  {k} pixels        {percent(measured['plain'][direction][k])}    {percent(measured['conv'][direction][k])}    {percent(before['where'][direction][k])}")

print("\nDID ROUNDING THE WEIGHTS TO 8 BITS HURT? (right out of 10,000, unshifted and 4 pixels right)")
for kind in ("plain", "conv"):
    full, small = before[kind]["right"], measured[kind]["right"]
    print(f"  {kind:5s}  before rounding {full[0]} and {full[4]}    after rounding {small[0]} and {small[4]}")

print("\nSCRAMBLED PIXELS (one fixed shuffle, the same for every picture, then train from scratch)")
shuffle = torch.randperm(28 * 28, generator=torch.Generator().manual_seed(SEED)).to(DEVICE)


def scramble(x):
    return x.flatten(1)[:, shuffle].reshape(-1, 28, 28)


scrambled = {name: count_right(train(name + " on scrambled digits", build, scramble(train_x), scramble(practice_x)), scramble(test_x), test_y)
             for name, build in (("plain", PlainNet), ("conv", ConvNet))}
for kind in ("plain", "conv"):
    print(f"  {kind:5s}  {percent(measured[kind]['right'][0])} of the normal test digits right, {percent(scrambled[kind])} of the scrambled ones")

print("\nCOUNTING WEIGHTS (biases included)")
photo = 4000 * 3000 * 3
print(f"  plain network: {size(plain):,}      conv network: {size(conv):,}      conv that remembers where: {size(where):,}")
print(f"  first layer of the plain network: 784 x 64 = {784 * 64:,} weights + 64 biases      first layer of the conv network: {size(conv.conv1):,}")
print(f"  conv network, layer by layer: {size(conv.conv1):,} + {size(conv.conv2):,} + {size(conv.out):,}")
print(f"  one 5x5 filter on a gray picture: {5 * 5} weights + 1 bias, whether the picture is 28x28 or 4000x3000")
print(f"  one 5x5 filter on a color photo : {5 * 5 * 3} weights + 1 bias (it reads red, green and blue)")
print(f"  64 plain neurons on a 12-megapixel color photo: {photo:,} inputs x 64 = {photo * 64:,} weights")

if "--save" in sys.argv:
    digit_of = {i: str(int(test_y[i])) for i in SAMPLES}
    with torch.no_grad():
        scores = {kind: {digit_of[i]: [round(float(s), 4) for s in net(test_x[i:i + 1])[0]] for i in SAMPLES}
                  for kind, net in (("plain", plain), ("conv", conv))}
    file = {
        "about": "Weights for the two networks in the Neuro 13 chapter 'How networks see', written by convnet.py. "
                 "Each weight is q (base64, signed 8-bit) times scale.",
        "plain": packed["plain"],
        "conv": packed["conv"],
        "measured": {"digits": len(test_x), **measured},
        "samples": {digit_of[i]: base64.b64encode(test_pixels[i].tobytes()).decode() for i in SAMPLES},
        "check": scores,
    }
    (HERE / "convnet-weights.json").write_text(json.dumps(file, separators=(",", ":")) + "\n")
    print(f"\nwrote convnet-weights.json ({(HERE / 'convnet-weights.json').stat().st_size / 1000:.0f} KB)")
else:
    print("\n(run with --save to write convnet-weights.json)")
