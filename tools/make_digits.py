#!/usr/bin/env python3
"""
Build py/digits.bin, a balanced sample of the MNIST handwritten digits.

    python3 tools/make_digits.py data/MNIST/raw py/digits.bin

The first argument is a folder holding the four original MNIST files
(train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte,
t10k-labels-idx1-ubyte). Training images are drawn from MNIST's training set
and test images from MNIST's own test set, so the two never overlap. Each
digit 0-9 gets the same number of images, picked and shuffled with a fixed
seed.

MNIST is by Yann LeCun, Corinna Cortes and Christopher J.C. Burges,
licensed CC BY-SA 3.0.

Byte layout of digits.bin (every number is unsigned, multi-byte ones are
little-endian):

    offset            bytes           what
    0                 4               the letters "DGT1"
    4                 4               n_train, a uint32
    8                 4               n_test, a uint32
    12                4               side, a uint32 (28: images are 28 x 28)
    16                n_train         training labels, one byte each (0-9)
    16 + n_train      n_test          test labels, one byte each
    then              n_train * 784   training images
    then              n_test * 784    test images

An image is 784 bytes, row by row from the top-left corner. 0 is blank paper
and 255 is full ink.

With --samples the tool also prints one test image per digit as base64: for
each digit, the test image closest to the average of all its test images.
That is how the ten digits built into src/widgets/digits.js were made.
"""
import argparse
import base64
import struct
from pathlib import Path

import numpy as np

SEED = 13
SIDE = 28


def read_idx(path):
    raw = Path(path).read_bytes()
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2049:
        return np.frombuffer(raw, dtype=np.uint8, offset=8)
    if magic != 2051:
        raise SystemExit(f"{path} is not an MNIST idx file")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, SIDE * SIDE)


def balanced_sample(images, labels, per_digit, rng):
    picked = np.concatenate([
        rng.choice(np.flatnonzero(labels == digit), per_digit, replace=False)
        for digit in range(10)
    ])
    rng.shuffle(picked)
    return images[picked], labels[picked]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mnist_dir")
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()

    src = Path(args.mnist_dir)
    rng = np.random.default_rng(SEED)
    train_x, train_y = balanced_sample(
        read_idx(src / "train-images-idx3-ubyte"), read_idx(src / "train-labels-idx1-ubyte"),
        args.train // 10, rng)
    test_x, test_y = balanced_sample(
        read_idx(src / "t10k-images-idx3-ubyte"), read_idx(src / "t10k-labels-idx1-ubyte"),
        args.test // 10, rng)

    with open(args.out, "wb") as f:
        f.write(b"DGT1")
        f.write(struct.pack("<III", len(train_y), len(test_y), SIDE))
        f.write(train_y.tobytes())
        f.write(test_y.tobytes())
        f.write(train_x.tobytes())
        f.write(test_x.tobytes())
    size = Path(args.out).stat().st_size
    print(f"wrote {len(train_y)} training + {len(test_y)} test images to {args.out} ({size / 1e6:.2f} MB)")

    if args.samples:
        for digit in range(10):
            group = test_x[test_y == digit].astype(np.float64)
            typical = ((group - group.mean(axis=0)) ** 2).sum(axis=1).argmin()
            print(digit, base64.b64encode(group[typical].astype(np.uint8).tobytes()).decode())


if __name__ == "__main__":
    main()
