import numpy as np

# Step 6: animal classifier. This is the snack picker idea with a
# more interesting dataset: guess an animal from visible traits.
#
# New concept: categorical features become one-hot inputs.
#   size = small / medium / large
# becomes three input columns:
#   size_small, size_medium, size_large
#
# The traits below are intentionally shared by several animals. That
# makes the network combine clues instead of relying on one giveaway
# clue like "has trunk".
#
# Inputs:
#   size_small, size_medium, size_large,
#   place_home, place_farm, place_wild,
#   food_meat, food_plants, food_both,
#   feet_paws, feet_hooves,
#   color_brown, color_orange, color_black_white, color_pink
#
# Outputs:
#   dog, cat, rabbit, pig, horse, cow, lion, tiger

ANIMALS = ["dog", "cat", "rabbit", "pig", "horse", "cow", "lion", "tiger"]
FEATURES = [
    "size_small", "size_medium", "size_large",
    "place_home", "place_farm", "place_wild",
    "food_meat", "food_plants", "food_both",
    "feet_paws", "feet_hooves",
    "color_brown", "color_orange", "color_black_white", "color_pink",
]

x = np.array([
    # sm med lg  home farm wild  meat plants both  paws hooves  brown orange b/w pink
    [0, 1, 0,   1,   0,    0,   0,    0,     1,    1,   0,     1,    0,     0,  0],  # dog
    [1, 0, 0,   1,   0,    0,   1,    0,     0,    1,   0,     0,    1,     0,  0],  # cat
    [1, 0, 0,   1,   0,    0,   0,    1,     0,    1,   0,     1,    0,     0,  0],  # rabbit
    [0, 1, 0,   0,   1,    0,   0,    0,     1,    0,   1,     0,    0,     0,  1],  # pig
    [0, 0, 1,   0,   1,    0,   0,    1,     0,    0,   1,     1,    0,     0,  0],  # horse
    [0, 0, 1,   0,   1,    0,   0,    1,     0,    0,   1,     0,    0,     1,  0],  # cow
    [0, 0, 1,   0,   0,    1,   1,    0,     0,    1,   0,     1,    0,     0,  0],  # lion
    [0, 0, 1,   0,   0,    1,   1,    0,     0,    1,   0,     0,    1,     0,  0],  # tiger
], dtype=float)

y = np.eye(len(ANIMALS))

# Silence cosmetic Accelerate/NumPy warnings seen on some macOS builds
# during tiny matrix multiplies. The values remain finite and training
# converges; the warning is not useful for this lesson.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

np.random.seed(2)
w1 = np.random.randn(len(FEATURES), 10) * np.sqrt(2 / len(FEATURES))
b1 = np.zeros((1, 10))
w2 = np.random.randn(10, len(ANIMALS)) * np.sqrt(2 / 10)
b2 = np.zeros((1, len(ANIMALS)))
lr = 0.04

for epoch in range(6000):
    hidden_raw = x @ w1 + b1
    hidden = relu(hidden_raw)
    probs = softmax(hidden @ w2 + b2)

    d_out = (probs - y) / x.shape[0]
    d_hidden = (d_out @ w2.T) * drelu(hidden_raw)

    w2 -= lr * hidden.T @ d_out
    b2 -= lr * d_out.sum(axis=0, keepdims=True)
    w1 -= lr * x.T @ d_hidden
    b1 -= lr * d_hidden.sum(axis=0, keepdims=True)

    if epoch % 1000 == 0:
        loss = -np.sum(y * np.log(probs + 1e-9)) / x.shape[0]
        acc = (probs.argmax(axis=1) == y.argmax(axis=1)).mean()
        print(f"epoch {epoch}: loss {loss:.4f}  accuracy {acc:.3f}")

print()
print("Trained animal guesser")
print(f'  {"animal":>9}  {"prediction":>10}  confidence')
for animal, probs in zip(ANIMALS, probs):
    best = int(probs.argmax())
    print(f'  {animal:>9}  {ANIMALS[best]:>10}  {probs[best]:.3f}')

print()
print("Try a mixed-up animal: large + home + meat + hooves + brown")
weird = np.array([[0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]], dtype=float)
h = relu(weird @ w1 + b1)
p = softmax(h @ w2 + b2)[0]
for idx in p.argsort()[::-1][:4]:
    print(f"  {ANIMALS[idx]:>9}: {p[idx]:.3f}")
