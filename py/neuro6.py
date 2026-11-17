import numpy as np

# Step 6: meal guesser. This is the snack picker idea with a more
# interesting dataset: guess a meal from menu-style clues.
#
# New concept: categorical features become one-hot inputs.
#   course = main / snack / dessert
# becomes three input columns:
#   course_main, course_snack, course_dessert
#
# The traits below are intentionally shared by several meals, so the
# network has to combine clues instead of relying on one giveaway.
# Three pairs differ by exactly ONE trait (near-miss pairs):
#   cottage pie vs baked potato  -> only diet differs
#   spaghetti   vs mac & cheese  -> only diet differs
#   ice cream   vs fruit salad   -> only base differs
# ("spaghetti" here is spaghetti bolognese — that's where the meat is.)
#
# Inputs:
#   course_main, course_snack, course_dessert,
#   temp_hot, temp_cold,
#   taste_sweet, taste_savory,
#   diet_veg, diet_meat,
#   base_pastry, base_potato, base_pasta, base_dairy, base_fruit
#
# Outputs:
#   apple pie, sausage roll, cottage pie, baked potato,
#   spaghetti, mac & cheese, ice cream, fruit salad

MEALS = [
    "apple pie", "sausage roll", "cottage pie", "baked potato",
    "spaghetti", "mac & cheese", "ice cream", "fruit salad",
]
FEATURES = [
    "course_main", "course_snack", "course_dessert",
    "temp_hot", "temp_cold",
    "taste_sweet", "taste_savory",
    "diet_veg", "diet_meat",
    "base_pastry", "base_potato", "base_pasta", "base_dairy", "base_fruit",
]

x = np.array([
    # main snk des  hot cold  swt sav  veg meat  pastry pot pasta dairy fruit
    [0,   0,  1,   1,  0,    1,  0,   1,  0,    1,     0,  0,    0,    0],  # apple pie
    [0,   1,  0,   1,  0,    0,  1,   0,  1,    1,     0,  0,    0,    0],  # sausage roll
    [1,   0,  0,   1,  0,    0,  1,   0,  1,    0,     1,  0,    0,    0],  # cottage pie
    [1,   0,  0,   1,  0,    0,  1,   1,  0,    0,     1,  0,    0,    0],  # baked potato
    [1,   0,  0,   1,  0,    0,  1,   0,  1,    0,     0,  1,    0,    0],  # spaghetti
    [1,   0,  0,   1,  0,    0,  1,   1,  0,    0,     0,  1,    0,    0],  # mac & cheese
    [0,   0,  1,   0,  1,    1,  0,   1,  0,    0,     0,  0,    1,    0],  # ice cream
    [0,   0,  1,   0,  1,    1,  0,   1,  0,    0,     0,  0,    0,    1],  # fruit salad
], dtype=float)

y = np.eye(len(MEALS))

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
w2 = np.random.randn(10, len(MEALS)) * np.sqrt(2 / 10)
b2 = np.zeros((1, len(MEALS)))
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
print("Trained meal guesser")
print(f'  {"meal":>12}  {"prediction":>12}  confidence')
for meal, probs in zip(MEALS, probs):
    best = int(probs.argmax())
    print(f'  {meal:>12}  {MEALS[best]:>12}  {probs[best]:.3f}')

print()
print("Try a mixed-up meal: snack + cold + sweet + meat + pastry")
weird = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=float)
h = relu(weird @ w1 + b1)
p = softmax(h @ w2 + b2)[0]
for idx in p.argsort()[::-1][:4]:
    print(f"  {MEALS[idx]:>12}: {p[idx]:.3f}")
