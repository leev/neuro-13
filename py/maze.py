"""
Reinforcement learning: a creature learns a maze by trial and error.

Nobody ever tells the creature which move is right. All it gets is a reward
after each move: +10 for reaching the goal (G), -10 for stepping in lava (^),
and 0 for everything else. It starts at S. Walls are #.

From those rewards alone it fills in a table with one number for every move
in every cell: its guess at how many points that move leads to in the end.
The rule that fills in the table is called Q-learning:

    surprise  = reward + GAMMA * (best value in the cell it landed in) - old value
    new value = old value + ALPHA * surprise

There is no neural network in this script. The table is the whole brain.

Things to try (change one, run again):
    EPSILON = 0        never experiment
    EPSILON = 1        only experiment
    GAMMA = 0          only the very next reward counts
    STEP_REWARD = -1   every ordinary move costs a point (then try GAMMA = 1)

Run:
    python3 maze.py
"""
import random

MAZE = [
    "....^..",
    ".##.##.",
    ".......",
    ".#####.",
    ".#G....",
    ".^#^##.",
    "S......",
]
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.9
EPISODES = 500
MAX_MOVES = 200
GOAL_REWARD, LAVA_REWARD, STEP_REWARD = 10, -10, 0

MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
SIZE = len(MAZE)

# Delete this line and every run tells a different story.
random.seed(7)

cells = [(row, col) for row in range(SIZE) for col in range(SIZE) if MAZE[row][col] != "#"]
START = next(cell for cell in cells if MAZE[cell[0]][cell[1]] == "S")
GOAL = next(cell for cell in cells if MAZE[cell[0]][cell[1]] == "G")
Q = {cell: {move: 0.0 for move in MOVES} for cell in cells}


def name(cell):
    return "ABCDEFG"[cell[1]] + str(cell[0] + 1)


def land(cell, move):
    row, col = cell[0] + MOVES[move][0], cell[1] + MOVES[move][1]
    return (row, col) if (row, col) in Q else cell


def reward_for(old_cell, new_cell):
    bumped_a_wall = new_cell == old_cell
    kind = MAZE[new_cell[0]][new_cell[1]]
    if kind == "G":
        return GOAL_REWARD
    if kind == "^" and not bumped_a_wall:
        return LAVA_REWARD
    return STEP_REWARD


def best_value(cell):
    return max(Q[cell].values())


def best_move(cell):
    top = best_value(cell)
    return random.choice([move for move in MOVES if Q[cell][move] == top])


def run_episode():
    cell, moves, points = START, 0, 0
    while moves < MAX_MOVES and cell != GOAL:
        if random.random() < EPSILON:
            move = random.choice(list(MOVES))
        else:
            move = best_move(cell)
        new_cell = land(cell, move)
        reward = reward_for(cell, new_cell)
        future = 0 if new_cell == GOAL else GAMMA * best_value(new_cell)
        surprise = reward + future - Q[cell][move]
        Q[cell][move] += ALPHA * surprise
        cell, moves, points = new_cell, moves + 1, points + reward
    return moves, points, cell == GOAL


def plan():
    """The route it would walk right now with no experiments at all."""
    route = [START]
    while route[-1] != GOAL:
        cell = route[-1]
        next_cell = land(cell, max(Q[cell], key=Q[cell].get))
        if list(Q[cell].values()).count(best_value(cell)) > 1 or next_cell in route:
            break
        route.append(next_cell)
    return route


print("the maze:")
for row in MAZE:
    print("   " + " ".join(row))
print(f"\nexploring {EPSILON}, learning rate {ALPHA}, future counts {GAMMA}\n")

block_moves, block_wins = [], 0
for episode in range(1, EPISODES + 1):
    moves, points, reached = run_episode()
    block_moves.append(moves)
    block_wins += reached
    if episode <= 5:
        print(f"episode {episode}: {moves:3d} moves, {points:+4d} points, {'reached the goal' if reached else 'ran out of time'}")
    if episode % 50 == 0:
        average = sum(block_moves) / len(block_moves)
        print(f"episodes {episode - 49:3d} to {episode:3d}: reached the goal {block_wins:2d} times out of 50, {average:5.1f} moves on average")
        block_moves, block_wins = [], 0

print("\nwhat it learned (its best move in each cell; a dot means it has no idea):")
print("     " + "  ".join("ABCDEFG"))
for row in range(SIZE):
    line = []
    for col in range(SIZE):
        cell, kind = (row, col), MAZE[row][col]
        if kind in "#^G":
            line.append(kind)
        elif list(Q[cell].values()).count(best_value(cell)) > 1:
            line.append("·")
        else:
            line.append(ARROWS[max(Q[cell], key=Q[cell].get)])
    print(f"  {row + 1}  " + "  ".join(line))

route = plan()
if route[-1] == GOAL:
    print(f"\nits route: {' '.join(name(cell) for cell in route)}")
    print(f"that is {len(route) - 1} moves")
    print("\nthe value it gives each cell along the way:")
    print("   " + "  ".join(f"{best_value(cell):.2f}".replace("-0.00", "0.00") for cell in route[:-1]))
    if STEP_REWARD == 0:
        print(f"each one is {GAMMA} times the next: a reward one move further off counts a bit less")
    else:
        print(f"each one is {STEP_REWARD} + {GAMMA} times the next")
else:
    print(f"\nit has no complete route yet: its plan stops at {name(route[-1])}")

print(f"\nthe four numbers it keeps for the start cell {name(START)}:")
print("   " + ", ".join(f"{move} {value:.2f}" for move, value in Q[START].items()))
