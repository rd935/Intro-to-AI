import numpy as np

# Grid size 10x10
n_large = 10

# Starting positions
bull_pos_large = (0, 0)  # Bull starts at top-left corner
robot_pos_large = (9, 9)  # Robot starts at bottom-right corner
X_large = (5, 5)  # Target X in the center of the grid

# Obstacles around the target
obstacles_large = [(4, 4), (4, 5), (4, 6), (5, 4), (5, 6), (6, 4), (6, 5), (6, 6)]

# Directions for movement (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Initialize T* array to infinity for all states
T_star_large = np.full((n_large, n_large, n_large, n_large), np.inf)

# Set T* for terminal states where the bull is at X
for i in range(n_large):
    for j in range(n_large):
        T_star_large[X_large[0], X_large[1], i, j] = 0  # Bull at the target cell

# Function to check if a position is valid (within bounds and not an obstacle)
def valid_pos_large(x, y):
    return 0 <= x < n_large and 0 <= y < n_large and (x, y) not in obstacles_large

# Update T* values for all states
def update_T_star_large():
    updated = False
    for bull_x in range(n_large):
        for bull_y in range(n_large):
            if (bull_x, bull_y) in obstacles_large:  # Skip obstacles
                continue
            for robot_x in range(n_large):
                for robot_y in range(n_large):
                    if (bull_x, bull_y) == X_large:
                        continue  # Skip terminal state

                    # Calculate the minimum T* for this state
                    min_T_star = np.inf
                    for dx_b, dy_b in directions:  # Bull's moves
                        new_bull_x, new_bull_y = bull_x + dx_b, bull_y + dy_b
                        if not valid_pos_large(new_bull_x, new_bull_y):
                            continue

                        # Maximize over the robot's moves
                        max_T_star = 0
                        for dx_r, dy_r in directions:  # Robot's moves
                            new_robot_x, new_robot_y = robot_x + dx_r, robot_y + dy_r
                            if not valid_pos_large(new_robot_x, new_robot_y):
                                continue

                            # Get T* for the new state
                            max_T_star = max(max_T_star, T_star_large[new_bull_x, new_bull_y, new_robot_x, new_robot_y])

                        # Minimize over the bull's moves
                        min_T_star = min(min_T_star, max_T_star)

                    # Update T* if a lower value is found
                    new_T_star = 1 + min_T_star
                    if new_T_star < T_star_large[bull_x, bull_y, robot_x, robot_y]:
                        T_star_large[bull_x, bull_y, robot_x, robot_y] = new_T_star
                        updated = True
    return updated

# Run the iterative calculation until no more updates occur
def compute_T_star_large():
    iteration = 0
    while update_T_star_large():
        iteration += 1
    return iteration

# Calculate the minimal T* for the grid with obstacles
iterations_large = compute_T_star_large()

# Extract T* for the initial configuration where bull is at (0, 0) and robot at (9, 9)
T_star_pictured = T_star_large[bull_pos_large[0], bull_pos_large[1], robot_pos_large[0], robot_pos_large[1]]

# Output the results
print(f"Number of iterations for convergence: {iterations_large}")
if T_star_pictured < np.inf:
    print(f"Expected minimum moves T* from starting position: {T_star_pictured}")
    print("The robot can corral the bull in finite time.")
else:
    print("Expected minimum moves T* is infinity.")
    print("The robot cannot corral the bull under optimal conditions.")
