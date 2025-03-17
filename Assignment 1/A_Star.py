import numpy as np
import random
from matplotlib.colors import ListedColormap
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import matplotlib 
import BinaryHeap as bh
import time 
import sys
from mazeGen import gen_maze

# Part 1 
def dead_end(y, x, visited) :     
	for i in range(-1, 1):
		for j in range(-1, 1):
			# as if i == 0 and j == 0 then we are in the same cell

			if(not(i == 0 and j == 0)): 	 
				if(x + i >= 0 and x + i < num_cols ):
					if(y + j >= 0 and y + j < num_rows):
						if((x + i, y + j) not in visited):
							# There's an unvisited neighbour
							return False, y + j, x + i  
						
	# There's no unvisited neighbour
	return True, -1, -1;                           
   
def valid_row(y):
	if(y >= 0 and y < num_rows):
		return True; 

	return False; 

def valid_col(x):
	if(x >= 0 and x < num_cols):
		return True; 

	return False; 
	
def gen_maze(mazes_num, num_rows, num_cols):

	# Generate the 50 mazes 
	# initially set all of the cells as unvisited
	maze = np.zeros((mazes_num, num_rows, num_cols))


	for maze_ind in range(0, mazes_num) :
		print("Generate Maze : " + str(maze_ind + 1));
		# Set for visitied nodes 
		visited = set()  
		# Stack is empty at first
		stack   = []      

		# start from a random cell	
		# Must choose valid row index
		row_index = random.randint(0, num_rows - 1) 
		# Must choose valid col index 
		col_index = random.randint(0, num_cols - 1)
		# mark it as visitied 	
		print("_______________ Start ________________\n")
		# print("Loc["+ str(row_index)+"] , ["+ str(col_index)+"] = 1")
		# Visited 
		visited.add((row_index, col_index)) 
		# Unblocked      
		maze[maze_ind, row_index, col_index] = 1 
		

		# Select a random neighbouring cell to visit that has not yet been visited. 
		print("\n\n_______________ DFS ________________\n")

		# Repeat till visit all cells 
		while(len(visited) < num_cols*num_rows): 
			crnt_row_index = row_index+random.randint(-1,1)#neighbor
			crnt_col_index = col_index+random.randint(-1,1)#neighbor

			i=0; is_dead = False;

			while ((not valid_row(crnt_row_index)) or (not valid_col(crnt_col_index)) or ((crnt_row_index,crnt_col_index) in visited)):
				crnt_row_index = row_index+random.randint(-1, 1)
				crnt_col_index = col_index+random.randint(-1, 1)
				i = i + 1

				if(i==8):
					#Reach dead end 
					is_dead = True
					break

			if(not is_dead):
				visited.add((crnt_row_index, crnt_col_index)) 
			
			rand_num  = random.uniform(0, 1)

			if( rand_num < 0.3 and not is_dead) : 
				# With 30% probability mark it as blocked. 
				# Leave the block
				maze [maze_ind, crnt_row_index, crnt_col_index] = 0   
				#print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 0")				
				# to start get the neighbors of this cell next time 
				row_index = crnt_row_index
				col_index = crnt_col_index

			else : 
				if(not is_dead):
					# With 70% mark it as unblocked and in this case add it to the stack.
					# Unblocked
					maze [maze_ind, crnt_row_index, crnt_col_index] = 1  
					#print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 1")				
					stack.append((crnt_row_index, crnt_col_index))
					is_dead, unvisit_row, unvisit_col = dead_end(row_index, col_index, visited)

				# if no unvisited neighbour
				if(is_dead == True): 
					# backtrack to parent nodes on the search tree until it reaches a cell with an unvisited neighbour
					while(len(stack) > 0):
						parent_row, parent_col = stack.pop();
						is_dead, unvisit_row, unvisit_col = dead_end(parent_row, parent_col, visited)

						if(is_dead == False):
							break;
					
					# Now wither we reach not dead end or stack is empty 
					if(len(stack)>0):
						visited.add((unvisit_row, unvisit_col))
						row_index = unvisit_row
						col_index = unvisit_col

					else :
						# Repeat the whole process from a point not vistited
						row_index = random.randint(0,num_rows - 1)
						col_index = random.randint(0,num_cols - 1)

						if(len(visited) < num_cols*num_rows):
							while ((not valid_row(row_index)) or (not valid_col(col_index)) or ((row_index, col_index) in visited)):
								row_index = random.randint(0, num_rows-1)
								col_index = random.randint(0, num_cols-1)

						# mark it as visited 	
						visited.add((row_index, col_index))  

				#No dead Node     					
				else :  
					visited.add((unvisit_row, unvisit_col))
					row_index = unvisit_row
					col_index = unvisit_col

	return maze
	
def plot_line(A, col):
    if not A:  # Check if A is empty
        print("Warning: Empty list passed to plot_line.")
        return None  # or handle as needed

    x, y = zip(*A)  # Unpack the list into x and y coordinates
    temp = plt.plot(y, x, color=col)[0]  # Get the first element of the list, which is a Line2D object
    print("Line type: ", type(temp).__name__)  # This should print 'Line2D'
    return temp  # Return the Line2D object directly
	

# This method is to get Manhattan Distances for A*
def manahattan_dist(curr, goal):
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])

# This method will find the path for Forward A*
def forward_path(f_dict, curr, goal):
    if goal not in f_dict:
        print(f"Warning: Goal {goal} not in f_dict. Cannot trace back the path.")
        return []

    temp = [goal]
    next = f_dict[goal]

    while next != curr:
        temp.insert(0, next)
        next = f_dict.get(next)  # Use .get() to avoid KeyError

    temp.insert(0, next)

    return temp

# This method will find the path for the Backward A*
def backward_path(dict, curr, goal):
    temp = [curr]
    next = dict[curr]

    while next != goal:
        temp.append(next)
        next = dict[next]

    temp.append(next)

    return temp

# This method will find the neighbors of the current cell
def nextNeighbor (cell_num, length):
    neighbor_list = [(cell_num[0], cell_num[1] + 1), (cell_num[0], cell_num[1] - 1), (cell_num[0] + 1, cell_num[1]), (cell_num[0] - 1, cell_num[1])]
    temp = []

    for x in neighbor_list:
        if -1 < x[0] < length and -1 < x[0] < length:
            temp.append(x)

    return temp

# This method will update the neighbors to empty and blocked
def update(map, length, curr, b_list):
    goodCells = nextNeighbor(curr, length)

    for x in goodCells:
        # Ensure x is within bounds of the 101x101 grid
        if 0 <= x[0] < length and 0 <= x[1] < length:
            print(x)
            print(map[2][x])  # This should be safe now
            
            if x in b_list:
                # It's blocked
                map[2][x] = 2 
            else:
                # It's empty
                map[2][x] = 1
        else:
            print(f"Warning: Neighbor {x} is out of bounds.")


# This method will display the intial map
def disp_map(length, b_list = None, p_list = None, map = None, old = None):
    if not(map is None):
        d = map[2]
        f, a = plt.subplots()
        print(d)
        a.imshow(d)

    else:
        d = np.zeros((length, length))
        if not (b_list is None):
            x = b_list
        f, a = plt.subplots()
        a.imshow(d)

    print(a)
    a.grid(which = 'minor', color='black', linestyle = '-', linewidth = 1)
    
    if p_list != None:
        plot_line(p_list, 'red')

    if old != None:
        plot_line(old, 'green')
		 
    plt.show()


def show_launch(length, b_list=None, p_list=None, map=None, old=None):
    # This method will add the colors to the maze
    d = map[2]
    f, a = plt.subplots()
    
    # Set up the color map
    color_map = matplotlib.colors.ListedColormap(['grey', 'white', 'black'])
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    n = matplotlib.colors.BoundaryNorm(boundaries, color_map.N)
    
    # Display the maze
    image = a.imshow(d, interpolation='nearest', origin='lower')
    a.set_xticks(np.arange(-0.5, length, 1), minor=True)
    a.set_yticks(np.arange(-0.5, length, 1), minor=True)
    a.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # Handle the old path line
    o_line = plot_line(old, 'green') if old else None  # Only plot if old is not empty
    
    # Handle the new path line
    p_line = plot_line(p_list, 'red') if p_list else None  # Only plot if p_list is not empty
    
    plt.pause(1)

    return image, f, o_line, p_line


def show_update(image, f, o_line, p_line, map, a_list, b_list):
	# This method will add the path line to the maze 
    image.set_data(map)
    x, y = zip(*a_list)
    o_line.set_xdata(y)
    o_line.set_ydata(x)
    x,y = zip(*b_list)
    p_line.set_xdata(y)
    p_line.set_ydata(x)
    plt.pause(1)
    f.canvas.draw()

def adapted_astar(m_len, b_list, start, goal, show=False):
    max_g = m_len * m_len
    curr_cell = start 
    g_cell = goal
    mag_map = np.zeros((5, m_len, m_len))
    mag_map[3] = np.zeros((m_len, m_len), dtype=bool)
    mag_map[4] = np.zeros((m_len, m_len), dtype=bool)
    mag_map[2][curr_cell] = 1
    max_G = 0 
    t_steps = 0 
    t_expense = 0 
    curr_track = [start]
    
    # Check if the goal is blocked
    if goal in b_list:
        print(f"Goal {goal} is blocked. Cannot find path.")
        return None, None

    update(mag_map, m_len, curr_cell, b_list)

    while curr_cell != g_cell:
        t_steps += 1 
        mag_map[1][curr_cell] = 0
        mag_map[0][curr_cell] = t_steps
        mag_map[1][g_cell] = 0
        mag_map[0][g_cell] = t_steps
        o_list = []
        o_dict = dict()
        f_dict = dict()
        mag_map[4] = np.zeros((m_len, m_len), dtype=bool)
        bh.insert(o_list, o_dict, manahattan_dist(curr_cell, g_cell), curr_cell)
        
        while o_list and mag_map[1][g_cell] > o_list[0]:
            color_cell = bh.pop(o_list, o_dict)
            mag_map[4][color_cell] = True
            t_expense += 1
            
            for n_cell in nextNeighbor(color_cell, m_len):
                print(f"Considering neighbor: {n_cell}")  # Debug print

                if 0 <= n_cell[0] < m_len and 0 <= n_cell[1] < m_len:  # Boundary check
                    if mag_map[2][n_cell] != 2:
                        if mag_map[3][n_cell]:
                            new_heuris = max_G - mag_map[1][n_cell]
                        else:
                            new_heuris = manahattan_dist(n_cell, g_cell)

                        if mag_map[0][n_cell] < t_steps:
                            mag_map[1][n_cell] = np.inf
                            mag_map[0][n_cell] = t_steps

                        if mag_map[1][n_cell] > mag_map[1][color_cell] + 1:
                            f_dict[n_cell] = color_cell
                            print(f"Added to f_dict: {n_cell} -> {color_cell}")  # Debug print
                            new_g = mag_map[1][color_cell] + 1
                            bh.insert(o_list, o_dict, (max_g * (new_g + new_heuris) - new_g), n_cell)
                            mag_map[1][n_cell] = new_g

        mag_map[3] = mag_map[4]
        max_G = mag_map[1][g_cell]

        if not o_list:
            print("No valid path found. Exiting.")
            return None, None
        
        # Check if goal is reachable
        if g_cell not in f_dict:
            print(f"Goal {g_cell} was not reached. Pathfinding failed.")
            return None, None  # Handle gracefully
        
        curr_path = forward_path(f_dict, curr_cell, g_cell)

        if show:
            if curr_cell == start:
                plt.ion()
                im, fig, o_line, p_line = show_launch(m_len, map=mag_map, p_list=curr_path, old=curr_track)
                plt.show()
            else:
                show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

        for cell in curr_path:
            if cell == curr_cell:
                continue
            else:
                if mag_map[2][cell] != 2:
                    curr_track.append(cell)
                    curr_cell = cell
                    update(mag_map, m_len, curr_cell, b_list)
                else:
                    break 

    if show:
        show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

    return curr_track, t_expense

def backward_astar(m_len, b_list, start, goal, show=False):
    # Here we implement Backward A* which will take the length, start, and goal as inputs
    t_expense = 0 
    max_G = m_len * m_len
    curr_cell = start 
    g_cell = goal
    mag_map = np.zeros((4, m_len, m_len))
    mag_map[2][curr_cell] = 1
    t_steps = 0 
    curr_track = [start]
    update(mag_map, m_len, curr_cell, b_list)

    while curr_cell != g_cell:
        mag_map[3] = np.zeros((m_len, m_len))
        t_steps += 1 
        mag_map[1][curr_cell] = np.inf
        mag_map[0][curr_cell] = t_steps
        mag_map[1][g_cell] = 0
        mag_map[0][g_cell] = t_steps
        o_list = []
        o_dict = dict()
        f_dict = dict()
        bh.insert(o_list, o_dict, manahattan_dist(curr_cell, g_cell), g_cell)

        while o_list and mag_map[1][curr_cell] > o_list[0]:
            color_cell = bh.pop(o_list, o_dict)
            mag_map[3][color_cell] = 1
            t_expense += 1

            for n_cell in nextNeighbor(color_cell, m_len):
                # Ensure n_cell is within bounds
                if 0 <= n_cell[0] < m_len and 0 <= n_cell[1] < m_len:
                    if mag_map[2][n_cell] != 2:
                        if mag_map[0][n_cell] < t_steps:
                            mag_map[1][n_cell] = np.inf
                            mag_map[0][n_cell] = t_steps

                        if mag_map[1][n_cell] > mag_map[1][color_cell] + 1:
                            mag_map[1][n_cell] = mag_map[1][color_cell] + 1
                            f_dict[n_cell] = color_cell
                            bh.insert(o_list, o_dict, max_G * (mag_map[1][n_cell] + manahattan_dist(n_cell, curr_cell)) - mag_map[1][n_cell], n_cell)

        if not o_list:
            return None, None
        
        if show:
            curr_path = backward_path(f_dict, curr_cell, g_cell)

            if curr_cell == start:
                plt.ion()
                im, fig, o_line, p_line = show_launch(m_len, map=mag_map, p_list=curr_path, old=curr_track)
                plt.show()
            else:
                show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

        while curr_cell != g_cell:
            cell = f_dict[curr_cell]

            if mag_map[2][cell] != 2:
                curr_track.append(cell)
                curr_cell = cell
                update(mag_map, m_len, curr_cell, b_list)
            else:
                break

    if show:
        show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path) 

    return curr_track, t_expense

def forward_astar(m_len, b_list, start, goal, show=False):
    # Here we implement Forward A*
    t_expense = 0
    max_G = m_len * m_len
    curr_cell = start 
    g_cell = goal
    mag_map = np.zeros((4, m_len, m_len))
    mag_map[2][curr_cell] = 1
    t_steps = 0 
    curr_track = [start]
    update(mag_map, m_len, curr_cell, b_list)

    while curr_cell != g_cell:
        t_steps += 1 
        mag_map[1][curr_cell] = 0
        mag_map[0][curr_cell] = t_steps
        mag_map[1][g_cell] = np.inf
        mag_map[0][g_cell] = t_steps
        o_list = []
        o_dict = dict()
        f_dict = dict()
        bh.insert(o_list, o_dict, manahattan_dist(curr_cell, g_cell), curr_cell)

        while o_list and mag_map[1][g_cell] > o_list[0]:
            color_cell = bh.pop(o_list, o_dict)
            mag_map[3][color_cell] = 1
            t_expense += 1

            for n_cell in nextNeighbor(color_cell, m_len):
                # Ensure n_cell is within bounds
                if 0 <= n_cell[0] < m_len and 0 <= n_cell[1] < m_len:
                    if mag_map[2][n_cell] != 2:
                        if mag_map[0][n_cell] < t_steps:
                            mag_map[1][n_cell] = np.inf
                            mag_map[0][n_cell] = t_steps

                        if mag_map[1][n_cell] > mag_map[1][color_cell] + 1:
                            mag_map[1][n_cell] = mag_map[1][color_cell] + 1
                            f_dict[n_cell] = color_cell
                            bh.insert(o_list, o_dict, (max_G * (mag_map[1][n_cell] + manahattan_dist(n_cell, g_cell)) - mag_map[1][n_cell]), n_cell)

        if not o_list:
            return None, None

        curr_path = forward_path(f_dict, curr_cell, g_cell)

        if show:
            if curr_cell == start:
                plt.ion()
                im, fig, o_line, p_line = show_launch(m_len, map=mag_map, p_list=curr_path, old=curr_track)
                plt.show()
            else:
                show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

        for cell in curr_path:
            if cell == curr_cell:
                continue
            else:
                if mag_map[2][cell] != 2:
                    curr_track.append(cell)
                    curr_cell = cell
                    update(mag_map, m_len, curr_cell, b_list)

                else:
                    break 

    if show:
        show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

    return curr_track, t_expense


def repeated_forward(m_len, b_list, start, goal, show=False):
    # Here we implement Repeated Forward A*
    t_expense = 0
    max_G = m_len * m_len
    curr_cell = start 
    g_cell = goal
    mag_map = np.zeros((4, m_len, m_len))
    mag_map[2][curr_cell] = 1
    t_steps = 0 
    curr_track = [start]
    
    # Check if goal is blocked
    if goal in b_list:
        print(f"Goal {goal} is blocked. Cannot find path.")
        return None, None
    
    update(mag_map, m_len, curr_cell, b_list)

    while curr_cell != g_cell:
        t_steps += 1 
        mag_map[1][curr_cell] = 0
        mag_map[0][curr_cell] = t_steps
        mag_map[1][g_cell] = np.inf
        mag_map[0][g_cell] = t_steps
        o_list = []
        o_dict = dict()
        f_dict = dict()
        bh.insert(o_list, o_dict, manahattan_dist(curr_cell, g_cell), curr_cell)

        while o_list and mag_map[1][g_cell] > o_list[0]:
            color_cell = bh.pop(o_list, o_dict)
            t_expense += 1

            for n_cell in nextNeighbor(color_cell, m_len):
                if mag_map[2][n_cell] != 2:
                    if mag_map[0][n_cell] < t_steps:
                        mag_map[1][n_cell] = np.inf
                        mag_map[0][n_cell] = t_steps

                    if mag_map[1][n_cell] > mag_map[1][color_cell] + 1:
                        mag_map[1][n_cell] = mag_map[1][color_cell] + 1
                        f_dict[n_cell] = color_cell
                        bh.insert(o_list, o_dict, (max_G * (mag_map[1][n_cell] + manahattan_dist(n_cell, g_cell)) + mag_map[1][n_cell]), n_cell) 

        if not o_list:
            print("No valid path found. Exiting.")
            return None, None

        # Debug print for f_dict
        print("Current f_dict:", f_dict)

        if g_cell not in f_dict:
            print(f"Goal {g_cell} was not reached. Pathfinding failed.")
            return None, None  # Handle gracefully
        
        curr_path = forward_path(f_dict, curr_cell, g_cell)

        # Show updating logic
        if show:
            if curr_cell == start:
                plt.ion()
                im, fig, o_line, p_line = show_launch(m_len, map=mag_map, p_list=curr_path, old=curr_track)
                plt.show()
            else:
                show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

        # Updating current cell tracking
        for cell in curr_path:
            if cell == curr_cell:
                continue
            else:
                if mag_map[2][cell] != 2:
                    curr_track.append(cell)
                    curr_cell = cell
                    update(mag_map, m_len, curr_cell, b_list)
                else:
                    break 

    if show:
        show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

    return curr_track, t_expense


def repeated_forward2(m_len, b_list, start, goal, show=False):
    # Implement Repeated Forward A* with start, goal, and maze size m_len
    t_expense = 0
    curr_cell = start
    g_cell = goal
    mag_map = np.zeros((3, m_len, m_len))  # 3 layers of the maze
    mag_map[2][curr_cell] = 1  # Mark the start as visited
    t_steps = 0
    curr_track = [start]
    update(mag_map, m_len, curr_cell, b_list)

    while curr_cell != g_cell:
        t_steps += 1
        mag_map[1][curr_cell] = 0
        mag_map[0][curr_cell] = t_steps
        mag_map[1][g_cell] = np.inf
        mag_map[0][g_cell] = t_steps
        o_list = []
        o_dict = dict()
        f_dict = dict()
        bh.insert(o_list, o_dict, manahattan_dist(curr_cell, g_cell), curr_cell)

        while len(o_list) > 0 and mag_map[1][g_cell] > o_list[0]:
            color_cell = bh.pop(o_list, o_dict)
            t_expense += 1

            for n_cell in nextNeighbor(color_cell, m_len):
                # Ensure both row and column indices are within bounds
                if 0 <= n_cell[0] < m_len and 0 <= n_cell[1] < m_len:
                    if mag_map[2][n_cell] != 2:
                        if mag_map[0][n_cell] < t_steps:
                            mag_map[1][n_cell] = np.inf
                            mag_map[0][n_cell] = t_steps

                        if mag_map[1][n_cell] > mag_map[1][color_cell] + 1:
                            mag_map[1][n_cell] = mag_map[1][color_cell] + 1
                            f_dict[n_cell] = color_cell
                            bh.insert(o_list, o_dict, (mag_map[1][n_cell] + manahattan_dist(n_cell, g_cell)), n_cell)

        if not o_list:
            return None, None

        curr_path = forward_path(f_dict, curr_cell, g_cell)

        if show:
            if curr_cell == start:
                plt.ion()
                im, fig, o_line, p_line = show_launch(m_len, map=mag_map, p_list=curr_path, old=curr_track)
                plt.show()
            else:
                show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

        for cell in curr_path:
            if cell == curr_cell:
                continue
            else:
                if mag_map[2][cell] != 2:
                    curr_track.append(cell)
                    curr_cell = cell
                    update(mag_map, m_len, curr_cell, b_list)
                else:
                    break

    if show:
        show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

    return curr_track, t_expense


def repeated_backward(m_len, b_list, start, goal, show = False):
	# Here we implement Repeated Backward A* which will take the len, start, and goal as inputs
	t_expense = 0
	curr_cell = start 
	g_cell = goal
	mag_map = np.zeros((3, m_len, m_len))
	mag_map[2][curr_cell] = 1
	t_steps = 0 
	curr_track = [start]
	update(mag_map, m_len, curr_cell, b_list)

	while curr_cell != g_cell:
		t_steps += 1 
		mag_map[1][curr_cell] = np.inf
		print("mag_map")
		print(mag_map [1,1,2])
		print("curr_cell")
		print(curr_cell)
		
		mag_map[0][curr_cell] = t_steps
		mag_map[1][g_cell] = 0
		mag_map[0][g_cell] = t_steps
		o_list = []
		o_dict = dict()
		f_dict = dict()
		bh.insert(o_list, o_dict, manahattan_dist(curr_cell,g_cell), g_cell)
		print("o_list")
		print(o_list)

		while o_list :
			print(mag_map[1][curr_cell])
			
			if(mag_map[1][curr_cell] <= o_list[0]):
				break;
			
			t_expense += 1
			color_cell = bh.pop(o_list, o_dict)

			for n_cell in nextNeighbor(color_cell, m_len):
				# Ensure both row and column indices are within bounds
				if 0 <= n_cell[0] < m_len and 0 <= n_cell[1] < m_len:
					if mag_map[2][n_cell] != 2:
						if mag_map[0][n_cell] < t_steps:
							mag_map[1][n_cell] = np.inf
							mag_map[0][n_cell] = t_steps

						if mag_map[1][n_cell] > mag_map[1][color_cell] + 1:
							mag_map[1][n_cell] = mag_map[1][color_cell] + 1
							f_dict[n_cell] = color_cell
							bh.insert(o_list, o_dict, mag_map[1][n_cell] + manahattan_dist(n_cell, curr_cell), n_cell)
				else:
					print(f"Warning: Neighbor {n_cell} is out of bounds.")

		if not o_list:
			return None, None
		
		if show:
			curr_path = backward_path(f_dict, curr_cell, g_cell)

			if curr_cell == start:
				plt.ion()
				im, fig, o_line, p_line = show_launch(m_len, map = mag_map, p_list = curr_path, old = curr_track)
				plt.show()

			else:
				show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

		while curr_cell != g_cell:
			cell = f_dict[curr_cell]

			if mag_map[2][cell] != 2 :
				curr_track.append(cell)
				curr_cell = cell
				update(mag_map, m_len, curr_cell, b_list)

			else:
				break 

	if show:
		show_update(im, fig, o_line, p_line, mag_map[2], curr_track, curr_path)

	return curr_track, t_expense




if __name__ == '__main__':

	# Generate mazes 101*101
	num_rows = 101
	num_cols = 101
	mazes_num = 50
	mazes = gen_maze(mazes_num,num_rows,num_cols)
	#3D numpy array for the 50 mazes 
	
	# Generate mazes 5*5
	num_rows = 5
	num_cols = 5
	mazes_num = 3
	#mazes2 = gen_maze(mazes_num,num_rows,num_cols)

	m_len = 101;
     
	b_list = [(0, 1),   # Row 0, Column 1
    (1, 1),   # Row 1, Column 1
    (1, 3),   # Row 1, Column 3
    (50, 50), # A block in the middle
    (50, 51),
    (50, 52),
    (25, 25), # A block somewhere else
    (25, 26),
    (25, 27),
    (99, 98), # Block on the last row
    (99, 99),];

	p_list = [(0, 0),    # Start point
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 1),
    (9, 1),
    (10, 1),
    (11, 1),
    (12, 1),
    (12, 2),
    (12, 3),
    (12, 4),
    (12, 5),
    (12, 6),
    (13, 6),
    (14, 6),
    (15, 6),
    (16, 6),
    (17, 6),
    (18, 6),
    (19, 6),
    (20, 6),
    (21, 6),
    (22, 6),
    (23, 6),
    (24, 6),
    (25, 6),
    (26, 6),
    (27, 6),
    (28, 6),
    (29, 6),
    (30, 6),
    (31, 6),
    (32, 6),
    (33, 6),
    (34, 6),
    (35, 6),
    (36, 6),
    (37, 6),
    (38, 6),
    (39, 6),
    (40, 6),
    (41, 6),
    (42, 6),
    (43, 6),
    (44, 6),
    (45, 6),
    (46, 6),
    (47, 6),
    (48, 6),
    (49, 6),
    (50, 6),
    (51, 6),
    (52, 6),
    (53, 6),
    (54, 6),
    (55, 6),
    (56, 6),
    (57, 6),
    (58, 6),
    (59, 6),
    (60, 6),
    (61, 6),
    (62, 6),
    (63, 6),
    (64, 6),
    (65, 6),
    (66, 6),
    (67, 6),
    (68, 6),
    (69, 6),
    (70, 6),
    (71, 6),
    (72, 6),
    (73, 6),
    (74, 6),
    (75, 6),
    (76, 6),
    (77, 6),
    (78, 6),
    (79, 6),
    (80, 6),
    (81, 6),
    (82, 6),
    (83, 6),
    (84, 6),
    (85, 6),
    (86, 6),
    (87, 6),
    (88, 6),
    (89, 6),
    (90, 6),
    (91, 6),
    (92, 6),
    (93, 6),
    (94, 6),
    (95, 6),
    (96, 6),
    (97, 6),
    (98, 6),
    (99, 6),
    (100, 6)];


	
	#length = 5;
	#b_list = [(2,3),(3,4),(3,3),(4,3),(4,4),(5,4)]; # Example given in assignment
	#p_list = [(3,1),(3,4)];
	map   = mazes;
	old   = None;

	disp_map(list, b_list, p_list, map, old);
	
	startX = int(input("Please , Enter valid x coordinate for the start point: "));
	startY = int(input("Please , Enter valid y coordinate for the start point: "));
	
	goalX  = int(input("Please , Enter valid x coordinate for the goal point: ")); 
	goalY  = int(input("Please , Enter valid y coordinate for the goal point: ")); 
	
	start = (startX, startY);
	goal  = (goalX, goalY);

	# Repeated Forward A*
	print("Repeated Forward A*")
	repeated_forward2(m_len, b_list, start, goal, True);
	print("____________________________________")

	# Repeated Backward A*
	print("Repeated Backward A*")
	repeated_backward(m_len, b_list, start, goal, True);
	print("____________________________________")

	# Repeated forward A with smaller g*
	print("Repeated Forward A* with smaller g-values")
	repeated_forward(m_len, b_list, start, goal, True);
	print("____________________________________")

	# Adaptive A*
	print("Adapted A*")
	adapted_astar(m_len, b_list, start, goal, True);
		
	# forward A large g* 
	print("Repeated Forward A* with large g-values")
	forward_astar(m_len, b_list, start, goal, True);
	print("____________________________________")

	# backward A large g* 
	print("Repeated Backward A* with large g-values")		
	backward_astar(m_len, b_list, start, goal, True);
	print("____________________________________")
	
	