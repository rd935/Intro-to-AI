import numpy as np 
import random 

#Build the maze using depth first search 
num_rows    = 5
num_cols    = 5
mazes_num = 50


def dead_end(y, x, visited) :     
	for i in range(-1,1):   #i in -1,0,1
		for j in range(-1,1):#i in -1,0,1
			# as if i == 0 and j == 0 then we are in the same cell 
			if(not(i == 0 and j == 0)):	 
				if(x + i >= 0 and x + i < num_cols):
					if(y + j >= 0 and y + j < num_rows):
						if((x + i, y + j) not in visited):
							#There's an unvisited neighbour
							return False, y + j, x + i  
						
	#There's no unvisited neighbour
	return True, -1, -1;                           
   
def valid_row(y):
	if(y >= 0 and y < num_rows):
		return True; 

	return False; 

def valid_col(x):
	if(x >= 0 and x < num_cols):
		return True;
 
	return False; 
	
def gen_maze():
	# Generate the 50 mazes 
	# initially set all of the cells as unvisited
	maze = np.zeros((mazes_num, num_rows, num_cols))


	for maze_ind in range(0, mazes_num):
		print("Generate Maze : " + str(maze_ind + 1));
		visited = set()  # Set for visitied nodes 
		stack   = []     # Stack is empty at first 

		# start from a random cell	
		# Must choose valid row index
		row_index = random.randint(0, num_rows - 1) 
		# Must choose valid col index 
		col_index = random.randint(0, num_cols - 1)

		# mark it as visited 	
		print("_______________ Start ________________\n")
		print("Loc["+str(row_index)+"],["+str(col_index)+"] = 1")
		# Visited 
		visited.add((row_index , col_index))     
		# Unblocked   
		maze [maze_ind , row_index , col_index] = 1 
		
		# Select a random neighbouring cell to visit that has not yet been visited. 
		print("\n\n_______________ DFS ________________\n")
		# Repeat till visit all cells
		while(len(visited) < num_cols*num_rows):  
		
			crnt_row_index = row_index+random.randint(-1, 1)
			crnt_col_index = col_index+random.randint(-1, 1)
			i = 0; is_dead = False;

			while ((not valid_row(crnt_row_index)) or (not valid_col(crnt_col_index) )or ((crnt_row_index,crnt_col_index) in visited) ):
				crnt_row_index = row_index+random.randint(-1,1)
				crnt_col_index = col_index+random.randint(-1,1)
				i = i+1

				if(i==8):
					#Reach dead end 
					is_dead = True
					break

			if(not is_dead):
				visited.add((crnt_row_index , crnt_col_index)) 
			
			rand_num  = random.uniform(0, 1)

			if( rand_num < 0.3 and not is_dead) : 
				# With 30% probability mark it as blocked. 
				# Leave the block
				maze [maze_ind , crnt_row_index , crnt_col_index] = 0   
				print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 0")				
				# to start get the neighbors of this cell next time 
				row_index = crnt_row_index
				col_index = crnt_col_index

			else : 
				if(not is_dead):
					# With 70% mark it as unblocked and in this case add it to the stack.
					maze [maze_ind , crnt_row_index , crnt_col_index] = 1 #Unblocked 
					print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 1")				
					stack.append((crnt_row_index, crnt_col_index))
					is_dead, unvisit_row, unvisit_col = dead_end(row_index, col_index, visited)

				# if no unvisited neighbour 
				if(is_dead == True): 
					# backtrack to parent nodes on the search tree until it reaches a cell with an unvisited neighbour
					while(len(stack)>0):
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
						row_index = random.randint(0, num_rows - 1)
						col_index = random.randint(0, num_cols - 1)

						if(len(visited)< num_cols*num_rows):
							while ( (not valid_row(row_index)) or (not valid_col(col_index)) or ((row_index,col_index) in visited) ):
								row_index = random.randint(0, num_rows - 1)
								col_index = random.randint(0, num_cols - 1)

						# mark it as visitied 	
						visited.add((row_index , col_index))      
				
				# No dead Node 
				else : 
					visited.add((unvisit_row,unvisit_col))
					row_index = unvisit_row
					col_index = unvisit_col

				
	return maze
		
if __name__ == '__main__':
	mazes = generateMazes() #3D numpy array for the 50 mazes 
	
	for maze_ind in range(0, mazes_num):		 
		np.savetxt('maze '+str(maze_ind)+'.txt',mazes[maze_ind].astype(int) ,fmt='%i', delimiter=",")
