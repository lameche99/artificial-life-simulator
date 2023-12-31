from environment import Environment
import numpy as np

class Agent:
    def __init__(self, unique_id, x, y, view_sight, gather_sight, env_len, type_='agent'):
        self.unique_id = unique_id
        self.x = x                        # new position
        self.y = y                        # new position
        self.prevx = -1                   # one step previous position
        self.prevy = -1
        #self.pattern_index = 0
        # self.type_ = type_
        self.view = view_sight
        self.gather = gather_sight
        self.env = None
        self.env_len = env_len            # length of environment 
        self.new_enemy = list()    # for newly discovered enemies
        self.enemies_seen = list()        # for already analized enemies
        self.bushes = list()
        self.target = 0      # flag is agent is following an enemy
        self.target_id = None  # id of the enemy follwing
        self.target_dist = -1 
        self.target_x = ""
        self.target_y = ""
        self.move_x = ""
        self.move_y = ""
        self.enemy_end_1 = None
        self.enemy_end_2 = None

    def deterministic_search(self): # "deterministic" movement of the agents 
        # eight possible moves are there
        possible_moves = [
            (self.x + 1, self.y), # movement to the right
            (self.x + 1, self.y + 1),
            (self.x + 1, self.y - 1),
            (self.x - 1, self.y), # movement to the left
            (self.x - 1, self.y + 1),
            (self.x - 1, self.y - 1),
            (self.x, self.y + 1), # movement in y-direction only
            (self.x, self.y - 1)
        ]

        # Filter out moves that go outside the environment boundaries
        valid_moves = [
            (x, y) for x, y in possible_moves[self.pattern_index] if 0 <= x < self.env_len and 0 <= y < self.env_len
        ]

        # Filter out moves that correspond to bushes (since bushes are not in enemy camps)
        valid_moves = [move for move in valid_moves if move not in self.bushes]

        # Update the pattern index for the next move
        self.pattern_index = (self.pattern_index + 1) % len(possible_moves)

        # Return the first valid move if any, otherwise stay in the current position
        return valid_moves[0] if valid_moves else (self.x, self.y)

    def random_search(self, seed): # "stochastic" movement of the agents 
        # eight possible moves are there
        np.random.seed(seed=seed)
        possible_moves = [
            (self.x + 1, self.y), # movement to the right
            (self.x + 1, self.y + 1),
            (self.x + 1, self.y - 1),
            (self.x - 1, self.y), # movement to the left
            (self.x - 1, self.y + 1),
            (self.x - 1, self.y - 1),
            (self.x, self.y + 1), # movement in y-direction only
            (self.x, self.y - 1)
        ]

        # Filter out moves that go outside the environment boundaries
        valid_moves = [(x, y) for x, y in possible_moves if 0 <= x < self.env_len and 0 <= y < self.env_len]

        # Filter out moves that correspond to bushes (since bushes are not in enemy-camps)
        valid_moves = [move for move in valid_moves if move not in self.bushes]

        lmoves = len(valid_moves)
        if lmoves == 0:
            return (self.x, self.y)
        
        rng = np.random.randint(0, lmoves)
        return valid_moves[rng]
    
    def get_enemy_cells(self):
        id = self.target_id
        cell_list = []
        topy = max(0, self.y-self.view)
        boty = min(self.env_len, self.y+self.view+1)
        leftx = max(0, self.x-self.view)
        rightx = min(self.env_len, self.x+self.view+1)
        for i in range(topy, boty):
            for j in range(leftx, rightx):
                if (self.env[i][j][0] == 2): # change with enemy label in environment
                    if self.env[i][j][1] == id:
                        cell_list.append((i, j))
        return cell_list
    
    def get_enemy_distance(self, cell_list):
        min_dist = 4*self.view
        min_eu_dist = 4*self.view
        ei, ej = -1, -1
        for loc in cell_list:
            (i, j) = loc
            min_eu_dist = min(min_eu_dist, abs(self.x-i)+abs(self.y-j))
            if min_dist > max(abs(self.x-i), abs(self.y-j) ):
                min_dist =  max(abs(self.x-i), abs(self.y-j) )
                ei, ej = i,j
        return [min_dist, min_eu_dist, ei,ej]
    
    def is_in_limit(self, pos, radius):
        (pi, pj) = pos
        if(pi<self.x-radius or pi>self.x+radius or pj<self.y-radius or pj>self.y+radius):
            return False
        return True

    def approach_direction(self, pos, move_x, move_y):
        (pi, pj) = pos
        # should also add another condition to check if the bush is adjacent to an enemy
        # better to avoid such bush

        if (move_x=="" and move_y=="up"):
            if (pj>self.y):
                return True
            else:
                return False
        if (move_x=="" and move_y=="down"):
            if (pj<self.y):
                return True
            else:
                return False
        if (move_x=="right" and move_y==""):
            if (pi>self.x):
                return True
            else:
                return False
        if (move_x=="left" and move_y==""):
            if (pi<self.x):
                return True
            else:
                return False
        if (move_x=="right" and move_y=="up"):
            if (pi>=self.x and pj>=self.y):
                return True
            else:
                return False
        if (move_x=="right" and move_y=="down"):
            if (pi>=self.x and pj<=self.y):
                return True
            else:
                return False
        if (move_x=="left" and move_y=="up"):
            if (pi<=self.x and pj>=self.y):
                return True
            else:
                return False
        if (move_x=="left" and move_y=="down"):
            if (pi<=self.x and pj<=self.y):
                return True
            else:
                return False
        return False
    
    def set_corner(self, ci, cj, label):
        if(label==1):
            if(self.enemy_end_1 is not None):
                if(abs(self.y-cj)<abs(self.y-self.enemy_end_1[1])):
                    self.enemy_end_1 = (ci, cj)
            else:
                self.enemy_end_1 = (ci, cj)
        else:
            if(self.enemy_end_2 is not None):
                if(abs(self.x-ci)<abs(self.x-self.enemy_end_2[0])):
                    self.enemy_end_2 = (ci, cj)
            else:
                self.enemy_end_2 = (ci, cj)
   
    def check_corner(self):
        for i in range(max(0,self.x-self.gather), min(self.env_len, self.x+self.gather+1)):
            for j in range(max(0,self.y-self.gather), min(self.env_len, self.y+self.gather+1)):
                if self.env[i][j][0] == 2 and self.env[i][j][1] == self.target_id:
                    if(i==0 or i==self.env_len-1):
                        self.set_corner(i,j,label=2)
                    if(j==0 or j==self.env_len-1):
                        self.set_corner(i,j,label=1)
                    if(self.env[i-1][j][0] != 2 and self.env[i+1][j][0] != 2 and self.env[i][j-1][0] != 2):
                        self.set_corner(i,j,label=1)
                    if(self.env[i-1][j][0] != 2 and self.env[i+1][j][0] != 2 and self.env[i][j+1][0] != 2):
                        self.set_corner(i,j,label=1)
                    if(self.env[i-1][j][0] != 2 and self.env[i][j-1][0] != 2 and self.env[i][j+1][0] != 2):
                        self.set_corner(i,j,label=2)
                    if(self.env[i+1][j][0] != 2 and self.env[i][j+1][0] != 2 and self.env[i][j-1][0] != 2):
                        self.set_corner(i,j,label=2)

    def strategic_search(self, seed):
        # Implement a search function when the enemy is detected
        # Will implement the spliting of the searching agent team
        # may need to define a new class
        np.random.seed(seed=seed)
        cell_list = self.get_enemy_cells()
        [dist, eu_dist, ei, ej] = self.get_enemy_distance(cell_list)
        if(dist > self.gather):
            if(ei >= self.x):
                self.target_x = "right"
                if(ei > self.x+self.gather):
                    self.move_x = "right"
                else:
                    self.move_x = ""
            else:
                self.target_x = "left"
                if(ei < self.x-self.gather):
                    self.move_x = "left"
                else:
                    self.move_x = ""
            if(ej >= self.y):
                self.target_y = "up"
                if(ej > self.y+self.gather):
                    self.move_y = "up"
                else:
                    self.move_y = ""
            else:
                self.target_y = "down"
                if(ej < self.y-self.gather):
                    self.move_y = "down"
                else:
                    self.move_y = ""
            bush_around = [bush for bush in self.bushes if self.is_in_limit(bush, self.gather)]
            eff_bushes = [bush for bush in bush_around if self.approach_direction(bush, self.move_x, self.move_y)]
            self.target_dist = -1

            lbush = len(eff_bushes)
            if lbush == 0:
                if len(bush_around) == 0:
                    return (self.x, self.y)
                else:
                    laround = len(bush_around)
                    rng = np.random.randint(0, laround)
                    return bush_around[rng]
            rng = np.random.randint(0, lbush)
            return eff_bushes[rng]
        
        else:
            self.check_corner()    
            if(self.enemy_end_1 is not None and self.enemy_end_2 is not None):
                return (self.x, self.y)
            if(self.enemy_end_1 is not None):
                self.move_x = self.invert(self.target_x)
                self.move_y = self.target_y
                if(self.move_x == "left"):
                    x_pseudo = self.x + eu_dist - self.gather
                else:
                    x_pseudo = self.x - eu_dist + self.gather
                reg = self.get_region(self.x, self.y, self.move_x, self.move_y, label = 1) # to be implemented
                bush_around = [bush for bush in self.bushes if self.is_in_limit(bush, self.gather)]
                eff_bushes = [bush for bush in bush_around if self.approach_direction(bush, self.target_x, self.target_y)]
                opt_bushes = [bush for bush in bush_around if self.opt_region(bush, x_pseudo, self.y, self.move_x, self.move_y, label = 1)]
                lbush = len(opt_bushes)
                if lbush == 0:
                    if len(eff_bushes) == 0:
                        return (self.x, self.y)
                    else:
                        laround = len(eff_bushes)
                        rng = np.random.randint(0, laround)
                        return eff_bushes[rng]
                rng = np.random.randint(0, lbush)
                return opt_bushes[rng]
    
            else:
                self.move_x = self.target_x
                self.move_y = self.invert(self.target_y)
             
                if(self.move_x == "left"):
                    x_pseudo = self.x + eu_dist - self.gather
                else:
                    x_pseudo = self.x - eu_dist + self.gather
                reg = self.get_region(self.x, self.y, self.move_x, self.move_y, label = 2)
                bush_around = [bush for bush in self.bushes if self.is_in_limit(bush, self.gather)]
                eff_bushes = [bush for bush in bush_around if self.approach_direction(bush, self.move_x, self.move_y)]
                opt_bushes = [bush for bush in bush_around if (bush in reg)]
                lbush = len(opt_bushes)
                if lbush == 0:
                    if len(eff_bushes) == 0:
                        return (self.x, self.y)
                    else:
                        laround = len(eff_bushes)
                        rng = np.random.randint(0, laround)
                        return eff_bushes[rng]
                rng = np.random.randint(0, lbush)
                return opt_bushes[rng]

    def get_region(self, x, y, move_x, move_y, label):
        if (label==2):
            if(move_x=="left"):
                x_new = x - self.gather
            else:
                x_new = x + self.gather
            if(move_y=="up"):
                y_new = y + self.gather
            else:
                y_new = y - self.gather
            return self.get_region(x_new, y_new, self.invert(move_x), self.invert(move_y), 1)
        else:
            pass
    
    def opt_region(self, pos, x, y, move_x, move_y, label):
        if (label==2):
            if(move_x=="left"):
                x_new = x - self.gather
            else:
                x_new = x + self.gather
            if(move_y=="up"):
                y_new = y + self.gather
            else:
                y_new = y - self.gather
            return self.opt_region(pos, x_new, y_new, self.invert(move_x), self.invert(move_y), 1)
        else:
            (pi, pj) = pos
            if(move_y == "up" and (pj<y or pj>y+self.gather)):
                return False
            if(move_y == "down" and (pj>y or pj<y-self.gather)):
                return False
            if(move_x == "left" and (pi < x - abs(y-pj))):
                return False
            if(move_x == "right" and (pi > x + abs(y-pj))):
                return False
            return True

    def invert(self, direction):
        if direction == "up":
            return "down"
        if direction == "down":
            return "up"
        if direction == "left":
            return "right"
        if direction == "right":
            return "left"

    def find_bushes(self):
        # Can be used to make a list of bushes using the information from self.surr_field
        # assume self.surr_field is 2D grid slice of environment surrounding the agent
        bushes = list()
        topy = max(0, self.y-self.view)
        boty = min(self.env_len, self.y+self.view+1)
        leftx = max(0, self.x-self.view)
        rightx = min(self.env_len, self.x+self.view+1)
        for i in range(topy, boty):
            for j in range(leftx, rightx):
                if self.env[i][j][0] == 1: # change with bush label in environment
                    bushes.append((j, i))
        self.bushes = bushes

    def find_enemy(self, seen_enemies = []):
        # Can be used to make a list of nearby enemy cell
        # identify enemy ids also distinguish between new enemy and already analyzed enemy
        # return a list of nested tuples or empty list if there are no enemies
        # update enemies_seen list to keep track of them
        # if new enemy is found update new_enemy
        topy = max(0, self.y-self.view)
        boty = min(self.env_len, self.y+self.view+1)
        leftx = max(0, self.x-self.view)
        rightx = min(self.env_len, self.x+self.view+1)
        for i in range(topy, boty):
            for j in range(leftx, rightx):
                if (self.env[i][j][0] == 2): # change with enemy label in environment
                    if self.env[i][j][1] in seen_enemies:
                        self.enemies_seen.append((i, j)) # mark as seen
                    else:    
                        self.new_enemy.append((i, j)) # set new discovered enemy
        
        if(self.new_enemy):
            self.target = 1
            (ei, ej) = self.new_enemy[0]
            self.target_id = self.env[ei][ej][1]
            if(ei >= self.x):
                self.target_x = "right"
                self.move_x = "right"
            else:
                self.target_x = "left"
                self.move_x = "left"
            if(ej >= self.y):
                self.target_y = "up"
                self.move_y = "up"
            else:
                self.target_y = "down"
                self.move_y = "down"

    def scan_surrounding(self, env):
        # Implement function to get surrounging information from the Environemt Class for bushes and enemies present
        self.env = env
        # need to set self.surr_field as a slice of the total environment
        # centered around current cell
   
    def check_camp(self):
        if (self.enemy_end_1 is None) or (self.enemy_end_2 is None):
            return [-1, -1]
      
        if abs(self.enemy_end_1[0] - self.enemy_end_2[0]) == abs(self.enemy_end_1[1]-self.enemy_end_2[1]):
            #return the center of the rhombus
            if (self.enemy_end_1[0]-self.prevx)**2 +(self.enemy_end_2[1]-self.prevy)**2 >  (self.enemy_end_2[0]-self.prevx)**2 +(self.enemy_end_1[1]-self.prevy)**2:
                center_pos = (self.enemy_end_1[0],self.enemy_end_2[1])# or (self.enemy_end_2[0],self.enemy_end_1[1])
            else:
                center_pos = (self.enemy_end_2[0],self.enemy_end_1[1])
            enemy_size = abs(self.enemy_end_1[0] - center_pos[0]) + abs(self.enemy_end_1[1] - center_pos[1])
            return [center_pos, enemy_size]
        else: 
            if (self.enemy_end_1[0]-self.x)**2 +(self.enemy_end_1[1]-self.y)**2 < (self.enemy_end_2[0]-self.x)**2 +(self.enemy_end_2[1]-self.y)**2:
                self.enemy_end_2 = None
            else:
                self.enemy_end_1 = None
            return [-1, -1]  
    
    def print_pos(self):
      return "Agent {} is at ({}, {})".format(self.unique_id, self.x, self.y)

    def XY(self):
      return self.unique_id, self.x, self.y
