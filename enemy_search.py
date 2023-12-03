import Environment as Env
import Enemy
import random
import numpy as np


class Agent:
    def __init__(self, unique_id, x, y, agent_type, view_sight, gather_sight, environment_len):
        self.unique_id = unique_id
        self.x = x                        # new position
        self.y = y                        # new position
        self.prevx = -1                   # one step previous position
        self.prevy = -1
        self.pattern_index = 0
        self.agent_type = agent_type
        self.view = view_sight
        self.gather = gather_sight
        self.env_len = environment_len
        self.discovered_enemy = list()    # for newly discovered enemies
        self.enemies_seen = list()        # for already analized enemies
        self.bushes = list()
        self.surr_field = None
        self.target = 0      # flag is agent is following an enemy
        self.target_id = None  # id of the enemy follwing
        self.target_dist = -1 
        self.target_x = ""
        self.target_y = ""
        self.move_x = ""
        self.move_y = ""

    def deterministic_search(self): # "deterministic" movement of the agents 
        # eight possible moves are there
        possible_moves = [
            (self.x + 1, self.y) # movement to the right
            (self.x + 1, self.y + 1)
            (self.x + 1, self.y - 1)
            (self.x - 1, self.y) # movement to the left
            (self.x - 1, self.y + 1)
            (self.x - 1, self.y - 1)
            (self.x, self.y + 1) # movement in y-direction only
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

    def random_search(self): # "stochastic" movement of the agents 
        # eight possible moves are there
        possible_moves = [
            (self.x + 1, self.y) # movement to the right
            (self.x + 1, self.y + 1)
            (self.x + 1, self.y - 1)
            (self.x - 1, self.y) # movement to the left
            (self.x - 1, self.y + 1)
            (self.x - 1, self.y - 1)
            (self.x, self.y + 1) # movement in y-direction only
            (self.x, self.y - 1)
        ]

        # Filter out moves that go outside the environment boundaries
        valid_moves = [(x, y) for x, y in possible_moves if 0 <= x < self.env_len and 0 <= y < self.env_len]

        # Filter out moves that correspond to bushes (since bushes are not in enemy-camps)
        valid_moves = [move for move in valid_moves if move not in self.bushes]

        return random.choice(valid_moves) if valid_moves else (self.x, self.y)
    
    def get_enemy_cells(self):
        id = self.target_id
        cell_list = []
        for i in len(self.surr_field):
            for j in len(self.surr_field[0]):
                if self.surr_field[i][j] == 'enemy'and Env.enemy_id(i, j) == id:
                    cell_list.append((i, j))    # append cell locations oof only taht particular target
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

    def approach_direction(self, pos):
        (pi, pj) = pos
        if (self.move_x=="" and self.move_y=="up"):
            if (pj>self.y):
                return True
            else:
                return False
        if (self.move_x=="" and self.move_y=="down"):
            if (pj<self.y):
                return True
            else:
                return False
        if (self.move_x=="right" and self.move_y==""):
            if (pi>self.x):
                return True
            else:
                return False
        if (self.move_x=="left" and self.move_y==""):
            if (pi<self.x):
                return True
            else:
                return False
        if (self.move_x=="right" and self.move_y=="up"):
            if (pi>=self.x and pj>=self.y):
                return True
            else:
                return False
        if (self.move_x=="right" and self.move_y=="down"):
            if (pi>=self.x and pj<=self.y):
                return True
            else:
                return False
        if (self.move_x=="left" and self.move_y=="up"):
            if (pi<=self.x and pj>=self.y):
                return True
            else:
                return False
        if (self.move_x=="left" and self.move_y=="down"):
            if (pi<=self.x and pj<=self.y):
                return True
            else:
                return False
        return False

    def strategic_search(self, enemy_x, enemy_y):
        # Implement a search function when the enemy is detected
        # Will implement the spliting of the searching agent team
        # may need to define a new class
        cell_list = self.get_enemy_cells()
        [dist, eu_dist, ei, ej] = self.get_enemy_distance(cell_list)
        if(dist > 5):
            if(ei >= self.x):
                self.target_x = "right"
                if(ei > self.x+5):
                    self.move_x = "right"
                else:
                    self.move_x = ""
            else:
                self.target_x = "left"
                if(ei < self.x-5):
                    self.move_x = "left"
                else:
                    self.move_x = ""
            if(ej >= self.y):
                self.target_y = "up"
                if(ej > self.y+5):
                    self.move_y = "up"
                else:
                    self.move_y = ""
            else:
                self.target_y = "down"
                if(ej < self.y-5):
                    self.move_y = "down"
                else:
                    self.move_y = ""
            bush_around = [bush for bush in self.bush if self.is_in_limit(bush, self.gather)]
            eff_bushes = [bush for bush in bush_around if self.approach_direction(bush)]
            self.target_dist = -1
            return random.choice(eff_bushes) if eff_bushes else (random.choice(bush_around) if bush_around else (self.x, self.y))
        

            
    
        # ...
        pass

    def find_bushes(self):
        # Can be used to make a list of bushes using the information from self.surr_field
        # assume self.surr_field is 2D grid slice of environment surrounding the agent
        bushes = list()
        for i in len(self.surr_field):
            for j in len(self.surr_field[0]):
                if (self.x == i) and (self.y == j):
                    continue
                if self.surr_field[i][j] == 'bush': # change with bush label in environment
                    bushes.append((i, j))
        self.bushes = bushes

    def find_enemy(self, seen_enemies = []):
        # Can be used to make a list of nearby enemy cell
        # identify enemy ids also distinguish between new enemy and already analyzed enemy
        # return a list of nested tuples or empty list if there are no enemies
        # update enemies_seen list to keep track of them
        # if new enemy is found update discovered_enemy
        for i in len(self.surr_field):
            for j in len(self.surr_field[0]):
                if (self.x == i) and (self.y == j):
                    continue
                if self.surr_field[i][j] == 'enemy': # change with enemy label in environment
                    if Env.enemy_id(i,j) in seen_enemies:
                        self.enemies_seen.append((i,j)) # mark as seen
                    else:    
                        self.discovered_enemy.append((i,j)) # set new discovered enemy
        if(self.discovered_enemy):
            self.target = 1
            (ei, ej) = self.discovered_enemy[0]
            self.target_id = Env.enemy_id(ej,ei)
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

    def scan_surrounding(self):
        # Implement function to get surrounging information from the Environemt Class for bushes and enemies present
        self.surr_field = Env.get_surrounding(self.x, self.y)
        # need to set self.surr_field as a slice of the total environment
        # centered around current cell


class SearchModel:
    def __init__(self, environment_len, num_agents):
        self.len = environment_len
        n = self.len
        self.field = np.zeros((n, n))
        self.agents = []
        seen_enemies = list()   # store ids of already analyzed enemies

        # Assuming you have an Environment instance

        # Create agents
        for i in range(num_agents):
            agent = Agent(i, *self.random_position(), agent_type='search', view_sight=11, gather_sight=5, environment_len=self.len)
            self.agents.append(agent)

    def random_position(self):
        x = random.randrange(self.len)
        y = random.randrange(self.len)
        return x, y

    def step(self):
        for agent in self.agents:
            # If you have an Environment class, uncomment the line below
            # surrounding_info = self.environment.get_surrounding_info(agent.x, agent.y, agent)

            agent.scan_surrounding()

            bushes = agent.bushes # list of bush coordinates in field of vision
            enemies = agent.discovered_enemy # list of enemy coordinates in current field of vision
            seen = len(enemies)


            if seen > 1: # if multiple enemies in the field of vision
                # remove already seen enemies from list to focus on new ones
                for i in range(seen):
                    if enemies[i] in agent.enemies_seen:
                        enemies.remove(enemies[i])
                if (seen - len(enemies)) == 0: # already seen every enemy
                    pass # discuss how to handle
                # if there are non-detected enemies implement strategic search for the first in the list
                # regardless of how many there are
                new_position = agent.strategic_search(enemies[0][0], enemies[0][1])
            
            elif seen == 1: # only one enemy
                if enemies[0] in agent.enemies_seen:
                    # if enemy has already been detected what to do?
                    pass
                # if the enemy was not detected already implement strategic search
                new_position = agent.strategic_search(enemies[0][0], enemies[0][1])
            
            else: # no enemies detected perform random search
                new_position = agent.random_search()
                        
            # if agent.discovered_enemy: 
            #     # need to handle case if there are multiple enemies in the agent's field of vision
            #     if agent.discovered_enemy in agent.enemies_seen:
            #         # still do random search if enemy has already been seen?
            #         # maybe just strategic search again
            #         new_position = agent.random_search()
            #     else:
            #         # Implement avoid_enemy_search when a new enemy is detected
            #         new_position = agent.strategic_search(agent.discovered_enemy[0], agent.discovered_enemy[1])
            # else:
            #     # Implement random_search when no enemy is detected
            #     new_position = agent.random_search()

            agent.x, agent.y = new_position













'''
# Example usage
width = 10
height = 10
num_agents = 8
view_sight = 3

model = SearchModel(width, height, num_agents, view_sight)

for i in range(50):
    model.step()
'''


