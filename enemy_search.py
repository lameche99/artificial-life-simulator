import Environment as Env
import Enemy
import random
import numpy as np


class Agent:
    def __init__(self, unique_id, x, y, agent_type, view_sight, gather_sight, environment_len):
        self.unique_id = unique_id
        self.x = x
        self.y = y
        self.prevx = -1
        self.prevy = -1
        self.agent_type = agent_type
        self.view = view_sight
        self.gather = gather_sight
        self.env_len = environment_len
        self.discovered_enemy = None
        self.enemies_seen = list()
        self.bushes = list()
        self.surr_field = None

    def random_search(self):
        possible_moves = [
            (self.x + 1, self.y),
            (self.x - 1, self.y),
            (self.x, self.y + 1),
            (self.x, self.y - 1)
        ]

        # Filter out moves that go outside the environment boundaries
        valid_moves = [(x, y) for x, y in possible_moves if 0 <= x < self.env_len and 0 <= y < self.env_len]

        return random.choice(valid_moves) if valid_moves else (self.x, self.y)

    def strategic_search(self, enemy_x, enemy_y):
        # Implement a search function when the enemy is detected
        # Will implement the spliting of the searching agent team
        # may need to define a new class
        # ...
        pass

    def find_bushes(self):
        # Can be used to make a list of bushes using the information from self.surr_field
        # assume self.surr_field is 2D grid slice of environment surrounding the agent
        bushes = list()
        for i in len(self.surr_field[0]):
            for j in len(self.surr_field[0][0]):
                if (self.x == i) and (self.y == j):
                    continue
                if self.surr_field[i][j] == 'bush': # change with bush label in environment
                    bushes.append((i, j))
        self.bushes = bushes

    def find_enemy(self):
        # Can be used to make a list of nearby enemy cell
        # identify enemy ids also distinguish between new enemy and already analyzed enemy
        # return a list of nested tuples or empty list if there are no enemies
        # update enemies_seen list to keep track of them
        # if new enemy is found update discovered_enemy
        for i in len(self.surr_field[0]):
            for j in len(self.surr_field[0][0]):
                if (self.x == i) and (self.y == j):
                    continue
                if self.surr_field[i][j] == 'enemy': # change with enemy label in environment
                    if (i,j) not in self.enemies_seen:
                        self.discovered_enemy = (i,j) # set new discovered enemy for strategic search
                        self.enemies_seen.append((i,j)) # mark as seen
                    else:
                        self.discovered_enemy = (i,j) # set discovered enemy already seen

    def scan_surrounding(self):
        # Implement function to get surrounging information from the Environemt Class for bushes and enemies present
        self.find_bushes()
        self.find_enemy()


class SearchModel:
    def __init__(self, environment_len, num_agents):
        self.len = environment_len
        n = self.len
        self.field = np.zeros((n, n))
        self.agents = []

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

            if agent.discovered_enemy: 
                # need to handle case if there are multiple enemies in the agent's field of vision
                if agent.discovered_enemy in agent.enemies_seen:
                    # still do random search if enemy has already been seen?
                    # maybe just strategic search again
                    new_position = agent.random_search()
                else:
                    # Implement avoid_enemy_search when a new enemy is detected
                    new_position = agent.strategic_search(agent.discovered_enemy[0], agent.discovered_enemy[1])
            else:
                # Implement random_search when no enemy is detected
                new_position = agent.random_search()

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


