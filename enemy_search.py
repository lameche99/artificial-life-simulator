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
        pass

    def find_bushes(self):
        # Can be used to make a list of nearby enemy cell
        # identify enemy ids also distinguish between new enemy and already analyzed enemy
        # return value will be different for different cases
        pass

    def scan_surrounding(self):
        # Implement function to get surrounging information from the Environemt Class for bushes and enemies present
        pass


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
                # Implement avoid_enemy_search when an enemy is detected
                new_position = agent.strategic_search(agent.discovered_enemy.x, agent.discovered_enemy.y)
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


