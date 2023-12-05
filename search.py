import random
import numpy as np
from agent import Agent
from environment import Environment

class SearchModel:
    def __init__(self, environment_len, num_agents, env: Environment):
        self.len = environment_len
        n = self.len
        self.field = np.zeros((n, n))
        self.env = env
        self.agents = []
        seen_enemies = list()   # store ids of already analyzed enemies

        # Assuming you have an Environment instance

        # Add agents in environment not here
        # Create agents
        # for i in range(num_agents):
        #     agent = Agent(i, *self.random_position(), agent_type='search', view_sight=11, gather_sight=5, environment_len=self.len)
        #     self.agents.append(agent)

    def random_position(self):
        x = random.randrange(self.len)
        y = random.randrange(self.len)
        return x, y

    def step(self):
        for agent in self.env.agents:
            # If you have an Environment class, uncomment the line below
            # surrounding_info = self.environment.get_surrounding_info(agent.x, agent.y, agent)

            agent.scan_surrounding()

            bushes = agent.bushes # list of bush coordinates in field of vision
            enemies = agent.discovered_enemy # list of enemy coordinates in current field of vision
            seen = len(enemies)
            print(f'Starting position: ({agent.x}, {agent.y})')
            if seen > 1: # if multiple enemies in the field of vision
                # remove already seen enemies from list to focus on new ones
                for i in range(seen):
                    if enemies[i] in agent.enemies_seen:
                        enemies.remove(enemies[i])
                if (seen - len(enemies)) == 0: # already seen every enemy
                    print('All enemies already detected. Performing Random Search.')
                    new_position = agent.random_search()
                # if there are non-detected enemies implement strategic search for the first in the list
                # regardless of how many there are
                print(f'New enemy detected at ({enemies[0][1], enemies[0][0]}). Performing Strategic Search')
                new_position = agent.strategic_search(enemies[0][1], enemies[0][0])
            
            elif seen == 1: # only one enemy
                if enemies[0] in agent.enemies_seen:
                    # if enemy has already been detected what to do?
                    print('All enemies already detected. Performing Random Search.')
                    new_position = agent.random_search()
                print(f'New enemy detected at ({enemies[0][1], enemies[0][0]}), discovering.')
                # if the enemy was not detected already implement strategic search
                new_position = agent.strategic_search(enemies[0][1], enemies[0][0])
            
            else: # no enemies detected perform random search
                print('No enemies detected. Performing Random Search.')
                new_position = agent.random_search()
                        
            print(f'New Positon: ({new_position[0]}, {new_position[1]})')
            # agent.x, agent.y = new_position
            self.env.move_agent(agent, new_position[0], new_position[1])