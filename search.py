

import numpy as np
from agent import Agent
from environment import Environment


class SearchModel:
    def __init__(self, environment_len, agents, env: np.array):
        self.len = environment_len
        self.field = np.zeros((self.len, self.len))
        self.env = env
        self.agents = agents
        self.seen_enemies = list()   # store ids of already analyzed enemies

        # Assuming you have an Environment instance

        # Add agents in environment not here
        # Create agents
        # for i in range(num_agents):
        #     agent = Agent(i, *self.random_position(), agent_type='search', view_sight=11, gather_sight=5, environment_len=self.len)
        #     self.agents.append(agent)

    def random_position(self):
        x = np.random.randrange(self.len)
        y = np.random.randrange(self.len)
        return x, y

    def step(agents, grid, seen_enemies):
        new_agents_positions = [] 
        for agent in agents:
            # If you have an Environment class, uncomment the line below
            # surrounding_info = self.environment.get_surrounding_info(agent.x, agent.y, agent)

            agent.scan_surrounding(grid)

            bushes = agent.find_bushes()# list of bush coordinates in field of vision
            agent.find_enemy(seen_enemies=seen_enemies)
            enemies = agent.new_enemy # list of enemy coordinates in current field of vision

            if len(enemies):
                if(agent.target != 1):
                    enemy_pos = [enemies[0][0], enemies[0][1]]
                    print('New enemy detected at ({},{}). Performing Strategic Search'.format(enemy_pos[0],enemy_pos[1]))
                    agent.target = 1
                    agent.target_id = grid[enemies[0][0]][enemies[0][0]][1]  # to be changes according to environment code
                elif(agent.target_id not in seen_enemies):
                    new_position = agent.strategic_search()
                    [enemy_center_pos, enemy_size] = agent.check_camp()
                    if enemy_center_pos==-1:
                        new_position = agent.strategic_search()
                    else:
                        seen_enemies.append(enemy_center_pos)  
                        print("Enemy discovered with centre at ({}, {}) and of the size of {}".format(enemy_center_pos[0], enemy_center_pos[1], enemy_size))

            else:
                new_position = agent.random_search()
            
            agent.prevx, agent.prevy = agent.x, agent.y 
            agent.x, agent.y = new_position[0], new_position[1]
            new_agents_positions.append([agent.x, agent.y])
            #  self.env.move_agent(agent, new_position[0], new_position[1])
        return new_agents_positions, seen_enemies


"""
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
                else:
                  enemy_pos = {enemies[0][1], enemies[0][0]}; 
                  print(f'New enemy detected at ({enemies[0][1], enemies[0][0]}). Performing Strategic Search')
                  new_position = agent.strategic_search(enemies[0][1], enemies[0][0])   # also implement function to check if both the enemey corner are set
                  enemy_center_pos = agent.check_camp()
                  if enemy_center_pos==None:
                    #TO DO agent.strategic_search()
                    pass
                  else:
                    self.seen_enemies.append(enemy_center_pos)  
                # if both enemy cornnrs are st and in are in a digonal, print enemy location # after computing its center
                # else delete the old corner location and again search for the new location since the enemy has modes 
            # elif seen == 1: # only one enemy
            #     if enemies[0] in agent.enemies_seen:
            #         # if enemy has already been detected what to do?
            #         print('All enemies already detected. Performing Random Search.')
            #         new_position = agent.random_search()
            #     print(f'New enemy detected at ({enemies[0][1], enemies[0][0]}), discovering.')
            #     # if the enemy was not detected already implement strategic search
            #     new_position = agent.strategic_search(enemies[0][1], enemies[0][0])
            else: # no enemies detected perform random search
                print('No enemies detected. Performing Random Search.')
                new_position = agent.random_search()
                        
            print(f'New Positon: ({new_position[0]}, {new_position[1]})')
            # agent.x, agent.y = new_position
            self.env.move_agent(agent, new_position[0], new_position[1])


            # fu
"""