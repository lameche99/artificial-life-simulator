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