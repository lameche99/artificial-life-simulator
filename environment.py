import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Enemy:
    def __init__(self, uid, xinit, yinit, type_='enemy') -> None:
        self.uid = uid
        self.x = xinit
        self.y = yinit
        self.type_ = type_
        self.env = None

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.grid = np.tile(None, (width, height))
        self.agents = list()
        self.enemies = list()

    def add_agent(self, agent):
        x, y = agent.x, agent.y
        if self.grid[y][x] is None:
            self.grid[y][x] = agent
            agent.env = self
            if agent.type_ == 'agent':
                self.agents.append(agent)
            else:
                self.enemies.append(agent)
            return True
        else:
            return False

    def move_agent(self, agent, new_x, new_y):
        if (0 <= new_x < self.width) and (0 <= new_y < self.height) and (self.grid[new_y][new_x] is None):
            self.grid[agent.y][agent.x] = None
            agent.prevx, agent.prevy = agent.x, agent.y
            agent.x, agent.y = new_x, new_y
            self.grid[new_y][new_x] = agent
            return True
        else:
            return False
        
    def enemy_id(self, xpos, ypos):
        enemy = self.grid[xpos][ypos]
        return enemy.uid
    
    def get_surrounding(self, x, y, sight):
        topy = max(0, y-sight)
        boty = min(self.height, y+sight+1)
        leftx = max(0, x-sight)
        rightx = min(self.width, x+sight+1)
        return self.grid[topy:(boty), leftx:(rightx)]

    def print_environment(self):
        # Print top border
        print("+" + "-" * (2 * self.width - 1) + "+")

        for y in range(self.height-1, -1, -1):
            print(f"{y}|", end=" ")  # Add row index
            for x in range(self.width):
                if self.grid[y][x] is None:
                    print(" ", end=" ")
                else:
                    print("X", end=" ")  # You can customize this to represent different agent types
            print("|")

        # Print bottom border
        print("+" + "-" * (2 * self.width - 1) + "+")

        # Print column indices
        print(" " * 3 + "".join([f"{i} " for i in range(self.width)]))

    def plot_environment(self):
        fig, ax = plt.subplots()

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.bushes:
                    circle = patches.Circle((x + 0.5, y + 0.5), 0.4, color='green', fill=True)
                    ax.add_patch(circle)
                elif self.grid[y][x] is not None:
                    ax.text(x + 0.5, y + 0.5, 'X', ha='center', va='center', color='black', fontsize=12)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', 'box')
        plt.show()
        