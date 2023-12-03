import numpy as np

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]

    def add_agent(self, agent):
        x, y = agent.x, agent.y
        if self.grid[y][x] is None:
            self.grid[y][x] = agent
            agent.environment = self
            return True
        else:
            return False

    def move_agent(self, agent, new_x, new_y):
        if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_y][new_x] is None:
            self.grid[agent.y][agent.x] = None
            agent.prevx, agent.prevy = agent.x, agent.y
            agent.x, agent.y = new_x, new_y
            self.grid[new_y][new_x] = agent
            return True
        else:
            return False

    def print_environment(self):
        # Print top border
        print("+" + "-" * (2 * self.width - 1) + "+")

        for y in range(self.height):
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
        