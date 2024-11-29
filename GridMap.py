import numpy as np
class GridMap:
    def __init__(self, array):
        self.map = np.array(array)
        self.map[7][7] = 400
        self.currentMap = self.map
        self.currentState = None

    def reset(self):
        self.currentMap = self.map

    def move(self, state, next_state, reward):
        self.currentMap[state[1], state[0]] = -1
        self.currentMap[next_state[1], next_state[0]] = 0
        self.currentState = next_state
        if self.map[next_state[1], next_state[0]] == 1:
            if reward not in [0,-1000]:
                self.map[next_state[1], next_state[0]] = reward

    def merge(self):
        return self.currentMap.flatten()