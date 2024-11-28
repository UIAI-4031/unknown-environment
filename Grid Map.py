
class GridMap:
    def __init__(self, array):
        self.map = array
        self.currentMap = array

    def get_map(self):
        return self.currentMap

    def reset(self):
        self.currentMap = self.map
        self.currentMap[0][0] = 0

    def move(self, state, next_state):
        self.currentMap[state[0], state[1]] = -1
        self.currentMap[next_state[0], next_state[1]] = 0