import random

import numpy as np
import pygame
from environment import UnknownAngryBirds, PygameInit
import matplotlib.pyplot as plt

map = np.ones((8, 8), dtype=int)
map[0][0] = -1


class Leaner:
    def __init__(self):
        self.env = UnknownAngryBirds()
        self.map = np.ones((8, 8), dtype=int)
        self.map[0][0] = -1

        screen, clock = PygameInit.initialization()
        FPS = 100000
        for i in range(10):
            state = self.env.reset()
            running = True
            total_reward = 0
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

                self.env.render(screen)
                action = random.choice([0, 1, 2, 3])
                next_state, reward, done = self.env.step(action)
                self.upgrade(state, next_state, action, reward)
                if done:
                    print(f"Episode {i} finished with reward: {total_reward}")
                    running = False
                pygame.display.flip()
                state = next_state
                clock.tick(FPS)

    def get_map(self):
        return self.map

    def get_env(self):
        return self.env

    def upgrade(self, state, next_state, action, reward):
        if self.map[next_state[0]][next_state[1]] in [1, 0]:
            self.map[next_state[0]][next_state[1]] = reward
        if next_state == state:
            if reward != -2000:
                if 0 < state[0] < 7 and 0 < state[1] < 7:
                    if (action == 0 and self.map[state[0], state[1] + 1] not in [1, -10] and self.map[state[0], state[1] - 1] not in [1, -10]
                            and self.map[state[0] - 1, state[1]] == 1):
                        self.map[state[0] - 1, state[1]] = -10
                    elif (action == 1 and self.map[state[0], state[1] + 1] not in [1, -10] and self.map[state[0], state[1] - 1] not in [1, -10]
                            and self.map[state[0] + 1, state[1]] == 1):
                        self.map[state[0] + 1, state[1]] = -10
                    elif action == 2 and self.map[state[0] - 1, state[1]] not in [1, -10] and self.map[state[0] + 1, state[1]] not in [1, -10] and self.map[state[0], state[1] - 1] == 1:
                        self.map[state[0], state[1] - 1] = -10
                    elif (action == 3 and self.map[state[0] - 1, state[1]] not in [1, -10] and self.map[state[0] + 1, state[1]] not in [1, -10]
                            and self.map[state[0], state[1] + 1] == 1):
                        self.map[state[0], state[1] + 1] = -10
