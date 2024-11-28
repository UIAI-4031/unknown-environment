import numpy as np
import pygame
from environment import UnknownAngryBirds, PygameInit
from Learn_map import Leaner
from Agent import *


def choose_step(env, agent, state):
    action = agent.epsilon_greedy_policy(state)
    next_state, reward, done = env.step(action)
    temp = reward

    if next_state == state:
        temp = -10
    agent.memory.append((state, action, temp, next_state, done))
    return next_state, action, reward, done


if __name__ == "__main__":
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))

    base = Leaner()
    map = base.get_map()
    env = base.get_env()
    screen, clock = PygameInit.initialization()
    FPS = 1000
    print(map)
    episode_reward = []
    agent = Agent()

    for episode in range(100):
        state = env.reset()
        running = True
        total_reward = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            env.render(screen)

            next_state, action, reward, done = choose_step(env, agent, state)
            state = next_state
            total_reward += reward
            if len(agent.memory) > 64:
                agent.training_step(64)

            if done:
                print(f"Episode {episode} finished with reward: {total_reward}")
                episode_reward.append(total_reward)
                running = False

            pygame.display.flip()
            clock.tick(100000)

    print(f'MEAN REWARD: {sum(episode_reward) / len(episode_reward)}')

    pygame.quit()
