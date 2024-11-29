import numpy as np
import pygame
from environment import UnknownAngryBirds, PygameInit
from Learn_map import Leaner
from Agent import *
from GridMap import GridMap as  gp

def choose_step(visited,env, agent, state,gridMap):
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    temp = reward
    if reward != -1:
        visited = [next_state]
        temp = reward
    agent.memory.add([state, action, temp, next_state, done])
    return visited,next_state, action, reward, done


if __name__ == "__main__":
    import requests


    base = Leaner()
    base_map = base.get_map()
    env = base.get_env()
    screen, clock = PygameInit.initialization()
    FPS = 100000
    grid_map = gp(base_map)
    episode_reward = []
    agent = Agent()
    for episode in range(5000):
        state = env.reset()
        visited_node = []
        running = True
        total_reward = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()


            list, next_state, action, reward, done = choose_step(visited_node,env, agent, state,grid_map)
            visited_node = list
            state = next_state
            total_reward += reward
            if len(agent.memory) > 64:
                agent.train()

            if done:
                print(f"Episode {episode} finished with reward: {total_reward}")
                grid_map.reset()
                running = False
    agent.test()
    for episode in range(5):
        screen, clock = PygameInit.initialization()
        FPS = 8
        state = env.reset()
        visited_node = []
        running = True
        total_reward = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            env.render(screen)

            list, next_state, action, reward, done = choose_step(visited_node,env, agent, state,grid_map)
            visited_node = list
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {episode} finished with reward: {total_reward}")
                grid_map.reset()
                episode_reward.append(total_reward)
                running = False
            pygame.display.flip()
            clock.tick(FPS)
    print(f'MEAN REWARD: {sum(episode_reward) / len(episode_reward)}')

    pygame.quit()
