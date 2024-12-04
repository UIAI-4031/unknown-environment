import numpy as np
import pygame
from environment import UnknownAngryBirds, PygameInit
import matplotlib.pyplot as plt
from Agent import *


def choose_step(env, agent, state,previous_pigs):
    pushState = np.append(np.array(state),np.array(previous_pigs))
    action = agent.select_action(pushState)
    next_state, reward,pigs, done = env.step(action)
    temp = reward
    if state == next_state:
        temp = -10
    if done:
        agent.n_games = agent.n_games + 1
        if agent.n_games % 10 == 0:
            agent.target.load_state_dict(agent.policy.state_dict())
        agent.epsilon = agent.epsilon * agent.epsilon_decay
    pushNextState = np.append(np.array(next_state),np.array(pigs))
    agent.memory.add([pushState, action, temp, pushNextState, done])
    return pigs, next_state, action, reward, done


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = UnknownAngryBirds()
    screen, clock = PygameInit.initialization()
    episode_reward = []
    agent = Agent(10,128,4)
    for episode in range(1000):
        screen, clock = PygameInit.initialization()
        state = env.reset()
        visited_node = []
        running = True
        total_reward = 0
        pigs = [1,1,1,1,1,1,1,1]
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                env.render(screen)
            pigs, next_state, action, reward, done = choose_step(env, agent, state,pigs)
            visited_node = list
            state = next_state
            total_reward += reward
            if len(agent.memory) > 256:
                agent.train()

            if done:
                print(f"Episode {episode} finished with reward: {total_reward}")
                episode_reward.append(total_reward)
                running = False
    agent.test()
    for episode in range(5):
        screen, clock = PygameInit.initialization()
        FPS = 15
        state = env.reset()
        visited_node = []
        pigs = [1,1,1,1,1,1,1,1]
        running = True
        total_reward = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            env.render(screen)

            pigs, next_state, action, reward, done = choose_step(env, agent, state,pigs)
            visited_node = list
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {episode} finished with reward: {total_reward}")
                episode_reward.append(total_reward)
                running = False
            pygame.display.flip()
            clock.tick(FPS)
    print(f'MEAN REWARD: {sum(episode_reward) / len(episode_reward)}')
    plt.scatter(range(len(episode_reward)), episode_reward, c='blue', alpha=0.7, label='Data Points')
    plt.xlabel('Index')
    plt.ylabel('Reward')
    plt.title('Reward Table')
    plt.legend()
    plt.grid()
    plt.show()
    loss_list = np.array(agent.loss_history)
    plt.scatter(range(len(loss_list)), loss_list, c='red', label='loss')
    plt.xlabel('ُStep')
    plt.ylabel('Loss')
    plt.title('Loss Table')
    plt.legend()
    plt.grid()
    plt.show()
    pygame.quit()
