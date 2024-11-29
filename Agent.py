import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.01
batch_size = 64
memory_size = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fs0 = nn.Linear(input_size,hidden_size)
        self.fs1 = nn.Linear(hidden_size,hidden_size)
        self.fs2 = nn.Linear(hidden_size,output_size)
        self.loss_fn = torch.nn.MSELoss()
        torch.nn.init.kaiming_uniform_(self.fs0.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fs1.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.fs2.weight)

    def forward(self, x):
        x = torch.relu(self.fs0(x))
        x = torch.relu(self.fs1(x))
        return self.fs2(x)


class ReplayBuffer:
    def __init__(self, length):
        self.memory = deque(maxlen=length)

    def add(self, member):
        state, action, reward, next_state, done = member
        self.memory.append((
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.long),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (

            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones)
        )

    def __len__(self):
        return self.memory.__len__()


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.memory = ReplayBuffer(memory_size)
        self.policy = QNetwork(input_size=2, hidden_size=32, output_size=4)
        self.target = QNetwork(input_size=2, hidden_size=32, output_size=4)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def learn(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                return torch.argmax(self.policy(state)).item()

    def train(self):
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def test(self):
        self.epsilon_min = 0
        self.epsilon = 0
