import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 256
memory_size = 100000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.FC:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

    def forward(self, x):
        return self.FC(x)


class ReplayBuffer:
    def __init__(self, length):
        self.memory = deque(maxlen=length)

    def add(self, member):
        self.memory.append((
            torch.tensor(member[0], dtype=torch.float32),
            torch.tensor(member[1], dtype=torch.long),
            torch.tensor(member[2], dtype=torch.long),
            torch.tensor(member[3], dtype=torch.float32),
            torch.tensor(member[4], dtype=torch.bool)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.stack(actions).to(device),
            torch.stack(rewards).to(device),
            torch.stack(next_states).to(device),
            torch.stack(dones).to(device)
        )

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.n_games = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.memory = ReplayBuffer(memory_size)
        self.policy = QNetwork(input_size, hidden_size, output_size).to(device)
        self.target = QNetwork(input_size, hidden_size, output_size).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.loss_history = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                return torch.argmax(self.policy(state)).item()

    def train(self):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
        self.epsilon = max(self.epsilon_min, self.epsilon )

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def test(self):
        self.epsilon = 0.0
        self.epsilon_min = 0
