import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from collections import deque
import random

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 0.001
batch_size = 64
memory_size = 100000


# تعریف مدل
def make_model(input_shape, hidden_size, output_size):
    model = Sequential([
        Dense(hidden_size, activation='relu'),
        #Dense(hidden_size, activation='relu', input_shape=input_shape),
        Dense(hidden_size, activation='relu'),
        Dense(output_size, activation='linear')  # خروجی برای 4 اکشن
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.model = make_model(input_shape=(2,), hidden_size=8, output_size=4)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    def epsilon_greedy_policy(self, state):
        state = np.expand_dims(np.array(state), axis=0)
        if np.random.rand() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            Q_values = self.model.predict(state, verbose=0)
            return np.argmax(Q_values[0])

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_experiences(self, batch_size):
        indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def training_step(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)
        next_Q_values = self.model.predict(next_states, verbose=0)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = rewards + (1 - dones) * self.gamma * max_next_Q_values
        actions_one_hot = tf.one_hot(actions, depth=4)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states, training=True)
            Q_values = tf.reduce_sum(all_Q_values * actions_one_hot, axis=1)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # کاهش مقدار epsilon برای اکتشاف-استفاده (exploration-exploitation)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay