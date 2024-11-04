import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import random

# Initialize the environment
env = gym.make("CarRacing-v0")

# Hyperparameters
state_size = env.observation_space.shape
action_size = env.action_space.shape[0]
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
learning_rate = 0.001
memory = deque(maxlen=2000)

# Define the Q-network
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=state_size),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

model = build_model()

# Experience replay function
def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Training the agent
for episode in range(100):
    state = env.reset()
    state = np.reshape(state, [1, *state_size])
    for time in range(500):
        env.render()
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state)[0])
        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, *state_size])
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        if done:
            print(f"Episode: {episode+1}/{100}, Score: {time}")
            break
        replay()
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()
