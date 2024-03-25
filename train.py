import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env.custom_environment import PreyPredatorEnv
import pygame

class DQN(nn.Module):
    def __init__(self, state_size = 24, action_size = 5):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Adjust these layers as needed
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)
        self.state_size = state_size

    def forward(self, x):
        x = x.reshape((1, self.state_size))
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size = 24, action_size = 5, mode = 'train'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99  # discount rate
        if mode == 'train':
            self.epsilon = 1.0  # exploration rate
        else:
            self.epsilon = 0.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, file_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, file_name)

    @classmethod
    def load(cls, file_name, mode = 'train'):
        agent = cls(mode = mode)
        checkpoint = torch.load(file_name)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        return agent

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model.forward(state)
        return np.argmax(action_values.detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model.forward(next_state).detach()).item())
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model.forward(state)
            target_f.flatten()[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model.forward(state), target_f)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        self.loss = total_loss / len(minibatch)
    
def train_dqn(env):
    # Assume the state_size and action_size are the same for both types of agents for simplicity
    state_size = 24  # Update this based on your actual observation space
    action_size = env.action_space.n

    # Instantiate two separate agents for prey and predator
    prey_agent = DQNAgent(state_size, action_size)
    predator_agent = DQNAgent(state_size, action_size)

    episodes = 1000
    batch_size = 32
    ep_avg = 0
    for e in range(episodes):
        env.reset()
        # Example state initialization; adjust according to your environment's observation space
        prey_state = env.default_observation
        predator_state = env.default_observation

        for time in range(10000):  # Assuming a max number of steps per episode
            if env.render_mode == 'human':
                event = pygame.event.get()
            # The following part needs significant adjustments:
            # - Determine if the current agent is a prey or a predator
            # - Fetch the appropriate state for the agent
            # - Decide on actions for both prey and predator
            # - Step through the environment and update states and memories accordingly

            # Simplified action selection for both agents (you need logic to distinguish between agent types)
            prey_action = prey_agent.act(prey_state)
            predator_action = predator_agent.act(predator_state)

            # Your environment needs to handle multi-agent steps (this is highly simplified)
            next_prey_state, prey_reward, done = env.step(prey_action)  # Adjust this method
            next_predator_state, predator_reward, done = env.step(predator_action)  # Adjust

            # Example: Push to memory and update state; repeat for predator
            prey_agent.memory.push(prey_state, prey_action, prey_reward, next_prey_state, done)
            prey_state = next_prey_state

            # Example: Perform replay if enough memories are collected; repeat for predator
            if len(prey_agent.memory) > batch_size:
                prey_agent.replay(batch_size)
            
            predator_agent.memory.push(predator_state, predator_action, predator_reward, next_predator_state, done)
            predator_state = next_predator_state

            # Example: Perform replay if enough memories are collected; repeat for predator
            if len(predator_agent.memory) > batch_size:
                predator_agent.replay(batch_size)

            if env.stored_num_predators + env.stored_num_prey == 0:
                break
            env.render()
            
        ep_avg += time
        if (e + 1) % 10 == 0:
            predator_agent.save('predator.pth')
            prey_agent.save('prey.pth')
            # Example logging
            print(f"Episode: {e + 1}/{episodes}, Score: {ep_avg / 10}, Prey Epsilon: {prey_agent.epsilon}, Predator Epsilon: {predator_agent.epsilon}")
            ep_avg = 0
        # Update epsilon for exploration; repeat for predator
        prey_agent.update_epsilon()
        predator_agent.update_epsilon()


if __name__ == '__main__':
    env = PreyPredatorEnv(num_prey=10, num_predators=4, grid_size=40, max_steps_per_episode=100000, padding = 10, food_probability=0.01, render_mode="non")
    train_dqn(env)