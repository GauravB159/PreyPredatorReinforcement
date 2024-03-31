import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from env.custom_environment import PreyPredatorEnv
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
    def __init__(self, state_size=24, action_size=5, history_length=5):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size * history_length, 64)  # Adjust input size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)
    
    def forward(self, x):
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
    def __init__(self, state_size = 24, action_size = 5, mode = 'train', epsilon = 0.995, history_length = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(100000)
        self.gamma = 0.99  # discount rate
        if mode == 'train':
            self.epsilon = 1.0  # exploration rate
        else:
            self.epsilon = 0.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon
        self.model = DQN(state_size, action_size, history_length=history_length)
        self.model = self.model.to(device)
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
    def load(cls, file_name, mode = 'train', history_length = 5):
        agent = cls(mode = mode, history_length = history_length)
        checkpoint = torch.load(file_name, map_location=device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        return agent

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        action_values = self.model.forward(state)
        return np.argmax(action_values.cpu().detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        # Extract information from the minibatch
        states = torch.FloatTensor(np.array([s[0] for s in minibatch])).to(device)
        actions = torch.LongTensor(np.array([s[1] for s in minibatch])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([s[2] for s in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([s[3] for s in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([s[4] for s in minibatch])).to(device)

        # Compute Q values for current states
        current_q_values = self.model(states).gather(1, actions)

        # Compute Q values for next states
        next_q_values = self.model(next_states).detach().max(1)[0]
        # Compute the target of the current Q values
        target_q_values = (rewards + (self.gamma * next_q_values * (1 - dones))).unsqueeze(1)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.update_epsilon()
    
def train_dqn(env, episodes = 1000, epsilon = 0.995, avg_length = 10):
    # Assume the state_size and action_size are the same for both types of agents for simplicity
    state_size = 24  # Update this based on your actual observation space
    action_size = env.action_space.n

    # Instantiate two separate agents for prey and predator
    prey_agent = DQNAgent(state_size, action_size, epsilon = epsilon, history_length=env.observation_history_length)
    predator_agent = DQNAgent(state_size, action_size, epsilon = epsilon, history_length=env.observation_history_length)
    f = open("train.log", "a+")
    batch_size = 32
    ep_avg = 0
    for e in range(episodes):
        env.reset()
        # Example state initialization; adjust according to your environment's observation space
        prey_state = env.stacked_flattened_default_observation

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

            # Your environment needs to handle multi-agent steps (this is highly simplified)
            next_prey_state, prey_reward, done = env.step(prey_action)  # Adjust this method

            # Example: Push to memory and update state; repeat for predator
            prey_agent.memory.push(prey_state, prey_action, prey_reward, next_prey_state, done)
            prey_state = next_prey_state

            # Example: Perform replay if enough memories are collected; repeat for predator
            if len(prey_agent.memory) > batch_size:
                prey_agent.replay(batch_size)
                
            if env.stored_num_prey == 0:
                break
            env.render()
            
        ep_avg += time
        print(f"{e + 1}/{episodes} done!")
        if (e + 1) % avg_length == 0:
            predator_agent.save('predator.pth')
            prey_agent.save('prey.pth')
            # Example logging
            f.write(f"Episode: {e + 1}/{episodes}, Score: {ep_avg / avg_length}, Prey Epsilon: {prey_agent.epsilon}, Predator Epsilon: {predator_agent.epsilon}\n")
            print("Saving current model")
            print("")
            f.flush()
            ep_avg = 0
    f.close()


if __name__ == '__main__':
    env = PreyPredatorEnv(num_prey=1, num_predators=0, grid_size=40, max_steps_per_episode=100000, padding = 10, food_probability=0.1, render_mode="non", prey_split_probability=0, observation_history_length=10, food_energy_gain = 40)
    train_dqn(env, epsilon=0.9999, episodes=1000, avg_length=10)