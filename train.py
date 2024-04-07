import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from env.custom_environment import PreyPredatorEnv
import pygame
import random
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * input_shape[1] * input_shape[2], 32)  # Adjust for output size of conv2
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, action_size)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(self.dropout(x)))
        return self.fc2(x)

    

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha  # Controls how much prioritization is used, 0 means uniform
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Store priorities for each experience
        self.max_priority = 1.0  # Start with a high priority for every new experience

    def push(self, state, action, reward, next_state, done):
        # Store the new experience with the maximum current priority
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta=0.4):
        # Convert priorities to probabilities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        
        # Sample experiences based on these probabilities
        sampled_indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sample_probs)
        samples = [self.buffer[idx] for idx in sampled_indices]
        
        # Compute importance-sampling weights to correct bias
        total = len(self.buffer)
        weights = (total * sample_probs[sampled_indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        
        return samples, weights, sampled_indices

    def update_priorities(self, indices, new_priorities):
        # Update priorities based on TD error
        new_priorities = new_priorities.flatten()
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority.item()
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
    
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
    def __init__(self, input_shape = 24, action_size = 5, mode = 'train', epsilon_decay = 0.995, history_length = 5):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = ReplayBuffer(10000000)
        self.beta_start = 0.4  # Start value of beta
        self.beta_increment_per_sampling = 0.001
        self.beta = self.beta_start
        self.gamma = 0.99  # discount rate
        if mode == 'train':
            self.epsilon = 1.0  # exploration rate
        else:
            self.epsilon = 0.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.model = DQN(input_shape, action_size).to(device)
        self.target_model = DQN(input_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00625)
        self.update_target_model() 

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update the target model to match the current model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, file_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, file_name)

    @classmethod
    def load(cls, file_name, mode = 'train', env = None):
        input_shape = (3, env.observation_size, env.observation_size)
        agent = cls(input_shape = input_shape, mode = mode, history_length = env.observation_history_length)
        checkpoint = torch.load(file_name, map_location=device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        return agent

    def act(self, state, test = False):
        if np.random.rand() <= self.epsilon and not test:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
        action_values = self.model(state)
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
        next_q_values = self.target_model(next_states).detach().max(1)[0]
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
        
def test_dqn(env, agent, episodes=10):
    """Test the DQN agent over a specified number of episodes."""
    agent.model.eval()
    total_reward = 0
    for episode in tqdm(range(episodes)):
        env.reset()
        state = env.initial_obs
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, test=True)  # Ensure the agent selects the action with the highest Q-value
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break
        total_reward += episode_reward
    avg_reward = total_reward / episodes
    print(f"Test Episodes: {episodes}, Average Reward: {avg_reward}")
    agent.model.train()
    return avg_reward
    
def train_dqn(env, episodes = 1000, epsilon_decay = 0.995, avg_length = 10, target_update_freq = 100, load_saved = False):
    # Assume the state_size and action_size are the same for both types of agents for simplicity
    input_shape = (3, env.observation_size, env.observation_size)  # Assuming 3 channels for prey, predators, and food
    action_size = env.action_space.n
    if load_saved:
        prey_agent = DQNAgent.load("prey.pth", env = env)
        predator_agent = DQNAgent(input_shape, action_size, epsilon_decay = epsilon_decay, history_length=env.observation_history_length)
    else:
        prey_agent = DQNAgent(input_shape, action_size, epsilon_decay = epsilon_decay, history_length=env.observation_history_length)
        predator_agent = DQNAgent(input_shape, action_size, epsilon_decay = epsilon_decay, history_length=env.observation_history_length)
    log = []
    batch_size = 16
    ep_avg = 0
    count = 0
    for e in range(episodes):
        env.reset()
        # Example state initialization; adjust according to your environment's observation space
        prey_state = env.initial_obs
        if e % target_update_freq == 0:
            prey_agent.update_target_model()
            predator_agent.update_target_model()

        for time in range(100):  # Assuming a max number of steps per episode
            if env.render_mode == 'human':
                event = pygame.event.get()
                
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
        reward = env.get_average_rewards()
        count += 1
        ep_avg += reward
        print(f"{e + 1}/{episodes} done! Reward: {reward} Running Average: {ep_avg / count}")
        if (e + 1) % avg_length == 0:
            count = 0
            test_avg = test_dqn(env, prey_agent, episodes=avg_length)
            predator_agent.save('predator.pth')
            prey_agent.save('prey.pth')
            # Example logging
            row = {
                "Episode": e + 1,
                "Total Episodes": episodes,
                "Train Score": ep_avg / avg_length,
                "Test Score": test_avg,
                "Prey Epsilon": prey_agent.epsilon,
                "Predator Epsilon": predator_agent.epsilon
            }
            log.append(row)
            print(f"Saving current model with avg: {ep_avg/avg_length}")
            print("")
            pd.DataFrame(log, columns=list(row.keys())).to_csv("log.csv", index=None)
            ep_avg = 0

if __name__ == '__main__':
    env = PreyPredatorEnv(num_prey=1, num_predators=0, grid_size=10, max_steps_per_episode=100000, food_probability=1, max_food_count = 1, render_mode="non", prey_split_probability=0, observation_history_length=10, food_energy_gain = 40, std_dev=0.6)
    train_dqn(env, epsilon_decay=0.99999, episodes=20000, avg_length=1000, target_update_freq=250, load_saved=False)
