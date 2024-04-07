import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from env.custom_environment import PreyPredatorEnv
import pandas as pd
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, observation_space_dim, action_space_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_space_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_space_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared_layers(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

class PPO:
    def __init__(self, observation_space_dim, action_space_dim, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCritic(observation_space_dim, action_space_dim).float().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(observation_space_dim, action_space_dim).float().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        
    def save_model(self, file_name):
        torch.save(self.policy.state_dict(), file_name)
        print(f"Model saved to {file_name}")
    
    def load_model(self, file_name):
        self.policy.load_state_dict(torch.load(file_name, map_location=device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"Model loaded from {file_name}")

    def select_action(self, state, memory = None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_probs, _ = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        if memory:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values = self.policy(old_states)
            dist = Categorical(logprobs)
            entropy = dist.entropy()
            new_logprobs = dist.log_prob(old_actions)
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def run(load = False, test = False, render_mode = "non"):
    env = PreyPredatorEnv(num_prey=5, num_predators=1, grid_size=30, max_steps_per_episode=100000, food_probability=1, max_food_count = 5, render_mode=render_mode, prey_split_probability=0, observation_history_length=10, food_energy_gain = 40, std_dev=5, padding = 2)
    observation_space_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
    action_space_dim = env.action_space.n

    # Initialize PPO agent
    prey_ppo = PPO(observation_space_dim, action_space_dim)
    predator_ppo = PPO(observation_space_dim, action_space_dim)
    if load:
        prey_ppo.load_model("prey_ppo_agent.pth")
        predator_ppo.load_model("predator_ppo_agent.pth")

    # Separate memories for prey and predator
    prey_memory = Memory()
    predator_memory = Memory()
    
    # Training hyperparameters
    max_episodes = 100000  # Adjust accordingly
    max_timesteps = 600  # Adjust accordingly
    update_timestep = 3000  # Update policy every n timesteps
    logging_interval = 20  # Log avg reward after interval
    save_interval = 500
    timestep_count = 0
    prey_rewards = []
    predator_rewards = []
    avg_length = 0
    logs = []
    
    # Adjust training loop to handle both prey and predator
    for episode in range(1, max_episodes+1):
        env.reset()
        env_state = env.initial_obs
        ep_predator_reward = 0
        ep_prey_reward = 0
        for t in range(max_timesteps):
            avg_length += 1
            timestep_count += 1
            # Select and perform actions for both prey and predator
            agent_count = env.stored_num_predators + env.stored_num_prey
            agents_moved = 0
            while agents_moved < agent_count:
                if env.render_mode == 'human':
                    event = pygame.event.get()
                current_agent = env.agent_selection
                agents_moved += 1
                if env.terminations[current_agent]:
                    continue
                agent_x, agent_y = env.agents_positions[current_agent] 
                env_state[0, agent_x, agent_y] = 1
                env_state = env_state.reshape(-1)
                if 'prey' in current_agent:
                    action = prey_ppo.select_action(env_state, prey_memory)
                else:
                    action = predator_ppo.select_action(env_state, predator_memory)

                # Step the environment
                next_state, reward, done = env.step(action)
                env_state = next_state
                # Update memories
                if 'prey' in current_agent:
                    prey_memory.rewards.append(reward)
                    prey_memory.is_terminals.append(done)
                    ep_prey_reward += reward
                else:
                    predator_memory.rewards.append(reward)
                    predator_memory.is_terminals.append(done)
                    ep_predator_reward += reward

            # Update agents if it's time
            if not test and (timestep_count + 1) % update_timestep == 0:
                prey_ppo.update(prey_memory)
                predator_ppo.update(predator_memory)
                prey_memory.clear_memory()
                predator_memory.clear_memory()
                timestep_count = 0
            env.render()
            if env.stored_num_prey == 0:
                break
        
        prey_rewards.append(ep_prey_reward)
        predator_rewards.append(ep_predator_reward)
        
        # Save models periodically
        if not test and episode % save_interval == 0:
            prey_ppo.save_model("prey_ppo_agent.pth")
            predator_ppo.save_model("predator_ppo_agent.pth")
        print(f'Episode {episode} \t Prey Reward: {ep_prey_reward:.2f} \t Predator Reward: {ep_predator_reward:.2f}')
        if not test and episode % logging_interval == 0:
            avg_length = int(avg_length/logging_interval)
            avg_prey_reward = sum(prey_rewards)/len(prey_rewards)
            avg_predator_reward = sum(predator_rewards)/len(predator_rewards)
            log = {
                "Episode": episode,
                "Avg length": avg_length, 
                "Avg prey reward": avg_prey_reward,
                "Avg predator reward": avg_predator_reward
            }
            print(log)
            print()
            logs.append(log)
            pd.DataFrame(logs).to_csv("ppo_log.csv", index=None)
            prey_rewards = []
            predator_rewards = []
            avg_length = 0
    
    env.close()