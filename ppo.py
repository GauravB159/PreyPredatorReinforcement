import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from custom_environment import PreyPredatorEnv
import pandas as pd
import pygame
import json

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
        self.memory = Memory()
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

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).detach()

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
        self.memory.clear_memory()

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
        
class PPOTrainer:
    def __init__(self, config_name, load = False, render_mode = "non", test = False) -> None:
        config = json.loads(open(f"configs/{config_name}.json").read())
        config["name"] = config_name
        self.config = config
        self.env = PreyPredatorEnv(render_mode=render_mode, **config)
        self.test = test
        self.max_episodes = config["max_episodes"] 
        self.max_timesteps = config["max_timesteps"]
        self.update_timestep = config["update_timestep"]
        self.logging_interval = config["logging_interval"]
        self.save_interval = config["save_interval"]
        self.switch_train_episodes = config["switch_train_episodes"]
        self.observation_space_dim = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]*self.env.observation_space.shape[2]
        self.action_space_dim = self.env.action_space.n
        self.prey_ppo = PPO(self.observation_space_dim, self.action_space_dim)
        self.predator_ppo = PPO(self.observation_space_dim, self.action_space_dim)
        self.currently_training = "prey"
        self.logs = []
        if load:
            self.prey_ppo.load_model(f"models/{config['name']}/prey_agent.pth")
            self.predator_ppo.load_model(f"models/{config['name']}/predator_agent.pth")
        self.reset()
        
    def reset(self):
        self.timestep_count = 0
        self.prey_rewards = []
        self.predator_rewards = []
        self.avg_prey_rewards = []
        self.avg_predator_rewards = []
        self.avg_length = 0
    
    def reset_episode(self):
        self.env.reset()
        self.env_state = self.env.initial_obs
        self.ep_predator_reward = 0
        self.ep_prey_reward = 0
        self.prey_reward_count = 0
        self.predator_reward_count = 0
        self.ep_length = 0
        
    def process_timestep(self, timestep):
        self.avg_length += 1
        self.ep_length += 1
        self.timestep_count += 1
        # Select and perform actions for both prey and predator
        agent_count = self.env.stored_num_predators + self.env.stored_num_prey
        agents_moved = 0
        while agents_moved < agent_count:
            if self.env.render_mode == 'human':
                event = pygame.event.get()
            current_agent = self.env.agent_selection
            if self.env.terminations[current_agent]:
                self.env.agent_selection = self.env._agent_selector.next()
                continue
            agents_moved += 1
            agent_x, agent_y = self.env.agents_positions[current_agent] 
            self.env_state[0, agent_x, agent_y] = 1
            self.env_state = self.env_state.reshape(-1)
            if 'prey' in current_agent:
                action = self.prey_ppo.select_action(self.env_state, self.prey_ppo.memory if self.currently_training == "prey" else None)
            else:
                action = self.predator_ppo.select_action(self.env_state, self.predator_ppo.memory if self.currently_training == "predator" else None)

            # Step the environment
            next_state, reward, done = self.env.step(action)
            self.env_state = next_state
            # Update memories
            if 'prey' in current_agent:
                self.prey_ppo.memory.rewards.append(reward)
                self.prey_ppo.memory.is_terminals.append(done)
                self.prey_reward_count += 1
                self.ep_prey_reward += reward
            else:
                self.predator_ppo.memory.rewards.append(reward)
                self.predator_ppo.memory.is_terminals.append(done)
                self.predator_reward_count += 1
                self.ep_predator_reward += reward

        # Update agents if it's time
        if not self.test and (self.timestep_count + 1) % self.update_timestep == 0:
            if self.currently_training == 'prey':
                self.prey_ppo.update()
            else:
                self.predator_ppo.update()
            self.timestep_count = 0
        self.env.render()
        if self.env.stored_num_prey == 0:
            return True
        return False
    
    def run_episode(self, episode):
        
        if episode % self.switch_train_episodes == 0:
            print(f"Currently training: {currently_training}")
            currently_training = "prey" if currently_training == "predator" else "predator"
            print(f"Now switching to: {currently_training}")
            self.timestep_count = 0
            self.prey_ppo.memory.clear_memory()
            self.predator_ppo.memory.clear_memory()
        for t in range(self.max_timesteps):
            completed = self.process_timestep(t)
            if completed:
                break
        avg_ep_prey_reward = self.ep_prey_reward / self.prey_reward_count
        avg_ep_predator_reward = self.ep_predator_reward / self.predator_reward_count if self.predator_reward_count else 0
        self.prey_rewards.append(self.ep_prey_reward)
        self.predator_rewards.append(self.ep_predator_reward)
        self.avg_prey_rewards.append(avg_ep_prey_reward)
        self.avg_predator_rewards.append(avg_ep_predator_reward)
        
        # Save models periodically
        if not self.test and episode % self.save_interval == 0:
            self.prey_ppo.save_model(f"models/{self.config['name']}/prey_agent.pth")
            self.predator_ppo.save_model(f"models/{self.config['name']}/predator_agent.pth")
        print(f'Episode {episode}\tEpisode Length: {self.ep_length}\tPrey Reward: {self.ep_prey_reward:.2f}\tPredator Reward: {self.ep_predator_reward:.2f}')
        if not self.test and episode % self.logging_interval == 0:
            avg_length = int(avg_length / self.logging_interval)
            avg_prey_reward = sum(self.avg_prey_rewards)/len(self.prey_rewards)
            avg_predator_reward = sum(self.avg_predator_rewards)/len(self.predator_rewards)
            prey_reward = sum(self.prey_rewards)/len(self.prey_rewards)
            predator_reward = sum(self.predator_rewards)/len(self.predator_rewards)
            log = {
                "Episode": episode,
                "Avg length": avg_length, 
                "Total prey reward": prey_reward,
                "Avg prey reward": avg_prey_reward,
                "Total predator reward": predator_reward,
                "Avg predator reward": avg_predator_reward
            }
            print(log)
            print()
            self.logs.append(log)
            pd.DataFrame(self.logs).to_csv(f"{self.config['name']}/log.csv", index=None)
            self.reset()
    
    def run(self):
        for episode in range(1, self.max_episodes+1):
            self.reset_episode()
            self.run_episode(episode=episode) 
        self.env.close()

def run(load = False, test = False, render_mode = "non", config_name = None):
    trainer = PPOTrainer(config_name=config_name, load=load, render_mode=render_mode, test=test)
    trainer.run()