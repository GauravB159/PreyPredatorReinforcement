import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from env.custom_environment import PreyPredatorEnv

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

def train(load = False):
    env = PreyPredatorEnv(num_prey=1, num_predators=0, grid_size=8, max_steps_per_episode=100000, food_probability=1, max_food_count = 1, render_mode="non", prey_split_probability=0, observation_history_length=10, food_energy_gain = 40, std_dev=0.5)
    observation_space_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
    action_space_dim = env.action_space.n

    # Initialize PPO agent
    ppo = PPO(observation_space_dim, action_space_dim)
    if load:
        ppo.load_model("ppo_agent.pth")
    memory = Memory()

    # Training hyperparameters
    max_episodes = 100000  # Adjust accordingly
    max_timesteps = 100  # Adjust accordingly
    update_timestep = 3000  # Update policy every n timesteps
    logging_interval = 200  # Log avg reward after interval
    save_interval = 50000
    timestep_count = 0
    rewards = []
    avg_length = 0

    # Main training loop
    for episode in range(1, max_episodes+1):
        env.reset()
        state = env.initial_obs.reshape(-1)
        for t in range(max_timesteps):
            timestep_count += 1
            
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done = env.step(action)  # Adjust according to how your env returns values
            
            # Save in memory
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # Update if its time
            if timestep_count % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep_count = 0
            
            if env.stored_num_prey == 0:
                break
            
            state = state.reshape(-1)
            rewards.append(reward)
            env.render()
            if done:
                break

        avg_length += t

        # Logging
        if episode % save_interval == 0:
            ppo.save_model("ppo_agent.pth")
            
        if episode % logging_interval == 0:
            avg_length = int(avg_length/logging_interval)
            avg_reward = sum(rewards)/len(rewards)
            print(f'Episode {episode} \t Avg length: {avg_length} \t Avg reward: {avg_reward:.2f}')
            rewards = []
            avg_length = 0
    
    env.close()

if __name__ == '__main__':
    train()