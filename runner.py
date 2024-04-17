from custom_environment import PreyPredatorEnv
import pandas as pd
import pygame
import json
from ppo import PPO
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Runner:
    def __init__(self, config_name, load = False, render_mode = "non", test = False) -> None:
        config = json.loads(open(f"configs/{config_name}.json").read())
        self.logger = SummaryWriter(f"tensorboard_logs/{config_name}")
        self.config_name = config_name
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
        Path(f"models/{self.config_name}").mkdir(parents=True, exist_ok=True)
        Path(f"logs").mkdir(parents=True, exist_ok=True)
        self.logs = []
        self.episode_offset = 0
        if load:
            self.saved_config = open(f"models/{self.config_name}/saved_config.json").read()
            assert json.dumps(json.loads(self.saved_config)) == json.dumps(self.config), "Saved config for this model does not match the config currently being used"
            self.logs = pd.read_csv(f"logs/{self.config_name}.csv").to_dict(orient="records")
            self.prey_ppo.load_model(f"models/{self.config_name}/prey_agent.pth")
            self.predator_ppo.load_model(f"models/{self.config_name}/predator_agent.pth")
            if len(self.logs):
                self.episode_offset = self.logs[-1]["Episode"]
        self.reset_logs()
        
    def reset_logs(self):
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
    
    def log_data(self, episode):
        total_avg_length = int(self.avg_length / self.logging_interval)
        avg_prey_reward = sum(self.avg_prey_rewards)/len(self.prey_rewards)
        avg_predator_reward = sum(self.avg_predator_rewards)/len(self.predator_rewards)
        prey_reward = sum(self.prey_rewards)/len(self.prey_rewards)
        predator_reward = sum(self.predator_rewards)/len(self.predator_rewards)
        
        # Tensorboard
        self.logger.add_scalar("Episode Length", self.ep_length, self.episode_offset + episode)
        self.logger.add_scalar("Total Prey Reward", prey_reward, self.episode_offset + episode)
        self.logger.add_scalar("Total Predator Reward", predator_reward, self.episode_offset + episode)
        self.logger.add_scalar("Average Prey Reward", avg_prey_reward, self.episode_offset + episode)
        self.logger.add_scalar("Average Predator Reward", avg_predator_reward, self.episode_offset + episode)
        self.logger.flush()
        
        # CSV Log
        log = {
            "Episode": self.episode_offset + episode,
            "Avg length": total_avg_length, 
            "Total prey reward": prey_reward,
            "Avg prey reward": avg_prey_reward,
            "Total predator reward": predator_reward,
            "Avg predator reward": avg_predator_reward
        }
        self.logs.append(log)
        pd.DataFrame(self.logs).to_csv(f"logs/{self.config_name}.csv", index=None)
    
    def run_episode(self, episode):
        
        if episode % self.switch_train_episodes == 0:
            print(f"Currently training: {self.currently_training}")
            self.currently_training = "prey" if self.currently_training == "predator" else "predator"
            print(f"Now switching to: {self.currently_training}")
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
            with open(f"models/{self.config_name}/saved_config.json", "w") as f:
                f.write(json.dumps(self.config, indent=4))
            self.prey_ppo.save_model(f"models/{self.config_name}/prey_agent.pth")
            self.predator_ppo.save_model(f"models/{self.config_name}/predator_agent.pth")
        print(f'Episode {self.episode_offset + episode}\tEpisode Length: {self.ep_length}\tPrey Reward: {self.ep_prey_reward:.2f}\tPredator Reward: {self.ep_predator_reward:.2f}')
        
        if not self.test and episode % self.logging_interval == 0:
            self.log_data(episode)
            self.reset_logs()
    
    def run(self):
        for episode in range(1, self.max_episodes+1):
            self.reset_episode()
            self.run_episode(episode=episode) 
        self.logger.close()
        self.env.close()    