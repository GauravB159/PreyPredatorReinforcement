import numpy as np
from env.custom_environment import PreyPredatorEnv  # Ensure this points to the correct path of your environment
from train import DQNAgent  # Ensure this imports your DQNAgent class correctly
import pygame

def test_pettingzoo_env(prey_model_path, predator_model_path, num_episodes=5):
    # Initialize environment
    env = PreyPredatorEnv(num_prey=1, num_predators=0, grid_size=25, max_steps_per_episode=100000, padding = 10, food_probability=0.1, render_mode="human", prey_split_probability=0, observation_history_length=10, food_energy_gain = 40)

    prey_agent = DQNAgent.load(prey_model_path, mode='test', env=env)
    # predator_agent = DQNAgent.load(predator_model_path, mode='test', env=env)

    for e in range(num_episodes):
        env.reset()
        # Example state initialization; adjust according to your environment's observation space
        prey_state = env.initial_obs

        for time in range(500):  # Assuming a max number of steps per episode
            if env.render_mode == 'human':
                event = pygame.event.get()
                
            # Simplified action selection for both agents (you need logic to distinguish between agent types)
            prey_action = prey_agent.act(prey_state)

            # Your environment needs to handle multi-agent steps (this is highly simplified)
            next_prey_state, prey_reward, done = env.step(prey_action)  # Adjust this method
            prey_state = next_prey_state
            if env.stored_num_prey == 0:
                break
            env.render()
        reward = env.get_average_rewards()
        print(f"{e + 1}/{num_episodes} done! Reward: {reward}")

# Paths to your saved models
prey_model_path = 'prey.pth'
predator_model_path = 'predator.pth'

# Run the test
test_pettingzoo_env(prey_model_path, predator_model_path, 100)
