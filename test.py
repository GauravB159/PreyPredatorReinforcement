import numpy as np
from env.custom_environment import PreyPredatorEnv  # Ensure this points to the correct path of your environment
from train import DQNAgent  # Ensure this imports your DQNAgent class correctly
import pygame

def test_pettingzoo_env(prey_model_path, predator_model_path, num_episodes=5):
    # Initialize environment
    env = PreyPredatorEnv(num_prey=1, num_predators=0, grid_size=40, max_steps_per_episode=100000, padding = 10, food_probability=0.1, render_mode="human", prey_split_probability=0, observation_history_length=10, food_energy_gain = 40)

    prey_agent = DQNAgent.load(prey_model_path, mode='test', history_length=env.observation_history_length)
    predator_agent = DQNAgent.load(predator_model_path, mode='test', history_length=env.observation_history_length)

    for episode in range(num_episodes):
        env.reset()
        done = False
        count = 0
        while True:
            if env.render_mode == 'human':
                event = pygame.event.get()
            for agent in env.agent_iter(max_iter=env.stored_num_predators + env.stored_num_prey):
                observation, _, done, _, _ = env.last()
                if done:
                    action = None
                else:
                    if 'prey' in agent:
                        action = prey_agent.act(np.array(observation))
                    elif 'predator' in agent:
                        action = predator_agent.act(np.array(observation))
                env.step(action)
            if env.stored_num_predators + env.stored_num_prey == 0:
                break
            env.render()
            count += 1
            if count > env.max_steps_per_episode:  # You need to define this condition
                break

# Paths to your saved models
prey_model_path = 'prey.pth'
predator_model_path = 'predator.pth'

# Run the test
test_pettingzoo_env(prey_model_path, predator_model_path, 100)
