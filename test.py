import numpy as np
from env.custom_environment import PreyPredatorEnv  # Ensure this points to the correct path of your environment
from train import DQNAgent  # Ensure this imports your DQNAgent class correctly

def test_pettingzoo_env(prey_model_path, predator_model_path, num_episodes=5):
    # Initialize environment
    env = PreyPredatorEnv(num_prey=2, num_predators=1, grid_size=30, max_steps_per_episode=10000, padding = 10, food_probability=0.005, render_mode="human")

    # Load the trained models for prey and predators
    # Assuming the state_size and action_size can be inferred or are fixed. Adjust as necessary.
    state_size = 24  # Update this based on your actual observation space
    action_size = env.action_space.n  # Assuming this works for your setup

    prey_agent = DQNAgent.load(prey_model_path, mode='test')
    predator_agent = DQNAgent.load(predator_model_path, mode='test')

    for episode in range(num_episodes):
        env.reset()
        for agent in env.agent_iter():
            observation, _, done, _, _ = env.last()
            if done:
                action = None
            else:
                if 'prey' in agent:
                    action = prey_agent.act(np.array(observation))
                elif 'predator' in agent:
                    action = predator_agent.act(np.array(observation))
            env.step(action)
            env.render()

# Paths to your saved models
prey_model_path = 'prey.pth'
predator_model_path = 'predator.pth'

# Run the test
test_pettingzoo_env(prey_model_path, predator_model_path)
