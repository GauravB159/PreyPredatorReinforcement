from env.custom_environment import PreyPredatorEnv
import numpy as np
from time import sleep

def simple_random_policy(agent):
    return np.random.choice([0, 1, 2, 3, 4])  # Assuming 5 possible actions: stay, up, down, left, right

env = PreyPredatorEnv(num_prey=100, num_predators=25, grid_size=130, max_steps_per_episode=10000, padding = 10)
env.pygame_init()

for episode in range(1):  # Run a single episode for testing
    env.reset()
    termination = False
    count = 0
    while True:
        for agent in env.agent_iter(max_iter=env.stored_num_predators + env.stored_num_prey):  # Loop through each agent
            observation, reward, termination, truncation, info = env.last()  # Get the last state for the agent
            action = simple_random_policy(agent)  # Decide action based on a simple policy
            env.step(action)  # Apply action
        env.render()  # Render the state of the environment
        count += 1
        if count > env.max_steps_per_episode:  # You need to define this condition
            break