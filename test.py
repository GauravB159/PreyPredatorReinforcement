from env.custom_environment import PreyPredatorEnv
import numpy as np
from time import sleep

def simple_random_policy(agent):
    return np.random.choice([0, 1, 2, 3, 4])  # Assuming 5 possible actions: stay, up, down, left, right

env = PreyPredatorEnv(num_prey=100, num_predators=25, grid_size=100, max_steps_per_episode=1000)
env.pygame_init()

for episode in range(1):  # Run a single episode for testing
    env.reset()
    termination = False
    count = 0
    while True:
        for agent in env.agent_iter(max_iter=125):  # Loop through each agent
            observation, reward, termination, truncation, info = env.last()  # Get the last state for the agent
            action = simple_random_policy(agent)  # Decide action based on a simple policy
            env.step(action)  # Apply action
        env.render()  # Render the state of the environment
        print(f"Step: {count + 1}")
        count += 1
        if count > env.max_steps_per_episode:  # You need to define this condition
            break