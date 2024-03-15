from env.custom_environment import PreyPredatorEnv
import numpy as np
from time import sleep

def simple_random_policy(agent):
    return np.random.choice([0, 1, 2, 3, 4])  # Assuming 5 possible actions: stay, up, down, left, right

env = PreyPredatorEnv(num_prey=10, num_predators=2, grid_size=10)
env.pygame_init()

for episode in range(1):  # Run a single episode for testing
    env.reset()
    termination = False
    count = 0
    while True:
        for agent in env.agent_iter(max_iter=12):  # Loop through each agent
            observation, reward, termination, truncation, info = env.last()  # Get the last state for the agent
            action = simple_random_policy(agent)  # Decide action based on a simple policy
            print(f"Step: {count + 1} Agent: {agent} Action: {action}")
            env.step(action)  # Apply action
        sleep(0.2)
        env.render()  # Render the state of the environment

        # Check for some termination condition to end the episode
        # For simplicity, let's just run a fixed number of steps
        count += 1
        if count > 100:  # You need to define this condition
            break