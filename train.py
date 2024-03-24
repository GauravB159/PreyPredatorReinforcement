import numpy as np
from env.custom_environment import PreyPredatorEnv
import os

log = 'train_log.txt'

# Initialize the environment
env = PreyPredatorEnv(num_prey=10, num_predators=2, grid_size=10)

# Parameters
num_episodes = 10000
learning_rate = 0.1
discount_factor = 0.95

# Initialize Q-tables for each agent, assuming discrete actions
# For simplicity, we're using a single Q-table per agent type
num_actions = env.action_space.n
prey_q_table = np.zeros((env.grid_size, env.grid_size, num_actions))
predator_q_table = np.zeros((env.grid_size, env.grid_size, num_actions))

# delete train_log
if os.path.exists(log):
    os.remove(log)
    with open(log, 'w') as f:
        f.write("{: >0} {: >20} {: >20} {: >20}\n".format('Episode', 'Prey Reward', 'Predator Reward', 'Total Reward'))

for episode in range(num_episodes):
    total_reward = 0
    prey_reward = 0
    predator_reward = 0
    done = False
    env.reset()
    
    while not done:
        for agent in env.agents:
            current_pos = env.get_position(agent)
           
            if env.agents_alive[agent] == False or current_pos == (None, None):
                continue

            if 'prey' in agent:
                # check if prey is alive

                action = np.argmax(prey_q_table[current_pos])
                next_state, reward, done = env.step(action)
                next_pos = env.get_position(agent)
                # Update Q-table for prey
                old_value = prey_q_table[current_pos + (action,)]
                next_max = np.max(prey_q_table[next_pos])
                new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
                prey_q_table[current_pos + (action,)] = new_value
                prey_reward += reward

            elif 'predator' in agent:
                action = np.argmax(predator_q_table[current_pos])
                next_state, reward, done = env.step(action)
                next_pos = env.get_position(agent)
                # Update Q-table for predator
                old_value = predator_q_table[current_pos + (action,)]
                next_max = np.max(predator_q_table[next_pos])
                new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
                predator_q_table[current_pos + (action,)] = new_value
                predator_reward += reward

            total_reward += reward
            if done:
                break
    
    print(f"Episode {episode} Total Reward: {total_reward}")
    
    # write the episode and rewards to a log
    with open(log, 'a') as f:
        f.write("{: >0} {: >20} {: >20} {: >20}\n".format(episode, prey_reward, predator_reward, total_reward))
        f.close()

# Additional code for policy evaluation, saving models, etc., can be added here