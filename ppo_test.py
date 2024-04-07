from env.custom_environment import PreyPredatorEnv
from ppo_train import PPO
import pygame

def test():
    env = PreyPredatorEnv(num_prey=1, num_predators=0, grid_size=25, max_steps_per_episode=100000, food_probability=1, max_food_count = 1, render_mode="human", prey_split_probability=0, observation_history_length=10, food_energy_gain = 40, std_dev=7)
    observation_space_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
    action_space_dim = env.action_space.n

    # Initialize PPO agent
    ppo = PPO(observation_space_dim, action_space_dim)
    ppo.load_model("ppo_agent.pth")

    # Training hyperparameters
    max_episodes = 100  # Adjust accordingly
    max_timesteps = 100  # Adjust accordingly
    logging_interval = 1  # Log avg reward after interval
    timestep_count = 0
    rewards = []
    avg_length = 0

    # Main training loop
    for episode in range(1, max_episodes+1):
        env.reset()
        state = env.initial_obs.reshape(-1)
        for t in range(max_timesteps):
            if env.render_mode == 'human':
                event = pygame.event.get()
            timestep_count += 1
            
            # Running policy_old:
            action = ppo.select_action(state)
            state, reward, done = env.step(action)  # Adjust according to how your env returns values
            
            if env.stored_num_prey == 0:
                break
            
            state = state.reshape(-1)
            rewards.append(reward)
            env.render()
            if done:
                break

        avg_length += t

        if episode % logging_interval == 0:
            avg_length = int(avg_length/logging_interval)
            avg_reward = sum(rewards)/len(rewards)
            print(f'Episode {episode} \t Avg length: {avg_length} \t Avg reward: {avg_reward:.2f}')
            rewards = []
            avg_length = 0
    
    env.close()
    
if __name__ == '__main__':
    test()