from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

# Import the simple_world_comm_v3 environment
from pettingzoo.mpe import simple_world_comm_v3

from supersuit import pad_observations_v0, pad_action_space_v0

import pygame

def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Initialize the environment
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Wrap the environment with the necessary SuperSuit wrappers
    env = pad_observations_v0(env)  # Pad observations to ensure consistency
    env = pad_action_space_v0(env)  # Pad action spaces to ensure consistency
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    # Initialize the PPO model with an MLP policy
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    # Train the model
    model.learn(total_timesteps=steps)

    # Save the trained model
    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    # Close the environment
    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = pad_observations_v0(env)  # Pad observations to ensure consistency
    env = pad_action_space_v0(env)  # Pad action spaces to ensure consistency

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    SCREENWIDTH = 700
    SCREENHEIGHT = 700
    screen = pygame.Surface([SCREENWIDTH, SCREENHEIGHT])
    screen = pygame.display.set_mode(screen.get_size())

    pygame.display.update()

    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]
            
            # print(f"Agent {agent}: Action={act}, Observation={obs}, Reward={reward}")

            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward

if __name__ == "__main__":
    env_fn = simple_world_comm_v3  # Change to the simple_world_comm_v3 environment
    env_kwargs = { # Adjust any necessary environment keyword arguments here
        "num_good": 4,
        "num_adversaries": 2,
        "num_food": 20,
        "num_forests": 0,
        "num_obstacles": 0,
        "max_cycles": 25,
        "continuous_actions": False
    }  

    # Train a model
    train_butterfly_supersuit(env_fn, steps=109_608, seed=0, **env_kwargs)

    # Evaluate 10 games
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)