"""
Usage: from root directory:
    Train the model and test it:
        python existing_envs/ppo.py train test
    Train the model and save the model in the existing_envs directory:
        python existing_envs/ppo.py train 
    Test the model with the latest saved model in the existing_envs directory:
        python existing_envs/ppo.py test

"""

import os
import glob
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.distributions.categorical import Categorical

from supersuit import pad_observations_v0, pad_action_space_v0

from pettingzoo.mpe import simple_world_comm_v3

import pygame


TRAINING_EPISODES = 100000
TRAINING_LOG_INTERVAL = 10
TEST_EPISODES = 200
TEST_LOG_INTERVAL = 10

ENV_KWARGS = { # Adjust any necessary environment keyword arguments here
    "num_good": 4,
    "num_adversaries": 2,
    "num_food": 20,
    "num_forests": 0,
    "num_obstacles": 0,
    "max_cycles": 25,
    "continuous_actions": False
}  

class Agent(nn.Module):
    def __init__(self, num_actions, observation_size):
        super().__init__()
        self.network = nn.Sequential(
            self._layer_init(nn.Linear(observation_size, 512)),  # 420 is the size of your observation
            nn.ReLU(),
            self._layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()    
        # print("Action:", action)
        # print("Probs:", probs.log_prob(action))
        # print("Entropy:", probs.entropy())
        # print("Critic:", self.critic(hidden))  
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts MPE style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)

    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts MPE style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to MPE style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 4
    max_cycles = ENV_KWARGS["max_cycles"]

    """ ENV SETUP """
    env = simple_world_comm_v3.parallel_env(
        render_mode=None, **ENV_KWARGS
    )

    env = pad_observations_v0(env)  # Pad observations to ensure consistency
    env = pad_action_space_v0(env)  # Pad action spaces to ensure consistency

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions, observation_size=observation_size[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    test_total_episodic_return = 0
    rb_obs = torch.zeros((max_cycles, num_agents, *observation_size)).to(device)  # Adjust the dimensions as necessary
    rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
    rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    test_rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
    rb_values = torch.zeros((max_cycles, num_agents)).to(device)

    if("train" in sys.argv):
        """ TRAINING LOGIC """
        logging_output_file = f"existing_envs/log_training_{time.strftime('%Y%m%d-%H%M%S')}.csv"

        logging.basicConfig(filename=logging_output_file, level=logging.INFO, 
                format='%(message)s')
        
        print("Logging data to " + logging_output_file)
        logging.info("Episode,Train_Return,Episode_Length,Test_Return,Value_Loss,Policy_Loss,Old_Approx_KL,Approx_KL,Clip_Fraction,Explained_Variance")
        # train for n number of episodes
        for episode in range(TRAINING_EPISODES):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = env.reset(seed=None)
                # reset the episodic return
                total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, max_cycles):
                    # rollover the observation
                    obs = batchify_obs(next_obs, device)

                    # get action from the agent
                    actions, logprobs, _, values = agent.get_action_and_value(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = env.step(
                        unbatchify(actions, env)
                    )

                    # add to episode storage
                    rb_obs[step] = obs
                    rb_rewards[step] = batchify(rewards, device)
                    rb_terms[step] = batchify(terms, device)
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()

                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        break

            # bootstrap value if not done
            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards).to(device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t]
                        + gamma * rb_values[t + 1] * rb_terms[t + 1]
                        - rb_values[t]
                    )
                    rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
                rb_returns = rb_advantages + rb_values

            # convert our episodes to batch of individual transitions
            b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
            b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
            b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
            b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
            b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
            b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(3):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = agent.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[batch_index]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[batch_index] * ratio
                    pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss

                    # value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(total_episodic_return)}")
            print(f"Episode Length: {end_step}")
            print("")
            print(f"Value Loss: {v_loss.item()}")
            print(f"Policy Loss: {pg_loss.item()}")
            print(f"Old Approx KL: {old_approx_kl.item()}")
            print(f"Approx KL: {approx_kl.item()}")
            print(f"Clip Fraction: {np.mean(clip_fracs)}")
            print(f"Explained Variance: {explained_var.item()}")
            print("\n-------------------------------------------\n")

            agent.eval()

            with torch.no_grad():
                test_total_episodic_return = 0
                obs, infos = env.reset(seed=None)
                obs = batchify_obs(obs, device)
                terms = [False]
                truncs = [False]
                counter = 0
                while not any(terms) and not any(truncs):
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                    obs = batchify_obs(obs, device)
                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]
                    
                    test_rb_rewards[counter] = batchify(rewards, device)

                    test_total_episodic_return += test_rb_rewards[counter].cpu().numpy()
                    counter += 1

                print(f"Test episode {episode}")
                print(f"Test Episodic Return: {np.mean(test_total_episodic_return)}")
                # print(f"Episode Length: {end_step}")
                print("")
                # print(f"Value Loss: {v_loss.item()}")
                # print(f"Policy Loss: {pg_loss.item()}")
                # print(f"Old Approx KL: {old_approx_kl.item()}")
                # print(f"Approx KL: {approx_kl.item()}")
                # print(f"Clip Fraction: {np.mean(clip_fracs)}")
                # print(f"Explained Variance: {explained_var.item()}")
                print("\n-------------------------------------------\n")

            agent.train()

            if episode % TRAINING_LOG_INTERVAL == 0:
                logging.info(f"{episode}, {np.mean(total_episodic_return)}, {end_step}, {np.mean(test_total_episodic_return)},{v_loss.item()}, {pg_loss.item()}, {old_approx_kl.item()}, {approx_kl.item()}, {np.mean(clip_fracs)}, {explained_var.item()}")

        torch.save(agent.state_dict(), f"existing_envs/simple_world_comm_ppo_{time.strftime('%Y%m%d-%H%M%S')}.pth")


    """ TEST THE POLICY """

    if("test" in sys.argv):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging_output_file = f"existing_envs/log_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"

        logging.basicConfig(filename=logging_output_file, level=logging.INFO, 
                format='%(message)s')
        
        print("Logging data to " + logging_output_file)
        logging.info("Episode,Test_Return")

        try:
            latest_policy = max(
                glob.glob(f"existing_envs/simple_world_comm_ppo*.pth"), key=os.path.getctime
            )
        except ValueError:
            print("Policy not found.")
            exit(0)

        model = Agent(num_actions=num_actions, observation_size=observation_size[0])  # replace with your model class
        model.load_state_dict(torch.load(latest_policy))
        model.eval()  # sets the model to evaluation mode
        env = simple_world_comm_v3.parallel_env(
            render_mode=None, **ENV_KWARGS
        )

        env = pad_observations_v0(env)  # Pad observations to ensure consistency
        env = pad_action_space_v0(env)  # Pad action spaces to ensure consistency

        agent.eval()

        # SCREENWIDTH = 700
        # SCREENHEIGHT = 700
        # screen = pygame.Surface([SCREENWIDTH, SCREENHEIGHT])
        # screen = pygame.display.set_mode(screen.get_size())

        with torch.no_grad():
            # render 5 episodes out
            for episode in range(TEST_EPISODES):
                test_total_episodic_return = 0
                obs, infos = env.reset(seed=None)
                obs = batchify_obs(obs, device)
                terms = [False]
                truncs = [False]
                counter = 0
                while not any(terms) and not any(truncs):
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                    if counter % 10 == 0:
                        print(rewards)
                    obs = batchify_obs(obs, device)
                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]
                    
                    test_rb_rewards[counter] = batchify(rewards, device)
                    
                    test_total_episodic_return += rb_rewards[counter].cpu().numpy()
                    counter += 1

                print(f"Test episode {episode}")
                print(f"Episodic Return: {np.mean(test_total_episodic_return)}")
                # print(f"Episode Length: {end_step}")
                print("")
                # print(f"Value Loss: {v_loss.item()}")
                # print(f"Policy Loss: {pg_loss.item()}")
                # print(f"Old Approx KL: {old_approx_kl.item()}")
                # print(f"Approx KL: {approx_kl.item()}")
                # print(f"Clip Fraction: {np.mean(clip_fracs)}")
                # print(f"Explained Variance: {explained_var.item()}")
                print("\n-------------------------------------------\n")
                if(episode % TEST_LOG_INTERVAL == 0):
                    logging.info(f"{episode}, {np.mean(total_episodic_return)}")