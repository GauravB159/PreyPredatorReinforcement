from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gym import spaces
import pygame
import sys

class PreyPredatorEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_prey=10, num_predators=2, grid_size=10, initial_energy=100, reproduction_energy=200):
        super().__init__()
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.grid_size = grid_size
        self.initial_energy = initial_energy
        self.reproduction_energy = reproduction_energy
        self.max_steps_per_episode = 100
        self.energy_gain_from_eating = 1
        # Define action and observation space
        self.action_space = spaces.Discrete(5) # Example: 0 = stay, 1-4 = move in directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)

        self.agents = [f"prey_{i}" for i in range(self.num_prey)] + [f"predator_{j}" for j in range(self.num_predators)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Initialize state variables
        self.reset()
        
    def observe(self, agent):
        # Initialize an observation array with zeros or another baseline value
        observation = np.zeros((self.grid_size, self.grid_size))

        # Example: Simple observation that marks the agent's position and others
        if self.terminations[agent]:
            return
        agent_x, agent_y = self.agents_positions[agent]
        observation[agent_x][agent_y] = 1  # Mark the agent's own position

        # Optionally, include positions of other agents in the observation
        for other_agent, position in self.agents_positions.items():
            if agent != other_agent:  # Ensure we don't include the agent itself
                if 'prey' in agent and 'prey' in other_agent or 'predator' in agent and 'predator' in other_agent:
                    # Mark positions of same type agents differently, for example, with a 2
                    observation[position[0]][position[1]] = 2
                else:
                    # Mark positions of other types of agents, for example, with a 3
                    observation[position[0]][position[1]] = 3

        # Depending on your scenario, you might also include information about the agent's energy level or other relevant state variables
        # This could be appended to the observation or managed in a structured observation space (e.g., a dictionary)

        return observation


    def reset(self):
        # Reset or initialize agents' states
        self.agents_positions = {agent: (np.random.randint(self.grid_size), np.random.randint(self.grid_size)) for agent in self.agents}
        self.agents_energy = {agent: self.initial_energy for agent in self.agents}
        self.agents_alive = {agent: True for agent in self.agents}  # Track whether agents are alive

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.steps = 0  # Reset step count

        # Reset agent selector for turn-based action selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def step(self, action):
        agent = self.agent_selection
        if not self.terminations[agent]:  # Proceed only if the agent's episode is not done
            # Apply action and update environment state
            if self.agents_alive[agent]:  # Proceed only if the agent is alive
                # Example movement action implementation
                if action == 1:  # Move up
                    self.agents_positions[agent] = (max(self.agents_positions[agent][0] - 1, 0), self.agents_positions[agent][1])
                elif action == 2:  # Move down
                    self.agents_positions[agent] = (min(self.agents_positions[agent][0] + 1, self.grid_size - 1), self.agents_positions[agent][1])
                elif action == 3:  # Move left
                    self.agents_positions[agent] = (self.agents_positions[agent][0], max(self.agents_positions[agent][1] - 1, 0))
                elif action == 4:  # Move right
                    self.agents_positions[agent] = (self.agents_positions[agent][0], min(self.agents_positions[agent][1] + 1, self.grid_size - 1))
                # Example energy consumption for moving
                self.agents_energy[agent] -= 1  # Deduct energy for taking a step

                # Example interaction: Predation or eating
                # You'll need to implement logic to check for such interactions based on positions and agent types
                if 'predator' in self.agent_selection:
                    predator_pos = self.agents_positions[self.agent_selection]
                    for prey_agent in [agent for agent in self.agents_positions.keys() if 'prey' in agent]:
                        prey_pos = self.agents_positions[prey_agent]
                        if predator_pos == prey_pos:
                            # Predation event: predator and prey are in the same position
                            print(f"{self.agent_selection} has eaten {prey_agent}")
                            
                            # Example: Remove the prey from the simulation
                            self.agents_positions.pop(prey_agent)
                            # Mark the prey as done (or removed)
                            self.terminations[prey_agent] = True
                            
                            # Optionally, update predator state (e.g., increase energy, contribute to reproduction counter)
                            self.agents_energy[self.agent_selection] += self.energy_gain_from_eating
                            
                            # If a predator can only eat once per turn, break here
                            break
                # Check for reproduction or death conditions
                # E.g., split or remove agents based on energy levels or other conditions
            # Update reward for the action taken
            reward = self.calculate_reward(agent, action)  # You'll need to implement this based on your game's rules
            self._cumulative_rewards[agent] += reward

            # Example condition to check if the agent's episode is done
            if self.agents_energy[agent] <= 0:  # Or any other condition
                self.terminations[agent] = True
                
            if self.steps >= self.max_steps_per_episode:
                self.truncations[agent] = True
                self.terminations[agent] = True  # Mark done as well when truncating

        # Proceed to next agent
        self.agent_selection = self._agent_selector.next()

    def get_position(self, agent):
        return self.agents_positions.get(agent, (None, None))


    def calculate_reward(self, agent, action):
        # Example reward structure
        reward = 0
        
        # Determine the type of agent
        if 'prey' in agent:
            # Example: Prey receives a positive reward for surviving a step
            reward += 1
            # Additional rewards or penalties can be added based on specific actions or outcomes
            # For example, consuming food could increase the reward
            # Being caught by a predator could result in a large negative reward
        elif 'predator' in agent:
            # Example: Predator receives a reward for moving closer to prey or capturing prey
            reward += 1
            # Similar to prey, you can add conditions for additional rewards or penalties
            # For example, capturing prey could give a significant positive reward

        return reward

    def pygame_init(self):
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.myfont = pygame.font.SysFont('Arial', 30)  # Create a font object (Arial, size 30)
        self.screen_size = 600  # Example size, adjust as needed
        self.cell_size = self.screen_size // self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pygame.time.Clock()  # For controlling the frame rate

    def pygame_quit(self):
        pygame.quit()
        sys.exit()


    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))  # Fill the screen with white

        for agent, (x, y) in self.agents_positions.items():
            if 'prey' in agent:
                color = (0, 255, 0)  # Green for prey
            else:
                color = (255, 0, 0)  # Red for predator

            pygame.draw.rect(self.screen, color, [x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size])

        pygame.display.flip()  # Update the full display Surface to the screen
        num_prey = sum('prey' in agent for agent in self.agents_positions.keys())
        num_predators = sum('predator' in agent for agent in self.agents_positions.keys())

        # Render text surfaces
        prey_text = self.myfont.render(f'Prey: {num_prey}', False, (0, 0, 0))
        predator_text = self.myfont.render(f'Predators: {num_predators}', False, (0, 0, 0))

        # Calculate positions for text (top right corner)
        prey_text_pos = self.screen.get_width() - prey_text.get_width() - 10
        predator_text_pos = self.screen.get_width() - predator_text.get_width() - 10

        # Draw the text surfaces onto the screen
        self.screen.blit(prey_text, (prey_text_pos, 10))
        self.screen.blit(predator_text, (predator_text_pos, 40))

        pygame.display.flip()
        self.clock.tick(60)  # Control the frame rate

    def close(self):
        # If your environment opens files or creates network connections, clean them up here
        # Example: close visualization windows
        self.pygame_quit()
