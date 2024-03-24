from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gym import spaces
import pygame
import sys
import random

class PreyPredatorEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_prey=10, num_predators=2, grid_size=10, initial_energy=100, reproduction_energy=200, max_steps_per_episode = 100, padding = 10, food_probability = 0.05, food_energy_gain = 50):
        super().__init__()
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.grid_size = grid_size
        self.initial_energy = initial_energy
        self.padding = padding
        self.reproduction_energy = reproduction_energy
        self.stored_num_predators = -1
        self.stored_num_prey = -1
        self.max_steps_per_episode = max_steps_per_episode
        self.energy_gain_from_eating = 100
        # Define action and observation space
        self.action_space = spaces.Discrete(5) # Example: 0 = stay, 1-4 = move in directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)
        
        self.agents = [f"prey_{i}" for i in range(self.num_prey)] + [f"predator_{j}" for j in range(self.num_predators)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        self.predator_prey_eaten = {f"predator_{i}": 0 for i in range(num_predators)}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        self.food_probability = food_probability
        self.food_energy_gain = food_energy_gain
        self.food_positions = []

        # Initialize state variables
        self.reset()
        
    def add_agent(self, agent_type, position=None):
        """
        Adds a new agent of the specified type to the environment.
        """
        if agent_type == "prey":
            new_id = f"prey_{len([a for a in self.agents if 'prey' in a])}"
            self.agents.append(new_id)
        elif agent_type == "predator":
            new_id = f"predator_{len([a for a in self.agents if 'predator' in a])}"
            self.predator_prey_eaten[new_id] = 0  # Initialize prey eaten count
            self.agents.append(new_id)
        
        # Set the initial position
        if position is None:
            position = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.agents_positions[new_id] = position
        self.agents_energy[new_id] = self.initial_energy
        self.agents_alive[new_id] = True
        self.terminations[new_id] = False
        self.truncations[new_id] = False
        self.infos[new_id] = {}
        self._cumulative_rewards[new_id] = 0
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
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
            if 'prey' in agent:
                if abs(position[0] - agent_x) > 4 or abs(position[1] - agent_y) > 4:
                    continue
            else:
                if abs(position[0] - agent_x) > 2 or abs(position[1] - agent_y) > 2:
                    continue
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
        self.food_positions = []
        # Reset agent selector for turn-based action selection
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()


    def step(self, action):
        agent = self.agent_selection
        done = self.terminations[agent]
        reward = 0

        if not done:  # Proceed only if the agent's episode is not done
            # Apply action and update environment state
            if self.agents_alive[agent] and self.agents_energy[agent] > 0:  # Proceed only if the agent is alive, or has energy
                if 'prey' in agent  and self.agents_positions.get(self.agent_selection) in self.food_positions:
                    # Agent consumes food
                    print(f"{self.agent_selection} has consumed food")
                    self.agents_energy[self.agent_selection] += self.food_energy_gain
                    self.food_positions.remove(self.agents_positions[self.agent_selection])
                # Example movement action implementation
                if action == 1:  # Move up
                    self.agents_positions[agent] = (max(self.agents_positions[agent][0] - 1, self.padding), self.agents_positions[agent][1])
                elif action == 2:  # Move down
                    self.agents_positions[agent] = (min(self.agents_positions[agent][0] + 1, self.grid_size - self.padding), self.agents_positions[agent][1])
                elif action == 3:  # Move left
                    self.agents_positions[agent] = (self.agents_positions[agent][0], max(self.agents_positions[agent][1] - 1, self.padding))
                elif action == 4:  # Move right
                    self.agents_positions[agent] = (self.agents_positions[agent][0], min(self.agents_positions[agent][1] + 1, self.grid_size - self.padding))
                # Example energy consumption for moving
                self.agents_energy[agent] -= 1 * (0.1 if 'predator' in agent else 0.5)  # Deduct energy for taking a step
                
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
                            self.predator_prey_eaten[agent] += 1
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
            if self.agents_energy[agent] <= 0:
                self.agents_positions.pop(agent)
                self.terminations[agent] = True
                self.agents_alive[agent] = False
                
            if self.steps >= self.max_steps_per_episode:
                self.truncations[agent] = True
                self.terminations[agent] = True  # Mark done as well when truncating

        # Proceed to next agent
        if 'prey' in agent and np.random.rand() < 0.01 and self.agents_energy[agent] > 70:  # 10% chance for prey to split
            self.add_agent("prey")
        elif 'predator' in agent and self.predator_prey_eaten[agent] >= 5:  # Predator splits after eating 5 prey
            self.add_agent("predator")
            self.predator_prey_eaten[agent] = 0
        self.agent_selection = self._agent_selector.next()

        # returns
        return self.observe(agent), reward, done

    def get_position(self, agent):
        return self.agents_positions.get(agent, (None, None))

    def calculate_reward(self, agent, action):
        reward = 0

        if 'prey' in agent:
            if agent in self.agents_positions and self.agents_positions[agent] in self.food_positions:
                # Prey consumes food, positive reward
                reward += self.food_energy_gain  # or any specific positive value
            if not self.agents_alive.get(agent, True):
                # Prey got eaten, negative reward
                reward -= 100  # Large negative reward for being eaten
            else:
                # Surviving each step gives a small positive reward
                reward += 1

        elif 'predator' in agent:
            # Predators get a reward for eating prey
            prey_eaten = self.predator_prey_eaten.get(agent, 0)
            reward += prey_eaten * self.energy_gain_from_eating  # Reward based on prey eaten
            # Surviving or other objectives might also contribute to the reward
            reward += 1  # small reward for being alive and taking actions

        return reward

    def pygame_init(self):
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.myfont = pygame.font.SysFont('sans-serif', 25)  # Create a font object (Arial, size 30)
        self.screen_size = 800  # Example size, adjust as needed
        self.cell_size = self.screen_size // self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size + 200, self.screen_size))
        self.clock = pygame.time.Clock()  # For controlling the frame rate
        self.actual_grid = pygame.Surface((self.screen_size - 2 * self.padding, self.screen_size), pygame.SRCALPHA)
        self.actual_grid.fill((0, 0, 0, 10))

    def pygame_quit(self):
        pygame.quit()
        sys.exit()
        
    def draw_food(self, pos):
        # Draw food on the grid, maybe as green squares
        center_x = pos[0] * self.cell_size + self.cell_size // 2
        center_y = pos[1] * self.cell_size + self.cell_size // 2
        
        # Define the size of the triangle
        triangle_size = self.cell_size  # Adjust this value as needed
        
        # Calculate the points of the triangle (equilateral for simplicity)
        point1 = (center_x, center_y - triangle_size // 2)
        point2 = (center_x - triangle_size // 2, center_y + triangle_size // 2)
        point3 = (center_x + triangle_size // 2, center_y + triangle_size // 2)
        
        # Draw the triangle
        pygame.draw.polygon(self.screen, (17, 186, 17), [point1, point2, point3]) 
        
    def draw_predator(self, pos):
        color = (229, 48, 48) # Red for predator
        pygame.draw.circle(self.screen, color, center=(pos[0]*self.cell_size, pos[1]*self.cell_size), radius=self.cell_size)
        
    def draw_prey(self, pos):
        color = (255, 189, 29)  # Yellow for prey
        pygame.draw.rect(self.screen, color, [pos[0]*self.cell_size, pos[1]*self.cell_size, self.cell_size, self.cell_size])

    def render(self, mode='human'):
        if random.random() < self.food_probability:
            # Add food at random positions, ensuring no duplicates
            while True:
                new_food_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
                if new_food_pos not in self.food_positions and new_food_pos not in self.agents_positions.values():
                    self.food_positions.append(new_food_pos)
                    break
        if mode != 'human':
            return
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.actual_grid, (0, 0))

        # For transparency, we need a surface that supports alpha
        viewport_size = 4
        transparent_surface = pygame.Surface((self.cell_size * viewport_size * 2, self.cell_size * viewport_size * 2), pygame.SRCALPHA)

        for food_pos in self.food_positions:
            self.draw_food(food_pos)

        for agent, (x, y) in self.agents_positions.items():
            # Draw agent as a circle
            if 'predator' in agent:
                self.draw_predator((x, y))
            else:
                self.draw_prey((x, y))

            if 'prey' in agent:
                transparent_surface.fill((255, 189, 29, 40))
                pygame.draw.rect(transparent_surface, (255, 189, 29, 40), transparent_surface.get_rect(), 1)
            else:
                transparent_surface.fill((229, 48, 48, 40))
                pygame.draw.rect(transparent_surface, (229, 48, 48, 40), transparent_surface.get_rect(), 1)
            
            # Blit the transparent surface onto the screen at the correct position, adjusting for the offset
            self.screen.blit(transparent_surface, (x*self.cell_size - self.cell_size*viewport_size, y*self.cell_size - self.cell_size*viewport_size))

        num_prey = sum('prey' in agent for agent in self.agents_positions.keys())
        num_predators = sum('predator' in agent for agent in self.agents_positions.keys())
        self.stored_num_predators = num_predators
        self.stored_num_prey = num_prey
        # Render text surfaces
        self.predator_text = self.myfont.render(f'Predators: {num_predators}', True, (0, 0, 0))            
        self.prey_text = self.myfont.render(f'Prey: {num_prey}', True, (0, 0, 0))            
        self.food_text = self.myfont.render(f'Food: {len(self.food_positions)}', True, (0, 0, 0))
        prey_energy = [self.agents_energy[k] for k in self.agents_energy if "prey" in k]
        predator_energy = [self.agents_energy[k] for k in self.agents_energy if "predator" in k]
        self.average_prey_energy_text = self.myfont.render(f'Prey Energy: {sum(prey_energy) / len(prey_energy):.2f}', True, (0, 0, 0))
        self.average_predator_energy_text = self.myfont.render(f'Predator Energy: {sum(predator_energy) / len(predator_energy):.2f}', True, (0, 0, 0))
        # Calculate positions for text (top right corner)
        prey_text_pos = self.screen.get_width() - self.prey_text.get_width() - 10
        predator_text_pos = self.screen.get_width() - self.predator_text.get_width() - 10
        food_text_pos = self.screen.get_width() - self.food_text.get_width() - 10
        prey_energy_text_pos = self.screen.get_width() - self.average_prey_energy_text.get_width() - 10
        predator_energy_text_pos = self.screen.get_width() - self.average_predator_energy_text.get_width() - 10

        # Draw the text surfaces onto the screen
        self.draw_prey(((prey_text_pos - 10) / self.cell_size, 15 / self.cell_size))
        self.screen.blit(self.prey_text, (prey_text_pos, 10))
        self.draw_predator(((predator_text_pos - 10) / self.cell_size, 48 / self.cell_size))
        self.screen.blit(self.predator_text, (predator_text_pos, 40))
        self.draw_food(((food_text_pos - 10) / self.cell_size, 75 / self.cell_size))
        self.screen.blit(self.food_text, (food_text_pos, 70))
        self.screen.blit(self.average_prey_energy_text, (prey_energy_text_pos, 100))
        self.screen.blit(self.average_predator_energy_text, (predator_energy_text_pos, 130))

        pygame.display.flip()
        self.clock.tick(60)  # Control the frame rate

    def close(self):
        # If your environment opens files or creates network connections, clean them up here
        # Example: close visualization windows
        self.pygame_quit()
