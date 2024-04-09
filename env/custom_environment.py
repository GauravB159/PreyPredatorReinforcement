from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gym import spaces
import pygame
from collections import deque
import sys
import random

class PreyPredatorEnv(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_prey=10, num_predators=2, grid_size=10, initial_energy=100, reproduction_energy=200, max_steps_per_episode = 100, food_probability = 0.05, food_energy_gain = 50, render_mode = 'human', observation_history_length = 5, prey_split_probability = 0.01, max_food_count = 5, generator_params = {}):
        super().__init__()
        self.observation_history_length = observation_history_length
        self.prey_split_probability = prey_split_probability
        self.num_prey = num_prey
        self.render_mode = render_mode
        self.num_predators = num_predators
        self.max_food_count = max_food_count
        self.current_food_count = 0
        self.grid_size = grid_size
        self.initial_energy = initial_energy
        self.reproduction_energy = reproduction_energy
        self.stored_num_predators = -1
        self.screen = None
        self.stored_num_prey = -1
        self.max_steps_per_episode = max_steps_per_episode
        self.max_detection_range = self.grid_size  # Maximum range a ray can detect
        self.no_detection_value = self.max_detection_range + 1  # Value for no detection in a direction
        self.generator_params = generator_params
        # Define action and observation space
        self.action_space = spaces.Discrete(5) # Example: 0 = stay, 1-4 = move in directions
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, self.grid_size, self.grid_size), dtype=int)
        self.agents = [f"prey_{i}" for i in range(self.num_prey)] + [f"predator_{j}" for j in range(self.num_predators)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        self.predator_prey_eaten = {f"predator_{i}": 0 for i in range(num_predators)}
        self.observation_histories = {agent: deque(maxlen=observation_history_length) for agent in self.agents}
        self.reset()
        self.initial_obs = self.observe()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        self.food_probability = food_probability
        self.food_energy_gain = food_energy_gain
        self.food_positions = []
    
    def normal_position(self, mean_position, generator_params):
        x, y = mean_position
        new_x = np.clip(np.random.normal(x, generator_params["std_dev"]), 0, self.grid_size - 1)
        new_y = np.clip(np.random.normal(y, generator_params["std_dev"]), 0, self.grid_size - 1)
        return int(new_x), int(new_y)

    def observe_old(self, agent):
        # Assuming the base observe method returns a numpy array
        current_observation = self.observe(agent)
        self.observation_histories[agent].append(current_observation)
        # Concatenate observations along a new dimension to maintain the order
        extended_observation = np.stack(self.observation_histories[agent], axis=0)
        return extended_observation.flatten()  # Flatten if your model expects flat vectors

        
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
        if agent_type == "prey":
            mean_position = self.generator_coords_prey
        elif agent_type == "predator":
            mean_position = self.generator_coords_predator
        # Use normal_position for the new position if none is provided
        if position is None:
            position = self.normal_position(mean_position, self.generator_params[agent_type])
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
        
    def get_average_rewards(self):
        values = self._cumulative_rewards.values()
        return sum(values) / len(values)
        
    def observe(self, agent=None):
        observation = np.zeros((4, self.grid_size, self.grid_size), dtype=int)

        if agent not in self.agents_positions:
            return observation
        
        for other_agent, (x, y) in self.agents_positions.items():
            if self.terminations[other_agent]:
                continue
            if 'prey' in other_agent:
                observation[1, x, y] = 1  # Prey
            elif 'predator' in other_agent:
                observation[2, x, y] = 1  # Predator

        # Handle food similarly
        for (x, y) in self.food_positions:
            observation[3, x, y] = 1  # Food

        return observation

    def reset(self):
        # Reset or initialize agents' states
        self.generator_coords_prey = (self.grid_size // 2 + self.generator_params["prey"]["padding"], self.generator_params["prey"]["padding"])
        self.generator_coords_predator = (self.generator_params["predator"]["padding"], self.grid_size - self.generator_params["predator"]["padding"] - 1)
        self.generator_coords_food = (self.grid_size - self.generator_params["food"]["padding"] - 1, self.grid_size - self.generator_params["food"]["padding"] - 1)
        self.agents = [f"prey_{i}" for i in range(self.num_prey)] + [f"predator_{j}" for j in range(self.num_predators)]
        self.stored_num_predators = self.num_predators
        self.stored_num_prey = self.num_prey
        self.current_food_count = 0
        self.agents_positions = {f"prey_{i}": self.normal_position(self.generator_coords_prey, self.generator_params["prey"]) for i in range(self.num_prey)}   
        self.agents_positions.update({f"predator_{j}": self.normal_position(self.generator_coords_predator, self.generator_params["predator"]) for j in range(self.num_predators)})     
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
        self.observation_histories = {agent: deque(maxlen=self.observation_history_length) for agent in self.agents}
        if self.render_mode == 'human':
            self.pygame_init()

    def generate_food(self):
        if random.random() < self.food_probability and self.current_food_count < self.max_food_count:
            self.current_food_count += 1
            # Add food at random positions, ensuring no duplicates
            new_food_pos = self.normal_position(self.generator_coords_food, self.generator_params["food"])
            if new_food_pos not in self.food_positions and new_food_pos not in self.agents_positions.values():
                self.food_positions.append(new_food_pos)

    def step(self, action):
        self.generate_food()
        agent = self.agent_selection
        done = self.terminations[agent]
        reward = 0
        achievement = None
        if not done:  # Proceed only if the agent's episode is not done
            # Apply action and update environment state
            if self.agents_alive[agent] and self.agents_energy[agent] > 0:  # Proceed only if the agent is alive, or has energy
                if 'prey' in agent and self.agents_positions.get(self.agent_selection) in self.food_positions:
                    # Agent consumes food
                    # print(f"{self.agent_selection} has consumed food")
                    achievement = "Prey ate food"
                    self.current_food_count -= 1
                    self.agents_energy[self.agent_selection] += self.food_energy_gain
                    self.food_positions.remove(self.agents_positions[self.agent_selection])
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
                self.agents_energy[agent] -= 1 # Deduct energy for taking a step
                
                # Example interaction: Predation or eating
                # You'll need to implement logic to check for such interactions based on positions and agent types
                if 'predator' in self.agent_selection:
                    predator_pos = self.agents_positions[self.agent_selection]
                    for prey_agent in [agent for agent in self.agents_positions.keys() if 'prey' in agent]:
                        prey_pos = self.agents_positions[prey_agent]
                        if predator_pos == prey_pos:
                            # Predation event: predator and prey are in the same position
                            # print(f"{self.agent_selection} has eaten {prey_agent}")
                            
                            # Example: Remove the prey from the simulation
                            self.agents_positions.pop(prey_agent)
                            self.predator_prey_eaten[agent] += 1
                            achievement = "Predator ate prey"
                            # Mark the prey as done (or removed)
                            self.terminations[prey_agent] = True
                            self.agents_alive[prey_agent] = False
                            self.agents_energy[prey_agent] = 0
                            # Optionally, update predator state (e.g., increase energy, contribute to reproduction counter)
                            self.agents_energy[self.agent_selection] += self.food_energy_gain
                            
                            # If a predator can only eat once per turn, break here
                            break
                # Check for reproduction or death conditions
                # E.g., split or remove agents based on energy levels or other conditions
            # Update reward for the action taken

            # Example condition to check if the agent's episode is done
            if self.agents_energy[agent] <= 0:
                self.agents_positions.pop(agent)
                self.terminations[agent] = True
                self.agents_alive[agent] = False
                
            if self.steps >= self.max_steps_per_episode:
                self.truncations[agent] = True
                self.terminations[agent] = True  # Mark done as well when truncating

        # Proceed to next agent
        if 'prey' in agent and np.random.rand() < self.prey_split_probability and self.agents_energy[agent] > 120:  # 10% chance for prey to split
            self.add_agent("prey")
        elif 'predator' in agent and self.predator_prey_eaten[agent] >= 2:  # Predator splits after eating 5 prey
            self.add_agent("predator")
            self.predator_prey_eaten[agent] = 0
        reward = self.calculate_reward(agent, action, achievement)  # You'll need to implement this based on your game's rules
        self._cumulative_rewards[agent] += reward
        self.agent_selection = self._agent_selector.next()
        
        num_prey = sum('prey' in agent for agent in self.agents_positions.keys())
        num_predators = sum('predator' in agent for agent in self.agents_positions.keys())
        self.stored_num_predators = num_predators
        self.stored_num_prey = num_prey
        # returns
        return self.observe(agent), reward, done

    def get_position(self, agent):
        return self.agents_positions.get(agent, (None, None))
    
    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    
    def calculate_proximity_reward(self, agent):
        proximity_reward = 0
        agent_pos = self.agents_positions[agent]
        
        if 'prey' in agent:
            # For prey, find the closest predator and food to adjust the reward based on distance
            closest_predator_distance = min([self.calculate_distance(agent_pos, self.agents_positions[a]) for a in self.agents_positions if 'predator' in a], default=float('inf'))
            closest_food_distance = min([self.calculate_distance(agent_pos, f) for f in self.food_positions], default=float('inf'))
            
            # Reward for getting closer to food and penalize for getting closer to a predator
            if closest_food_distance != float('inf'):
                proximity_reward += 1 / (1 + closest_food_distance)  # Scaled inversely by distance to food
            if closest_predator_distance != float('inf'):
                proximity_reward -= 1 / (1 + closest_predator_distance)  # Penalty scaled inversely by distance to predator

        elif 'predator' in agent:
            # For predators, find the closest prey to adjust the reward based on distance
            closest_prey_distance = min([self.calculate_distance(agent_pos, self.agents_positions[a]) for a in self.agents_positions if 'prey' in a], default=float('inf'))
            
            # Reward for getting closer to prey
            if closest_prey_distance != float('inf'):
                proximity_reward += 2 / (1 + closest_prey_distance)  # Reward scaled inversely by distance to prey

        return proximity_reward


    def calculate_reward(self, agent, action, achievement):
        reward = 0  # Survival reward for taking a step.
        if achievement == "Prey ate food" or achievement == "Predator ate prey":
            reward += 1
            
        # reward += self.calculate_proximity_reward(agent)

        # Significant penalty for death to emphasize survival.
        if self.terminations[agent]:
            reward -= 1

        return reward


    def pygame_init(self):
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.myfont = pygame.font.SysFont('sans-serif', 25)  # Create a font object (Arial, size 30)
        self.screen_size = 800  # Example size, adjust as needed
        self.cell_size = self.screen_size // self.grid_size
        self.screen = pygame.display.set_mode((self.screen_size + 200, self.screen_size))
        self.clock = pygame.time.Clock()  # For controlling the frame rate
        self.actual_grid = pygame.Surface((self.screen_size, self.screen_size), pygame.SRCALPHA)
        self.actual_grid.fill((0, 0, 0, 10))

    def pygame_quit(self):
        pygame.quit()
        sys.exit()
        
    def draw_food(self, pos, cell_size = None):
        # Draw food on the grid, maybe as green squares
        if not cell_size:
            cell_size = self.cell_size
        center_x = pos[0] * cell_size + cell_size // 2
        center_y = pos[1] * cell_size + cell_size // 2
        
        # Define the size of the triangle
        triangle_size = cell_size  # Adjust this value as needed
        
        # Calculate the points of the triangle (equilateral for simplicity)
        point1 = (center_x, center_y - triangle_size // 2)
        point2 = (center_x - triangle_size // 2, center_y + triangle_size // 2)
        point3 = (center_x + triangle_size // 2, center_y + triangle_size // 2)
        
        # Draw the triangle
        pygame.draw.polygon(self.screen, (17, 186, 17), [point1, point2, point3]) 
        
    def draw_predator(self, pos, cell_size = None):
        color = (229, 48, 48) # Red for predator
        if not cell_size:
            cell_size = self.cell_size
        pygame.draw.circle(self.screen, color, center=(pos[0]*cell_size + cell_size // 2, pos[1]*cell_size + cell_size // 2), radius= + cell_size // 2)
        
    def draw_prey(self, pos, cell_size = None):
        color = (255, 189, 29)  # Yellow for prey
        if not cell_size:
            cell_size = self.cell_size
        pygame.draw.rect(self.screen, color, [pos[0]*cell_size, pos[1]*cell_size, cell_size, cell_size])

    def render(self, mode='human'):
        if mode != 'human' or not self.screen:
            return
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.actual_grid, (0, 0))

        for food_pos in self.food_positions:
            self.draw_food(food_pos)
            
        max_detection_range = self.grid_size  # Assuming this is the max detection range used in observe method
        for agent, (x, y) in self.agents_positions.items():
            # Draw agent as a circle
            if 'predator' in agent:
                self.draw_predator((x, y))
            else:
                self.draw_prey((x, y))
            # observation = self.observe(agent)
            # directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

            # agent_center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
            # # Draw rays for detected objects
            # for i, (dx, dy) in enumerate(directions):
            #     detection_distance = np.min(observation[i])  # Get the smallest non-default distance in the observation
            #     if detection_distance < max_detection_range + 1:  # Check if something was detected
            #         ray_end = (agent_center[0] + dx * detection_distance * self.cell_size,
            #             agent_center[1] + dy * detection_distance * self.cell_size)
            #         pygame.draw.line(self.screen, (0, 0, 0), agent_center, ray_end, 1)  # Draw ray line
        
        # Render text surfaces
        self.predator_text = self.myfont.render(f'Predators: {self.stored_num_predators}', True, (0, 0, 0))            
        self.prey_text = self.myfont.render(f'Prey: {self.stored_num_prey}', True, (0, 0, 0))            
        self.food_text = self.myfont.render(f'Food: {len(self.food_positions)}', True, (0, 0, 0))
        prey_energy = [self.agents_energy[k] for k in self.agents_energy if "prey" in k]
        predator_energy = [self.agents_energy[k] for k in self.agents_energy if "predator" in k]
        icon_size = 15
        if self.num_prey > 0:
            self.average_prey_energy_text = self.myfont.render(f'Prey Energy: {sum(prey_energy) / len(prey_energy):.2f}', True, (0, 0, 0))
            prey_text_pos = self.screen.get_width() - self.prey_text.get_width() - 10
            prey_energy_text_pos = self.screen.get_width() - self.average_prey_energy_text.get_width() - 10
            self.draw_prey(((prey_text_pos - 20) / icon_size, 10 / icon_size), icon_size)
            self.screen.blit(self.prey_text, (prey_text_pos, 10))
            self.screen.blit(self.average_prey_energy_text, (prey_energy_text_pos, 100))
            
        if self.num_predators > 0:
            self.average_predator_energy_text = self.myfont.render(f'Predator Energy: {sum(predator_energy) / len(predator_energy):.2f}', True, (0, 0, 0))
            predator_energy_text_pos = self.screen.get_width() - self.average_predator_energy_text.get_width() - 10
            predator_text_pos = self.screen.get_width() - self.predator_text.get_width() - 10
            self.draw_predator(((predator_text_pos - 20) / icon_size, 40 / icon_size), icon_size)
            self.screen.blit(self.predator_text, (predator_text_pos, 40))
            self.screen.blit(self.average_predator_energy_text, (predator_energy_text_pos, 130))

        if self.food_probability > 0:
            # Calculate positions for text (top right corner)
            food_text_pos = self.screen.get_width() - self.food_text.get_width() - 10
            # Draw the text surfaces onto the screen
            self.draw_food(((food_text_pos - 20) / icon_size, 70 / icon_size), icon_size)
            self.screen.blit(self.food_text, (food_text_pos, 70))
        for x in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size))
        # Horizontal lines
        for y in range(0, self.screen_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size, y))


        pygame.display.flip()
        self.clock.tick(60)  # Control the frame rate

    def close(self):
        # If your environment opens files or creates network connections, clean them up here
        # Example: close visualization windows
        self.pygame_quit()
