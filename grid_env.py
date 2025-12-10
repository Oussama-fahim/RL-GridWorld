import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=7, start=(0, 0), goal=(4, 4), obstacles=None, render_mode=None):
        super().__init__()
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles is not None else []
        self.render_mode = render_mode
        
        # Define action and observation space
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)
        
        # Observation: Flattened index of the grid (0 to size*size - 1)
        self.observation_space = spaces.Discrete(size * size)
        
        self.agent_pos = start
        self.state_values = None # To store values for visualization
        self.current_episode = 0
        self.current_step = 0
        self.fig, self.ax = None, None

    def set_state_values(self, values):
        """
        Set state values to be displayed in the grid.
        values: 1D array of size size*size or 2D array (size, size)
        """
        self.state_values = values

    def set_episode_step(self, episode, step):
        self.current_episode = episode
        self.current_step = step

    def _get_obs(self):
        return self.pos_to_state(self.agent_pos)
    
    def pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def state_to_pos(self, state):
        return (state // self.size, state % self.size)

    def _get_info(self):
        return {"agent_pos": self.agent_pos}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), self._get_info()

    def get_transition(self, state, action):
        # Returns (next_state, reward, done)
        pos = self.state_to_pos(state)
        
        # 0=Up, 1=Right, 2=Down, 3=Left
        direction_map = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }
        
        delta = direction_map[action]
        new_pos = (pos[0] + delta[0], pos[1] + delta[1])
        
        # Check boundaries
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            if new_pos not in self.obstacles:
                pos = new_pos
        
        next_state = self.pos_to_state(pos)
        
        done = False
        reward = -1
        
        if pos == self.goal:
            done = True
            reward = 10
            
        return next_state, reward, done

    def step(self, action):
        current_state = self._get_obs()
        next_state, reward, terminated = self.get_transition(current_state, action)
        
        self.agent_pos = self.state_to_pos(next_state)
        
        if self.render_mode == "human":
            self.render()
            
        return next_state, reward, terminated, False, self._get_info()

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-0.5, self.size - 0.5)
            self.ax.set_ylim(self.size - 0.5, -0.5) # Invert y axis to match matrix coordinates
            self.ax.grid(True)
            
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(self.size - 0.5, -0.5)
        self.ax.grid(True)
        
        # Draw Goal
        goal_rect = patches.Rectangle((self.goal[1] - 0.5, self.goal[0] - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='green', alpha=0.5)
        self.ax.add_patch(goal_rect)
        self.ax.text(self.goal[1], self.goal[0], 'G', ha='center', va='center', color='white', fontweight='bold')

        # Draw Obstacles
        for obs in self.obstacles:
            obs_rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='black')
            self.ax.add_patch(obs_rect)
            
        # Draw Agent
        agent_circle = patches.Circle((self.agent_pos[1], self.agent_pos[0]), 0.3, color='blue')
        self.ax.add_patch(agent_circle)
        
        # Draw State Values
        if self.state_values is not None:
            flat_values = np.array(self.state_values).flatten()
            for r in range(self.size):
                for c in range(self.size):
                    idx = r * self.size + c
                    if idx < len(flat_values):
                        val = flat_values[idx]
                        self.ax.text(c, r, f'{val:.1f}', ha='center', va='center', color='black', fontsize=8, fontweight='normal')

        # Set Title with Episode and Step
        self.ax.set_title(f"Episode: {self.current_episode} | Step: {self.current_step}")

        plt.pause(0.1)

    def close(self):
        if self.fig:
            plt.close(self.fig)
