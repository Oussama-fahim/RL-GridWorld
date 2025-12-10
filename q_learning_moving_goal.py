import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from grid_env import GridWorldEnv
import time
class MovingGoalEnv(GridWorldEnv):
    def __init__(self, size=5, start=(0, 0), obstacles=None, render_mode=None):
        # Initialize with a dummy goal, we will set it in reset
        super().__init__(size, start, (size-1, size-1), obstacles, render_mode)
        
        # Update observation space to include goal position
        # State = agent_pos_idx * (size*size) + goal_pos_idx
        # This creates a unique state for every combination of agent and goal positions
        self.observation_space = spaces.Discrete((size * size) * (size * size))
        
    def _get_obs(self):
        agent_idx = self.pos_to_state(self.agent_pos)
        goal_idx = self.pos_to_state(self.goal)
        return agent_idx * (self.size * self.size) + goal_idx

    def step(self, action):
        # Execute action to move agent
        # We cannot use super().step(action) because it relies on get_transition 
        # which assumes a simple state space (agent position only).
        
        # 0=Up, 1=Right, 2=Down, 3=Left
        direction_map = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }
        
        delta = direction_map[action]
        new_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])
        
        # Check boundaries
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            if new_pos not in self.obstacles:
                self.agent_pos = new_pos
        
        terminated = False
        reward = -1
        
        if self.agent_pos == self.goal:
            terminated = True
            reward = 10
            
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, False, self._get_info()

    def reset(self, seed=None, options=None):
        # Randomize goal
        while True:
            goal_r = random.randint(0, self.size - 1)
            goal_c = random.randint(0, self.size - 1)
            self.goal = (goal_r, goal_c)
            if self.goal not in self.obstacles and self.goal != self.start:
                break
                
        return super().reset(seed, options)

def train_q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    
    for episode in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 200:
            steps += 1
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
                
            # Take action
            # We need to use the parent step, but we need to override how it calculates the next state index
            # because our state index is complex.
            # However, GridWorldEnv.step calls _get_obs(), which we overrode.
            # So calling super().step(action) should work and return our complex state.
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])
            
            state = next_state
            
    return Q

def run_moving_goal_agent():
    size = 7
    start = (0, 0)
    obstacles = [(1, 1), (1, 2), (2, 2), (3, 1)]
    
    # Create environment for training (no render to be fast)
    env = MovingGoalEnv(size=size, start=start, obstacles=obstacles, render_mode=None)
    
    print("Training Q-Learning Agent with Moving Goal (between episodes)...")
    Q = train_q_learning(env, episodes=15000) # More episodes because state space is larger
    print("Training Completed.")
    
    # Test with rendering
    env = MovingGoalEnv(size=size, start=start, obstacles=obstacles, render_mode="human")
    
    for i in range(3): # Run 3 test episodes
        print(f"Test Episode {i+1}")
        state, info = env.reset()
        
        # Visualize Value Function for the current goal
        current_goal = env.goal
        goal_idx = env.pos_to_state(current_goal)
        
        # Extract V for this goal
        V_grid = np.zeros((size, size))
        for r in range(size):
            for c in range(size):
                agent_pos = (r, c)
                agent_idx = env.pos_to_state(agent_pos)
                # State index in Q-table
                s_idx = agent_idx * (size * size) + goal_idx
                
                # V(s) = max_a Q(s, a)
                V_grid[r, c] = np.max(Q[s_idx])
        
        # Set values in env for visualization
        env.set_state_values(V_grid)
        
        terminated = False
        step = 0
        
        while not terminated and step < 50:
            step += 1
            env.set_episode_step(i + 1, step)
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            
            # print(f"Action: {action}, Pos: {info['agent_pos']}, Goal: {env.goal}")
            time.sleep(0.2)
            
        if terminated:
            print("Goal Reached!")
        else:
            print("Timeout - Goal not reached.")
        time.sleep(1)
        
    env.close()

if __name__ == "__main__":
    run_moving_goal_agent()
