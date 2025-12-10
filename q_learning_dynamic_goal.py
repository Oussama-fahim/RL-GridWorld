import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
from grid_env import GridWorldEnv
from q_learning_moving_goal import MovingGoalEnv, train_q_learning

class DynamicGoalEnv(MovingGoalEnv):
    def step(self, action):
        # Move agent
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Move goal randomly with some probability (e.g., 20%)
        if not terminated and random.random() < 0.5:
            # Try to move goal to adjacent cell
            direction_map = {
                0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)
            }
            move = direction_map[random.randint(0, 3)]
            new_goal = (self.goal[0] + move[0], self.goal[1] + move[1])
            
            # Check bounds and obstacles
            if (0 <= new_goal[0] < self.size and 
                0 <= new_goal[1] < self.size and 
                new_goal not in self.obstacles and
                new_goal != self.agent_pos): # Don't move on top of agent (unless we want to catch it?)
                
                self.goal = new_goal
                
        # Re-calculate observation because goal moved
        obs = self._get_obs()
        
        # Check termination again in case goal moved onto agent (unlikely with logic above) or agent moved onto goal
        if self.agent_pos == self.goal:
            terminated = True
            reward = 10
            
        return obs, reward, terminated, truncated, info

def run_dynamic_goal_agent():
    size = 7
    start = (0, 0)
    obstacles = [(1, 1), (1, 2), (2, 2), (3, 1)]
    
    # Create environment for training
    # Note: Training on a static goal environment (MovingGoalEnv) might be enough 
    # if the state space covers all (agent, goal) pairs. 
    # The transition dynamics of the goal moving might confuse standard Q-learning 
    # if it treats it as a static environment.
    # However, if the goal moves randomly, the agent just sees itself in a new state (agent, new_goal).
    # So a policy trained on "reach any goal" should work even if the goal moves, 
    # as long as the agent reacts to the new state.
    
    print("Training Q-Learning Agent (on Moving Goal Env)...")
    # We train on the MovingGoalEnv because it explores the state space (agent, goal) effectively.
    # If we trained on DynamicGoalEnv, it would also work, but maybe harder to converge if goal runs away.
    train_env = MovingGoalEnv(size=size, start=start, obstacles=obstacles, render_mode=None)
    Q = train_q_learning(train_env, episodes=15000) 
    print("Training Completed.")
    
    # Test with Dynamic Goal
    env = DynamicGoalEnv(size=size, start=start, obstacles=obstacles, render_mode="human")
    
    for i in range(3):
        print(f"Test Episode {i+1}")
        state, info = env.reset()
        
        terminated = False
        steps = 0
        
        while not terminated and steps < 100: # Limit steps
            # Update Value Function visualization for the CURRENT goal
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
            
            env.set_state_values(V_grid)
            
            steps += 1
            env.set_episode_step(i + 1, steps)

            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.2)
            
        if terminated:
            print("Goal Reached!")
        else:
            print("Timeout!")
            
        time.sleep(1)
        
    env.close()

if __name__ == "__main__":
    run_dynamic_goal_agent()
