import time
import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv

def evaluate_random_policy(env, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            # Random policy: probability 1/n_actions for each action
            expected_value = 0
            for a in range(n_actions):
                next_state, reward, done = env.get_transition(s, a)
                v_next = V[next_state]
                if done:
                    v_next = 0
                expected_value += (1/n_actions) * (reward + gamma * v_next)
            
            V[s] = expected_value
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break
    return V

def run_random_agent():
    # Configuration
    size = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (1, 2), (2, 2), (3, 1)]
    
    env = GridWorldEnv(size=size, start=start, goal=goal, obstacles=obstacles, render_mode="human")
    
    # Evaluate Random Policy
    print("Evaluating Random Policy...")
    V = evaluate_random_policy(env)
    
    # Set values in env for visualization
    env.set_state_values(V)

    obs, info = env.reset()
    print("Start Random Agent...")
    
    episodes = 3
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        obs, info = env.reset()
        for step in range(50): # Run for 50 steps max
            env.set_episode_step(episode + 1, step + 1)
            action = env.action_space.sample() # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # print(f"Action: {action}, Position: {info['agent_pos']}, Reward: {reward}")
            
            if terminated:
                print("Goal Reached!")
                break
            elif truncated:
                print("Timeout!")
                break
            
            time.sleep(0.05)
        time.sleep(1)
        
    env.close()

if __name__ == "__main__":
    run_random_agent()
