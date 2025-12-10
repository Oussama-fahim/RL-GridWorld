import numpy as np
import time
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv

def value_iteration(env, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    
    iterations = 0
    while True:
        iterations += 1
        delta = 0
        for s in range(n_states):
            v = V[s]
            # Calculate max_a Q(s, a)
            q_values = []
            for a in range(n_actions):
                next_state, reward, done = env.get_transition(s, a)
                # If done, next value is 0 (terminal state value is 0 usually, or we can say V(terminal) = 0)
                # But here get_transition returns the state index of the goal.
                # If we are AT the goal, we are done.
                # The transition logic: from s, take a -> s', r.
                # V(s) = max_a [ r + gamma * V(s') ]
                # If s' is terminal (goal), V(s') should be 0? 
                # Actually, if s' is the goal state, the episode ends.
                # So the value of the goal state itself is 0 (no future reward).
                # But the transition TO the goal gives reward.
                
                v_next = V[next_state]
                if done:
                    v_next = 0
                
                q_values.append(reward + gamma * v_next)
            
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break
            
    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            next_state, reward, done = env.get_transition(s, a)
            v_next = V[next_state]
            if done:
                v_next = 0
            q_values.append(reward + gamma * v_next)
        policy[s] = np.argmax(q_values)
        
    return V, policy, iterations

def run_value_agent():
    # Configuration
    size = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (1, 2), (2, 2), (3, 1)]
    
    env = GridWorldEnv(size=size, start=start, goal=goal, obstacles=obstacles, render_mode="human")
    
    print("Running Value Iteration...")
    V, policy, iterations = value_iteration(env)
    print(f"Value Iteration Converged in {iterations} iterations.")
    
    # Set values in env for visualization
    env.set_state_values(V)
    
    # Run Agent
    print("Start Value Agent...")
    
    episodes = 3
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        obs, info = env.reset()
        
        for step in range(20):
            env.set_episode_step(episode + 1, step + 1)
            action = policy[obs]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # print(f"Action: {action}, Position: {info['agent_pos']}, Reward: {reward}")
            
            if terminated:
                print("Goal Reached!")
                break
            
            time.sleep(0.2)
        time.sleep(1)
        
    env.close()

if __name__ == "__main__":
    run_value_agent()
