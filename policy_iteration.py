import numpy as np
import time
import matplotlib.pyplot as plt
from grid_env import GridWorldEnv

def policy_evaluation(env, policy, V, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            a = policy[s]
            
            next_state, reward, done = env.get_transition(s, a)
            v_next = V[next_state]
            if done:
                v_next = 0
            
            # State Value Formula for a fixed policy: V(s) = R + gamma * V(s')
            V[s] = reward + gamma * v_next
            
            delta = max(delta, abs(v - V[s]))
            
        if delta < theta:
            break
    return V

def policy_improvement(env, V, gamma=0.9):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)
    policy_stable = True
    
    for s in range(n_states):
        old_action = policy[s] # Note: In first run this is 0, but we usually pass the old policy. 
                               # Here we reconstruct it greedily from V.
        
        q_values = []
        for a in range(n_actions):
            next_state, reward, done = env.get_transition(s, a)
            v_next = V[next_state]
            if done:
                v_next = 0
            q_values.append(reward + gamma * v_next)
            
        best_action = np.argmax(q_values)
        policy[s] = best_action
        
        # We can't easily check stability without passing the old policy in, 
        # but for this simple implementation, we can just return the new policy.
        
    return policy

def policy_iteration(env, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize arbitrary policy and value function
    policy = np.random.randint(0, n_actions, n_states)
    V = np.zeros(n_states)
    
    iterations = 0
    while True:
        iterations += 1
        # 1. Policy Evaluation
        V = policy_evaluation(env, policy, V, gamma, theta)
        
        # 2. Policy Improvement
        new_policy = policy_improvement(env, V, gamma)
        
        # Check if policy changed
        if np.array_equal(new_policy, policy):
            break
        
        policy = new_policy
        
    return V, policy, iterations

def run_policy_iteration_agent():
    # Configuration
    size = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (1, 2), (2, 2), (3, 1)]
    
    env = GridWorldEnv(size=size, start=start, goal=goal, obstacles=obstacles, render_mode="human")
    
    print("Running Policy Iteration...")
    V, policy, iterations = policy_iteration(env)
    print(f"Policy Iteration Converged in {iterations} iterations.")
    
    # Set values in env for visualization
    env.set_state_values(V)
    
    # Run Agent
    print("Start Policy Iteration Agent...")
    
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
    run_policy_iteration_agent()
