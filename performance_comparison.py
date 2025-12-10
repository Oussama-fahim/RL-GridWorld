"""
Performance comparison utilities for RL algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple, Callable
import pandas as pd
from grid_env import GridWorldEnv
from value_agent import value_iteration
from policy_iteration import policy_iteration
from tqdm import tqdm
import json
import os


class PerformanceMetrics:
    """Class to track and compute performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episodes = []
        self.steps_per_episode = []
        self.rewards_per_episode = []
        self.success_per_episode = []
        self.training_time = 0
        self.convergence_iterations = 0
        
    def add_episode(self, steps: int, total_reward: float, success: bool):
        """Add episode results"""
        self.episodes.append(len(self.episodes) + 1)
        self.steps_per_episode.append(steps)
        self.rewards_per_episode.append(total_reward)
        self.success_per_episode.append(1 if success else 0)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_episodes': len(self.episodes),
            'success_rate': np.mean(self.success_per_episode) if self.success_per_episode else 0,
            'avg_steps': np.mean(self.steps_per_episode) if self.steps_per_episode else 0,
            'avg_reward': np.mean(self.rewards_per_episode) if self.rewards_per_episode else 0,
            'training_time': self.training_time,
            'convergence_iterations': self.convergence_iterations
        }
    
    def plot_learning_curves(self, window_size: int = 100, save_path: str = None):
        """Plot learning curves"""
        if not self.episodes:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Curves', fontsize=16, fontweight='bold')
        
        # Moving average helper
        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Steps per episode
        axes[0, 0].plot(self.episodes, self.steps_per_episode, alpha=0.3, label='Raw')
        if len(self.steps_per_episode) >= window_size:
            ma = moving_average(self.steps_per_episode, window_size)
            axes[0, 0].plot(range(window_size-1, len(self.steps_per_episode)), ma, 
                           linewidth=2, label=f'{window_size}-episode MA')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Steps')
        axes[0, 0].set_title('Steps per Episode')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rewards per episode
        axes[0, 1].plot(self.episodes, self.rewards_per_episode, alpha=0.3, label='Raw')
        if len(self.rewards_per_episode) >= window_size:
            ma = moving_average(self.rewards_per_episode, window_size)
            axes[0, 1].plot(range(window_size-1, len(self.rewards_per_episode)), ma,
                           linewidth=2, label=f'{window_size}-episode MA')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].set_title('Reward per Episode')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rate
        if len(self.success_per_episode) >= window_size:
            success_ma = moving_average(self.success_per_episode, window_size)
            axes[1, 0].plot(range(window_size-1, len(self.success_per_episode)), 
                           success_ma, linewidth=2)
            axes[1, 0].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title(f'Success Rate ({window_size}-episode MA)')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        summary = self.get_summary()
        summary_text = f"""
        Total Episodes: {summary['total_episodes']}
        Success Rate: {summary['success_rate']:.2%}
        Avg Steps: {summary['avg_steps']:.1f}
        Avg Reward: {summary['avg_reward']:.2f}
        Training Time: {summary['training_time']:.2f}s
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def evaluate_policy(env: GridWorldEnv, policy: np.ndarray, 
                   num_episodes: int = 100, max_steps: int = 100,
                   verbose: bool = False) -> PerformanceMetrics:
    """Evaluate a policy over multiple episodes
    
    Args:
        env: GridWorld environment
        policy: Policy to evaluate (state -> action mapping)
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        verbose: Whether to show progress bar
        
    Returns:
        PerformanceMetrics object with results
    """
    metrics = PerformanceMetrics()
    
    iterator = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)
    
    for episode in iterator:
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        
        for step in range(max_steps):
            action = policy[obs]
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        metrics.add_episode(steps, total_reward, terminated)
    
    return metrics


def compare_algorithms(env_config: Dict, num_trials: int = 5) -> pd.DataFrame:
    """Compare all algorithms on the same environment
    
    Args:
        env_config: Dictionary with environment configuration
        num_trials: Number of trials for each algorithm
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    print("=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        print("-" * 60)
        
        # Create environment
        env = GridWorldEnv(**env_config, render_mode=None)
        
        # Value Iteration
        print("\n1. Value Iteration...")
        start_time = time.time()
        V_vi, policy_vi, iters_vi = value_iteration(env)
        training_time_vi = time.time() - start_time
        
        metrics_vi = evaluate_policy(env, policy_vi, num_episodes=100)
        metrics_vi.training_time = training_time_vi
        metrics_vi.convergence_iterations = iters_vi
        
        summary_vi = metrics_vi.get_summary()
        summary_vi['algorithm'] = 'Value Iteration'
        summary_vi['trial'] = trial + 1
        results.append(summary_vi)
        
        print(f"   Converged in {iters_vi} iterations ({training_time_vi:.3f}s)")
        print(f"   Success Rate: {summary_vi['success_rate']:.2%}")
        
        # Policy Iteration
        print("\n2. Policy Iteration...")
        start_time = time.time()
        V_pi, policy_pi, iters_pi = policy_iteration(env)
        training_time_pi = time.time() - start_time
        
        metrics_pi = evaluate_policy(env, policy_pi, num_episodes=100)
        metrics_pi.training_time = training_time_pi
        metrics_pi.convergence_iterations = iters_pi
        
        summary_pi = metrics_pi.get_summary()
        summary_pi['algorithm'] = 'Policy Iteration'
        summary_pi['trial'] = trial + 1
        results.append(summary_pi)
        
        print(f"   Converged in {iters_pi} iterations ({training_time_pi:.3f}s)")
        print(f"   Success Rate: {summary_pi['success_rate']:.2%}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print("\nMean values across trials:")
    print(df.groupby('algorithm').mean().round(3))
    
    print("\nStandard deviation:")
    print(df.groupby('algorithm').std().round(3))
    
    return df


def plot_comparison(df: pd.DataFrame, save_path: str = None):
    """Plot comparison between algorithms
    
    Args:
        df: DataFrame from compare_algorithms
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['training_time', 'convergence_iterations', 'success_rate', 'avg_steps']
    titles = ['Training Time (s)', 'Convergence Iterations', 'Success Rate', 'Average Steps']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Create bar plot
        sns.barplot(data=df, x='algorithm', y=metric, ax=ax, 
                   palette='viridis', errorbar='sd')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def save_results(df: pd.DataFrame, filename: str = "results/comparison_results.csv"):
    """Save comparison results to CSV"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Example usage
    env_config = {
        'size': 7,
        'start': (0, 0),
        'goal': (6, 6),
        'obstacles': [(1, 1), (1, 2), (2, 2), (3, 1), (4, 4)]
    }
    
    print("Running comprehensive algorithm comparison...")
    df = compare_algorithms(env_config, num_trials=3)
    
    # Save results
    save_results(df)
    
    # Plot comparison
    plot_comparison(df, save_path="results/algorithm_comparison.png")