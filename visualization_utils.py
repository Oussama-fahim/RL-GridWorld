"""
Enhanced visualization utilities for RL GridWorld
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List, Tuple, Optional
import os


class GridWorldVisualizer:
    """Enhanced visualizer for GridWorld environment"""
    
    def __init__(self, size: int, figsize: Tuple[int, int] = (12, 10)):
        self.size = size
        self.figsize = figsize
        self.fig = None
        self.ax = None
        
    def visualize_policy(self, policy: np.ndarray, env_config: dict, 
                        save_path: str = None):
        """Visualize policy with arrows showing action directions
        
        Args:
            policy: Policy array (state -> action)
            env_config: Dictionary with start, goal, obstacles
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw grid
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5)
        
        # Draw obstacles
        for obs in env_config.get('obstacles', []):
            rect = patches.Rectangle(
                (obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                linewidth=2, edgecolor='red', facecolor='black'
            )
            ax.add_patch(rect)
        
        # Draw goal
        goal = env_config['goal']
        goal_rect = patches.Rectangle(
            (goal[1] - 0.5, goal[0] - 0.5), 1, 1,
            linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7
        )
        ax.add_patch(goal_rect)
        ax.text(goal[1], goal[0], 'G', ha='center', va='center',
               fontsize=20, fontweight='bold', color='darkgreen')
        
        # Draw start
        start = env_config['start']
        start_rect = patches.Rectangle(
            (start[1] - 0.5, start[0] - 0.5), 1, 1,
            linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5
        )
        ax.add_patch(start_rect)
        ax.text(start[1], start[0], 'S', ha='center', va='center',
               fontsize=20, fontweight='bold', color='darkblue')
        
        # Draw policy arrows
        arrow_map = {
            0: (0, -0.3),   # Up
            1: (0.3, 0),    # Right
            2: (0, 0.3),    # Down
            3: (-0.3, 0)    # Left
        }
        
        for r in range(self.size):
            for c in range(self.size):
                pos = (r, c)
                if pos in env_config.get('obstacles', []) or pos == goal:
                    continue
                
                state = r * self.size + c
                action = policy[state]
                
                dx, dy = arrow_map[action]
                ax.arrow(c, r, dx, dy, head_width=0.2, head_length=0.15,
                        fc='blue', ec='blue', alpha=0.6)
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(self.size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title('Policy Visualization', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Policy visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_value_function(self, V: np.ndarray, env_config: dict,
                                 save_path: str = None):
        """Visualize state value function as heatmap
        
        Args:
            V: Value function array
            env_config: Dictionary with start, goal, obstacles
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Reshape V to grid
        V_grid = V.reshape(self.size, self.size)
        
        # Create custom colormap
        colors = ['#d62728', '#ff7f0e', '#ffff00', '#2ca02c']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('value_cmap', colors, N=n_bins)
        
        # Plot heatmap
        im = ax.imshow(V_grid, cmap=cmap, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('State Value', fontsize=12, fontweight='bold')
        
        # Add grid
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1, alpha=0.5)
            ax.axvline(i - 0.5, color='white', linewidth=1, alpha=0.5)
        
        # Add value text
        for r in range(self.size):
            for c in range(self.size):
                pos = (r, c)
                if pos not in env_config.get('obstacles', []):
                    value = V_grid[r, c]
                    color = 'white' if value < V_grid.max() * 0.5 else 'black'
                    ax.text(c, r, f'{value:.1f}', ha='center', va='center',
                           color=color, fontsize=10, fontweight='bold')
        
        # Mark special positions
        goal = env_config['goal']
        ax.plot(goal[1], goal[0], 'g*', markersize=30, 
               markeredgecolor='white', markeredgewidth=2)
        
        start = env_config['start']
        ax.plot(start[1], start[0], 'b^', markersize=20,
               markeredgecolor='white', markeredgewidth=2)
        
        # Mark obstacles
        for obs in env_config.get('obstacles', []):
            ax.plot(obs[1], obs[0], 'rX', markersize=20,
                   markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(self.size - 0.5, -0.5)
        ax.set_title('Value Function Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Value function visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_q_function(self, Q: np.ndarray, env_config: dict,
                            save_path: str = None):
        """Visualize Q-function for all state-action pairs
        
        Args:
            Q: Q-function array (states x actions)
            env_config: Dictionary with start, goal, obstacles
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        action_names = ['Up', 'Right', 'Down', 'Left']
        
        for action in range(4):
            ax = axes[action // 2, action % 2]
            
            # Get Q-values for this action
            Q_action = Q[:, action].reshape(self.size, self.size)
            
            # Plot heatmap
            im = ax.imshow(Q_action, cmap='RdYlGn', interpolation='nearest')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Q-value', fontsize=10)
            
            # Add grid
            for i in range(self.size + 1):
                ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
                ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
            
            # Add values
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) not in env_config.get('obstacles', []):
                        value = Q_action[r, c]
                        ax.text(c, r, f'{value:.1f}', ha='center', va='center',
                               color='black', fontsize=8)
            
            # Mark goal and start
            goal = env_config['goal']
            ax.plot(goal[1], goal[0], 'g*', markersize=15)
            
            start = env_config['start']
            ax.plot(start[1], start[0], 'b^', markersize=12)
            
            ax.set_title(f'Q-values for Action: {action_names[action]}',
                        fontsize=12, fontweight='bold')
            ax.set_xlim(-0.5, self.size - 0.5)
            ax.set_ylim(self.size - 0.5, -0.5)
        
        plt.suptitle('Q-Function Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Q-function visualization saved to {save_path}")
        
        plt.show()
    
    def create_trajectory_animation(self, trajectory: List[Tuple[int, int]],
                                   env_config: dict, save_path: str = None,
                                   interval: int = 500):
        """Create animation of agent trajectory
        
        Args:
            trajectory: List of (row, col) positions
            env_config: Dictionary with start, goal, obstacles
            save_path: Path to save animation (as gif)
            interval: Milliseconds between frames
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def init():
            ax.clear()
            ax.set_xlim(-0.5, self.size - 0.5)
            ax.set_ylim(self.size - 0.5, -0.5)
            
            # Draw grid
            for i in range(self.size + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.5)
                ax.axvline(i - 0.5, color='gray', linewidth=0.5)
            
            # Draw obstacles
            for obs in env_config.get('obstacles', []):
                rect = patches.Rectangle(
                    (obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                    facecolor='black', edgecolor='red', linewidth=2
                )
                ax.add_patch(rect)
            
            # Draw goal
            goal = env_config['goal']
            goal_rect = patches.Rectangle(
                (goal[1] - 0.5, goal[0] - 0.5), 1, 1,
                facecolor='lightgreen', edgecolor='green', linewidth=2
            )
            ax.add_patch(goal_rect)
            ax.text(goal[1], goal[0], 'G', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='darkgreen')
            
            ax.set_aspect('equal')
            ax.axis('off')
            return []
        
        def animate(frame):
            if frame > 0:
                # Draw path
                path = trajectory[:frame+1]
                rows, cols = zip(*path)
                ax.plot(cols, rows, 'b-', linewidth=2, alpha=0.5)
                ax.plot(cols, rows, 'bo', markersize=8, alpha=0.3)
            
            # Draw current agent position
            pos = trajectory[frame]
            agent = patches.Circle((pos[1], pos[0]), 0.3, color='blue', zorder=10)
            ax.add_patch(agent)
            
            ax.set_title(f'Step {frame + 1}/{len(trajectory)}',
                        fontsize=14, fontweight='bold')
            
            return [agent]
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(trajectory), interval=interval,
                                      blit=False, repeat=True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Animation saved to {save_path}")
        
        plt.show()


def plot_convergence(iterations: List[int], deltas: List[float],
                    algorithm_name: str = "Algorithm", save_path: str = None):
    """Plot convergence curve
    
    Args:
        iterations: List of iteration numbers
        deltas: List of maximum value changes
        algorithm_name: Name of algorithm
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, deltas, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Value Change (Delta)', fontsize=12, fontweight='bold')
    ax.set_title(f'{algorithm_name} Convergence', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    size = 5
    visualizer = GridWorldVisualizer(size)
    
    # Example configuration
    env_config = {
        'start': (0, 0),
        'goal': (4, 4),
        'obstacles': [(1, 1), (1, 2), (2, 2), (3, 1)]
    }
    
    # Create dummy data
    policy = np.random.randint(0, 4, size*size)
    V = np.random.rand(size*size) * 10
    
    print("Creating visualizations...")
    visualizer.visualize_policy(policy, env_config, 
                                save_path="results/policy_visualization.png")
    visualizer.visualize_value_function(V, env_config,
                                       save_path="results/value_function.png")