"""
Configuration management for RL GridWorld Environment
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import os


@dataclass
class EnvironmentConfig:
    """Configuration for GridWorld environment"""
    size: int = 5
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (4, 4)
    obstacles: List[Tuple[int, int]] = None
    render_mode: Optional[str] = "human"
    max_steps: int = 100
    
    def __post_init__(self):
        if self.obstacles is None:
            self.obstacles = [(1, 1), (1, 2), (2, 2), (3, 1)]
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        data = {
            'size': self.size,
            'start': self.start,
            'goal': self.goal,
            'obstacles': self.obstacles,
            'render_mode': self.render_mode,
            'max_steps': self.max_steps
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)


@dataclass
class ValueIterationConfig:
    """Configuration for Value Iteration algorithm"""
    gamma: float = 0.9
    theta: float = 1e-6
    max_iterations: int = 1000


@dataclass
class PolicyIterationConfig:
    """Configuration for Policy Iteration algorithm"""
    gamma: float = 0.9
    theta: float = 1e-6
    max_iterations: int = 1000


@dataclass
class QLearningConfig:
    """Configuration for Q-Learning algorithm"""
    episodes: int = 15000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    max_steps_per_episode: int = 200


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 100
    animation_speed: float = 0.2
    save_frames: bool = False
    output_dir: str = "results"
    show_values: bool = True
    value_fontsize: int = 8
    title_fontsize: int = 14


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    env: EnvironmentConfig = None
    algorithm: str = "value_iteration"  # value_iteration, policy_iteration, q_learning
    value_iteration: ValueIterationConfig = None
    policy_iteration: PolicyIterationConfig = None
    q_learning: QLearningConfig = None
    visualization: VisualizationConfig = None
    
    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.value_iteration is None:
            self.value_iteration = ValueIterationConfig()
        if self.policy_iteration is None:
            self.policy_iteration = PolicyIterationConfig()
        if self.q_learning is None:
            self.q_learning = QLearningConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()


# Predefined configurations for common scenarios
SMALL_GRID_CONFIG = EnvironmentConfig(
    size=5,
    start=(0, 0),
    goal=(4, 4),
    obstacles=[(1, 1), (1, 2), (2, 2)]
)

MEDIUM_GRID_CONFIG = EnvironmentConfig(
    size=7,
    start=(0, 0),
    goal=(6, 6),
    obstacles=[(1, 1), (1, 2), (2, 2), (3, 1), (4, 4), (5, 3)]
)

LARGE_GRID_CONFIG = EnvironmentConfig(
    size=10,
    start=(0, 0),
    goal=(9, 9),
    obstacles=[
        (1, 1), (1, 2), (1, 3), (1, 4),
        (3, 1), (3, 2), (3, 3),
        (5, 5), (5, 6), (5, 7),
        (7, 1), (7, 2), (8, 2)
    ]
)

MAZE_CONFIG = EnvironmentConfig(
    size=10,
    start=(0, 0),
    goal=(9, 9),
    obstacles=[
        # Vertical walls
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
        (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
        (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8)
    ]
)


def get_config(preset: str = "small") -> EnvironmentConfig:
    """Get a predefined configuration
    
    Args:
        preset: One of 'small', 'medium', 'large', 'maze'
    
    Returns:
        EnvironmentConfig instance
    """
    configs = {
        "small": SMALL_GRID_CONFIG,
        "medium": MEDIUM_GRID_CONFIG,
        "large": LARGE_GRID_CONFIG,
        "maze": MAZE_CONFIG
    }
    
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs.keys())}")
    
    return configs[preset]


# Ensure output directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['results', 'saved_models', 'logs', 'plots']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    # Example usage
    ensure_directories()
    
    # Create and save a configuration
    config = TrainingConfig(
        env=get_config("medium"),
        algorithm="q_learning"
    )
    
    print("Environment Configuration:")
    print(f"  Grid Size: {config.env.size}x{config.env.size}")
    print(f"  Start: {config.env.start}")
    print(f"  Goal: {config.env.goal}")
    print(f"  Obstacles: {len(config.env.obstacles)} obstacles")
    print(f"\nAlgorithm: {config.algorithm}")