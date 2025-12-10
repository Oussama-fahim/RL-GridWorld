"""
Advanced usage examples for RL GridWorld

This script demonstrates:
1. Using configuration system
2. Logging experiments
3. Saving and loading models
4. Creating visualizations
5. Comparing algorithms
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from grid_env import GridWorldEnv
from value_agent import value_iteration
from policy_iteration import policy_iteration
from config import get_config, TrainingConfig
from logger import ExperimentLogger, ModelSaver, create_experiment_report
from visualization_utils import GridWorldVisualizer
from performance_comparison import evaluate_policy, compare_algorithms


def example_1_basic_training_with_logging():
    """Example 1: Train agent with full logging"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Training with Logging")
    print("="*70 + "\n")
    
    # Setup
    logger = ExperimentLogger("value_iteration_basic")
    config = get_config("medium")
    
    # Log hyperparameters
    hyperparameters = {
        'algorithm': 'Value Iteration',
        'gamma': 0.9,
        'theta': 1e-6,
        'grid_size': config.size
    }
    logger.log_hyperparameters(hyperparameters)
    
    # Create environment
    env = GridWorldEnv(
        size=config.size,
        start=config.start,
        goal=config.goal,
        obstacles=config.obstacles,
        render_mode=None
    )
    
    # Train
    print("Training Value Iteration...")
    start_time = time.time()
    V, policy, iterations = value_iteration(env)
    training_time = time.time() - start_time
    
    logger.log_training_time(training_time)
    logger.log_convergence(iterations, 1e-6)
    
    # Evaluate
    print("Evaluating policy...")
    metrics = evaluate_policy(env, policy, num_episodes=100, verbose=True)
    
    for episode in range(len(metrics.episodes)):
        logger.log_episode(
            metrics.episodes[episode],
            metrics.steps_per_episode[episode],
            metrics.rewards_per_episode[episode],
            metrics.success_per_episode[episode] == 1
        )
    
    # Save results
    logger.save_metrics()
    logger.print_summary()
    create_experiment_report(logger)
    
    # Save model
    saver = ModelSaver()
    env_config = {
        'size': config.size,
        'start': config.start,
        'goal': config.goal,
        'obstacles': config.obstacles
    }
    saver.save_value_function(V, "value_iteration_medium", env_config)
    saver.save_policy(policy, "value_iteration_policy_medium", env_config)
    
    print("\n✓ Example 1 completed!")


def example_2_visualization_pipeline():
    """Example 2: Complete visualization pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Visualization Pipeline")
    print("="*70 + "\n")
    
    # Setup
    config = get_config("small")
    env = GridWorldEnv(
        size=config.size,
        start=config.start,
        goal=config.goal,
        obstacles=config.obstacles,
        render_mode=None
    )
    
    # Train
    print("Training...")
    V, policy, iterations = value_iteration(env)
    
    # Create visualizer
    visualizer = GridWorldVisualizer(config.size)
    
    env_config = {
        'start': config.start,
        'goal': config.goal,
        'obstacles': config.obstacles
    }
    
    # Create all visualizations
    print("\nCreating visualizations...")
    
    os.makedirs("results/example2", exist_ok=True)
    
    visualizer.visualize_policy(
        policy, env_config,
        save_path="results/example2/policy.png"
    )
    
    visualizer.visualize_value_function(
        V, env_config,
        save_path="results/example2/value_function.png"
    )
    
    # Generate trajectory
    print("\nGenerating optimal trajectory...")
    trajectory = []
    obs, _ = env.reset()
    trajectory.append(env.agent_pos)
    
    for _ in range(20):
        action = policy[obs]
        obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(env.agent_pos)
        
        if terminated:
            break
    
    print(f"Trajectory length: {len(trajectory)} steps")
    print(f"Path: {' -> '.join([str(pos) for pos in trajectory])}")
    
    visualizer.create_trajectory_animation(
        trajectory, env_config,
        save_path="results/example2/trajectory.gif"
    )
    
    print("\n✓ Example 2 completed! Check results/example2/ folder")


def example_3_algorithm_comparison():
    """Example 3: Compare multiple algorithms"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Algorithm Comparison")
    print("="*70 + "\n")
    
    # Setup environment
    env_config = {
        'size': 5,
        'start': (0, 0),
        'goal': (4, 4),
        'obstacles': [(1, 1), (1, 2), (2, 2)]
    }
    
    # Run comparison
    print("Running comparison across 3 trials...")
    df = compare_algorithms(env_config, num_trials=3)
    
    # Save results
    from performance_comparison import save_results, plot_comparison
    
    os.makedirs("results/example3", exist_ok=True)
    save_results(df, "results/example3/comparison_results.csv")
    plot_comparison(df, save_path="results/example3/comparison.png")
    
    print("\n✓ Example 3 completed! Check results/example3/ folder")


def example_4_custom_environment():
    """Example 4: Create and solve custom environment"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Environment")
    print("="*70 + "\n")
    
    # Create a maze-like environment
    size = 8
    obstacles = [
        # Vertical walls
        (2, 1), (2, 2), (2, 3),
        (4, 4), (4, 5), (4, 6),
        (6, 1), (6, 2), (6, 3), (6, 4)
    ]
    
    env = GridWorldEnv(
        size=size,
        start=(0, 0),
        goal=(7, 7),
        obstacles=obstacles,
        render_mode=None
    )
    
    print("Environment created:")
    print(f"  Grid size: {size}x{size}")
    print(f"  Start: (0, 0)")
    print(f"  Goal: (7, 7)")
    print(f"  Obstacles: {len(obstacles)}")
    
    # Train both algorithms
    print("\nTraining Value Iteration...")
    start_time = time.time()
    V_vi, policy_vi, iters_vi = value_iteration(env)
    time_vi = time.time() - start_time
    
    print("\nTraining Policy Iteration...")
    start_time = time.time()
    V_pi, policy_pi, iters_pi = policy_iteration(env)
    time_pi = time.time() - start_time
    
    # Compare
    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    print(f"Value Iteration:  {iters_vi} iterations in {time_vi:.3f}s")
    print(f"Policy Iteration: {iters_pi} iterations in {time_pi:.3f}s")
    
    # Check if policies are the same
    policies_match = np.array_equal(policy_vi, policy_pi)
    print(f"\nPolicies match: {policies_match}")
    
    if not policies_match:
        diff_count = np.sum(policy_vi != policy_pi)
        print(f"Number of different actions: {diff_count}/{len(policy_vi)}")
    
    # Visualize
    print("\nCreating visualizations...")
    os.makedirs("results/example4", exist_ok=True)
    
    visualizer = GridWorldVisualizer(size)
    env_config = {
        'start': (0, 0),
        'goal': (7, 7),
        'obstacles': obstacles
    }
    
    visualizer.visualize_policy(
        policy_vi, env_config,
        save_path="results/example4/custom_maze_policy.png"
    )
    
    visualizer.visualize_value_function(
        V_vi, env_config,
        save_path="results/example4/custom_maze_values.png"
    )
    
    print("\n✓ Example 4 completed! Check results/example4/ folder")


def example_5_load_and_evaluate():
    """Example 5: Load saved model and evaluate"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Load and Evaluate Saved Model")
    print("="*70 + "\n")
    
    saver = ModelSaver()
    
    # List saved models
    models = saver.list_saved_models()
    
    if not models:
        print("No saved models found. Run example 1 first!")
        return
    
    print(f"Found {len(models)} saved model(s)")
    print("\nMost recent model:")
    print(f"  {models[0].name}")
    
    # Load most recent model
    print("\nLoading model...")
    data = saver.load_model(str(models[0]))
    
    if 'policy' in data['model']:
        policy = data['model']['policy']
        env_config = data['model']['env_config']
        
        print(f"\nPolicy shape: {policy.shape}")
        print(f"Environment: {env_config['size']}x{env_config['size']} grid")
        
        # Create environment
        env = GridWorldEnv(**env_config, render_mode=None)
        
        # Evaluate
        print("\nEvaluating loaded policy...")
        metrics = evaluate_policy(env, policy, num_episodes=100, verbose=True)
        
        summary = metrics.get_summary()
        print("\nEvaluation Results:")
        print(f"  Success Rate: {summary['success_rate']:.2%}")
        print(f"  Average Steps: {summary['avg_steps']:.2f}")
        print(f"  Average Reward: {summary['avg_reward']:.2f}")
    
    print("\n✓ Example 5 completed!")


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_training_with_logging,
        example_2_visualization_pipeline,
        example_3_algorithm_comparison,
        example_4_custom_environment,
        example_5_load_and_evaluate
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\n❌ Example {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "RL GRIDWORLD - ADVANCED EXAMPLES")
    print("="*70)
    
    print("\nAvailable examples:")
    print("  1. Basic training with logging")
    print("  2. Visualization pipeline")
    print("  3. Algorithm comparison")
    print("  4. Custom environment")
    print("  5. Load and evaluate saved model")
    print("  6. Run all examples")
    
    choice = input("\nSelect example (1-6): ").strip()
    
    examples = {
        '1': example_1_basic_training_with_logging,
        '2': example_2_visualization_pipeline,
        '3': example_3_algorithm_comparison,
        '4': example_4_custom_environment,
        '5': example_5_load_and_evaluate,
        '6': run_all_examples
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("Invalid choice!")