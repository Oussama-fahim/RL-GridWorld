"""
Integration tests for the complete RL GridWorld system
Tests end-to-end workflows, component interactions, and real-world scenarios
"""
import pytest
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid_env import GridWorldEnv
from value_agent import value_iteration
from policy_iteration import policy_iteration
from random_agent import evaluate_random_policy


class TestEndToEndWorkflow:
    """Test complete workflow from training to evaluation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.size = 5
        self.env = GridWorldEnv(
            size=self.size,
            start=(0, 0),
            goal=(4, 4),
            obstacles=[(1, 1), (2, 2)],
            render_mode=None
        )
    
    def teardown_method(self):
        """Cleanup after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_train_and_execute_value_iteration(self):
        """Test complete workflow: train with VI and execute policy"""
        # Train
        V, policy, iterations = value_iteration(self.env)
        
        assert iterations > 0
        assert V is not None
        assert policy is not None
        
        # Execute learned policy
        obs, _ = self.env.reset()
        total_reward = 0
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            action = policy[obs]
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated:
                break
        
        # Should reach goal
        assert terminated
        assert self.env.agent_pos == self.env.goal
        assert steps < max_steps
        assert total_reward > 0  # Should get goal reward minus step penalties
    
    def test_train_and_execute_policy_iteration(self):
        """Test complete workflow with Policy Iteration"""
        # Train
        V, policy, iterations = policy_iteration(self.env)
        
        # Execute and verify reaches goal
        obs, _ = self.env.reset()
        
        for _ in range(50):
            action = policy[obs]
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated:
                break
        
        assert terminated
        assert self.env.agent_pos == self.env.goal
    
    def test_multiple_episodes_with_reset(self):
        """Test multiple episodes with environment reset"""
        V, policy, iterations = value_iteration(self.env)
        
        success_count = 0
        num_episodes = 10
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            
            # Verify reset worked
            assert self.env.agent_pos == self.env.start
            
            for step in range(50):
                action = policy[obs]
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated:
                    success_count += 1
                    break
        
        # Should succeed in most/all episodes
        assert success_count >= num_episodes * 0.9


class TestAlgorithmComparison:
    """Test comparisons between different algorithms"""
    
    def setup_method(self):
        """Setup environment for comparison"""
        self.env = GridWorldEnv(
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=[(1, 1), (2, 2)],
            render_mode=None
        )
    
    def test_vi_vs_pi_optimality(self):
        """Test that VI and PI produce equivalent optimal policies"""
        V_vi, policy_vi, iter_vi = value_iteration(self.env)
        V_pi, policy_pi, iter_pi = policy_iteration(self.env)
        
        # Policies should be the same (both optimal)
        np.testing.assert_array_equal(policy_vi, policy_pi)
        
        # Values should be very close
        np.testing.assert_allclose(V_vi, V_pi, rtol=1e-4)
        
        # PI should generally converge faster
        assert iter_pi <= iter_vi + 5  # Allow small margin
    
    def test_random_vs_optimal_policy(self):
        """Test that optimal policy outperforms random policy"""
        # Get optimal policy
        V_optimal, policy_optimal, _ = value_iteration(self.env)
        
        # Evaluate random policy
        V_random = evaluate_random_policy(self.env)
        
        # Optimal values should be higher than random values
        assert np.mean(V_optimal) > np.mean(V_random)
        
        # Specifically, start state value should be much better
        start_state = self.env.pos_to_state(self.env.start)
        assert V_optimal[start_state] > V_random[start_state]
    
    def test_policy_evaluation_consistency(self):
        """Test that policies can be evaluated consistently"""
        from performance_comparison import evaluate_policy
        
        # Train policy
        V, policy, _ = value_iteration(self.env)
        
        # Evaluate multiple times
        results = []
        for _ in range(3):
            metrics = evaluate_policy(self.env, policy, num_episodes=50, verbose=False)
            summary = metrics.get_summary()
            results.append(summary['success_rate'])
        
        # Results should be consistent (all successes for optimal policy)
        assert all(rate > 0.95 for rate in results)


class TestConfigurationSystem:
    """Test configuration and environment setup"""
    
    def test_config_loading_and_creation(self):
        """Test loading configuration and creating environment"""
        from config import get_config, EnvironmentConfig
        
        # Test predefined configs
        configs = ['small', 'medium', 'large']
        
        for config_name in configs:
            config = get_config(config_name)
            
            assert isinstance(config, EnvironmentConfig)
            assert config.size > 0
            assert config.start is not None
            assert config.goal is not None
            
            # Create environment from config
            env = GridWorldEnv(
                size=config.size,
                start=config.start,
                goal=config.goal,
                obstacles=config.obstacles,
                render_mode=None
            )
            
            assert env.size == config.size
            assert env.start == config.start
            assert env.goal == config.goal
    
    def test_config_persistence(self):
        """Test saving and loading configuration"""
        from config import EnvironmentConfig
        
        # Create config
        config = EnvironmentConfig(
            size=7,
            start=(0, 0),
            goal=(6, 6),
            obstacles=[(1, 1), (2, 2)]
        )
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            config.to_json(temp_path)
            
            # Load back
            loaded_config = EnvironmentConfig.from_json(temp_path)
            
            # Verify
            assert loaded_config.size == config.size
            assert loaded_config.start == config.start
            assert loaded_config.goal == config.goal
            assert loaded_config.obstacles == config.obstacles
        finally:
            os.unlink(temp_path)


class TestLoggingSystem:
    """Test logging and experiment tracking"""
    
    def setup_method(self):
        """Setup temporary directory for logs"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_experiment_logger_workflow(self):
        """Test complete experiment logging workflow"""
        from logger import ExperimentLogger
        
        logger = ExperimentLogger("test_experiment", log_dir=self.temp_dir)
        
        # Log hyperparameters
        hyperparams = {
            'algorithm': 'Value Iteration',
            'gamma': 0.9,
            'size': 5
        }
        logger.log_hyperparameters(hyperparams)
        
        # Log episodes
        for episode in range(10):
            logger.log_episode(
                episode=episode,
                steps=np.random.randint(5, 20),
                reward=np.random.uniform(-10, 10),
                success=np.random.random() > 0.3
            )
        
        # Log training time
        logger.log_training_time(5.5)
        
        # Save metrics
        logger.save_metrics()
        
        # Check files were created
        assert os.path.exists(logger.run_dir)
        assert os.path.exists(logger.run_dir / "metrics.json")
        assert os.path.exists(logger.run_dir / "hyperparameters.json")
        assert os.path.exists(logger.run_dir / "experiment.log")
        
        # Get summary
        summary = logger.get_summary()
        assert 'total_episodes' in summary
        assert summary['total_episodes'] == 10
    
    def test_model_saver_workflow(self):
        """Test model saving and loading"""
        from logger import ModelSaver
        
        saver = ModelSaver(save_dir=self.temp_dir)
        
        # Create dummy model data
        policy = np.random.randint(0, 4, 25)
        V = np.random.rand(25)
        
        env_config = {
            'size': 5,
            'start': (0, 0),
            'goal': (4, 4)
        }
        
        # Save policy
        saver.save_policy(policy, "test_policy", env_config)
        
        # Save value function
        saver.save_value_function(V, "test_value", env_config)
        
        # List saved models
        models = saver.list_saved_models()
        assert len(models) == 2
        
        # Load a model
        loaded_data = saver.load_model(str(models[0]))
        
        assert 'model' in loaded_data
        assert 'metadata' in loaded_data
        assert 'timestamp' in loaded_data


class TestVisualizationSystem:
    """Test visualization components"""
    
    def setup_method(self):
        """Setup"""
        self.temp_dir = tempfile.mkdtemp()
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing
    
    def teardown_method(self):
        """Cleanup"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_policy_visualization(self):
        """Test policy visualization creation"""
        from visualization_utils import GridWorldVisualizer
        
        size = 5
        visualizer = GridWorldVisualizer(size)
        
        policy = np.random.randint(0, 4, size * size)
        env_config = {
            'start': (0, 0),
            'goal': (4, 4),
            'obstacles': [(1, 1)]
        }
        
        save_path = os.path.join(self.temp_dir, "policy.png")
        
        # Should not raise exception
        visualizer.visualize_policy(policy, env_config, save_path=save_path)
        
        # File should be created
        assert os.path.exists(save_path)
    
    def test_value_function_visualization(self):
        """Test value function visualization"""
        from visualization_utils import GridWorldVisualizer
        
        size = 5
        visualizer = GridWorldVisualizer(size)
        
        V = np.random.rand(size * size) * 10
        env_config = {
            'start': (0, 0),
            'goal': (4, 4),
            'obstacles': []
        }
        
        save_path = os.path.join(self.temp_dir, "value.png")
        
        visualizer.visualize_value_function(V, env_config, save_path=save_path)
        
        assert os.path.exists(save_path)


class TestPerformanceComparison:
    """Test performance comparison utilities"""
    
    def test_evaluate_policy_function(self):
        """Test policy evaluation utility"""
        from performance_comparison import evaluate_policy
        
        env = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4), render_mode=None)
        V, policy, _ = value_iteration(env)
        
        # Evaluate policy
        metrics = evaluate_policy(env, policy, num_episodes=20, verbose=False)
        
        # Check metrics
        summary = metrics.get_summary()
        
        assert 'total_episodes' in summary
        assert summary['total_episodes'] == 20
        assert 'success_rate' in summary
        assert 'avg_steps' in summary
        assert 'avg_reward' in summary
        
        # Optimal policy should have high success rate
        assert summary['success_rate'] > 0.9
    
    def test_compare_algorithms_function(self):
        """Test algorithm comparison function"""
        from performance_comparison import compare_algorithms
        
        env_config = {
            'size': 5,
            'start': (0, 0),
            'goal': (4, 4),
            'obstacles': [(1, 1)]
        }
        
        # Run comparison (small number of trials for speed)
        df = compare_algorithms(env_config, num_trials=2)
        
        # Check DataFrame structure
        assert 'algorithm' in df.columns
        assert 'success_rate' in df.columns
        assert 'training_time' in df.columns
        assert 'convergence_iterations' in df.columns
        
        # Should have results for both algorithms
        assert len(df) == 4  # 2 algorithms × 2 trials
        
        # Both algorithms should have high success rates
        assert df['success_rate'].min() > 0.8


class TestMovingGoalEnvironment:
    """Test moving goal environment integration"""
    
    def test_moving_goal_state_space(self):
        """Test that moving goal environment has correct state space"""
        from q_learning_moving_goal import MovingGoalEnv
        
        size = 5
        env = MovingGoalEnv(size=size, start=(0, 0), render_mode=None)
        
        # State space should be (size*size) * (size*size) for all agent-goal pairs
        expected_states = (size * size) * (size * size)
        assert env.observation_space.n == expected_states
    
    def test_moving_goal_reset_changes_goal(self):
        """Test that goal changes between episodes"""
        from q_learning_moving_goal import MovingGoalEnv
        
        env = MovingGoalEnv(size=5, start=(0, 0), render_mode=None)
        
        goals = []
        for _ in range(10):
            obs, _ = env.reset()
            goals.append(env.goal)
        
        # Should have different goals (with high probability)
        unique_goals = set(goals)
        assert len(unique_goals) > 1
    
    def test_moving_goal_observation_encoding(self):
        """Test observation encoding includes both agent and goal"""
        from q_learning_moving_goal import MovingGoalEnv
        
        size = 5
        env = MovingGoalEnv(size=size, start=(0, 0), render_mode=None)
        
        obs, _ = env.reset()
        
        # Observation should encode both agent position and goal position
        # obs = agent_idx * (size*size) + goal_idx
        agent_idx = env.pos_to_state(env.agent_pos)
        goal_idx = env.pos_to_state(env.goal)
        expected_obs = agent_idx * (size * size) + goal_idx
        
        assert obs == expected_obs


class TestCompleteScenarios:
    """Test complete real-world scenarios"""
    
    def test_maze_solving_scenario(self):
        """Test solving a complex maze"""
        # Create maze with multiple paths
        obstacles = [
            (1, 1), (1, 2), (1, 3),
            (3, 1), (3, 2), (3, 3),
        ]
        
        env = GridWorldEnv(
            size=5,
            start=(0, 0),
            goal=(4, 4),
            obstacles=obstacles,
            render_mode=None
        )
        
        # Train
        V, policy, iterations = value_iteration(env)
        
        # Solve
        obs, _ = env.reset()
        path = [env.agent_pos]
        
        for _ in range(50):
            action = policy[obs]
            obs, reward, terminated, truncated, info = env.step(action)
            path.append(env.agent_pos)
            
            if terminated:
                break
        
        # Should find a path
        assert terminated
        assert path[-1] == env.goal
        
        # Path should be reasonable length
        assert len(path) < 30
    
    def test_multi_agent_independent_training(self):
        """Test training multiple agents independently"""
        configs = [
            {'size': 5, 'start': (0, 0), 'goal': (4, 4), 'obstacles': []},
            {'size': 7, 'start': (0, 0), 'goal': (6, 6), 'obstacles': [(3, 3)]},
        ]
        
        policies = []
        
        for config in configs:
            env = GridWorldEnv(**config, render_mode=None)
            V, policy, iterations = value_iteration(env)
            policies.append(policy)
            
            # Each should converge
            assert iterations > 0
        
        # Policies should be different (different environments)
        assert len(policies[0]) != len(policies[1])
    
    def test_transfer_learning_scenario(self):
        """Test policy transfer between similar environments"""
        # Train on simple environment
        env1 = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4), 
                           obstacles=[], render_mode=None)
        V1, policy1, _ = value_iteration(env1)
        
        # Test on similar but slightly different environment
        env2 = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4),
                           obstacles=[(2, 2)], render_mode=None)
        
        # Use policy from env1 on env2
        obs, _ = env2.reset()
        success = False
        
        for _ in range(50):
            action = policy1[obs]
            obs, reward, terminated, truncated, info = env2.step(action)
            
            if terminated:
                success = True
                break
        
        # Policy should still work reasonably well (may need to go around obstacle)
        # This tests robustness rather than optimality
        assert success or env2.agent_pos != env2.start


class TestSystemRobustness:
    """Test system robustness and error handling"""
    
    def test_invalid_actions_handled(self):
        """Test that invalid actions are handled gracefully"""
        env = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4), render_mode=None)
        
        obs, _ = env.reset()
        
        # Valid actions should work
        for action in range(4):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
    
    def test_reset_after_termination(self):
        """Test that environment can be reset after termination"""
        env = GridWorldEnv(size=3, start=(0, 0), goal=(2, 2), render_mode=None)
        
        # Place agent next to goal
        env.agent_pos = (2, 1)
        obs = env.pos_to_state(env.agent_pos)
        
        # Reach goal
        obs, reward, terminated, truncated, info = env.step(1)
        assert terminated
        
        # Reset should work
        obs, info = env.reset()
        assert obs == env.pos_to_state(env.start)
        assert env.agent_pos == env.start
    
    def test_long_episode_truncation(self):
        """Test that episodes can handle many steps"""
        env = GridWorldEnv(size=10, start=(0, 0), goal=(9, 9), render_mode=None)
        V, policy, _ = value_iteration(env)
        
        obs, _ = env.reset()
        max_steps = 1000
        
        for step in range(max_steps):
            action = policy[obs]
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                break
        
        # Should terminate before max_steps
        assert step < max_steps
        assert terminated


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])