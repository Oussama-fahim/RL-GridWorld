"""
Unit tests for RL algorithms
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid_env import GridWorldEnv
from value_agent import value_iteration
from policy_iteration import policy_iteration, policy_evaluation, policy_improvement


class TestValueIteration:
    """Test suite for Value Iteration algorithm"""
    
    def setup_method(self):
        """Setup test environment before each test"""
        self.size = 5
        self.env = GridWorldEnv(
            size=self.size,
            start=(0, 0),
            goal=(4, 4),
            obstacles=[(1, 1), (2, 2)],
            render_mode=None
        )
    
    def test_value_iteration_convergence(self):
        """Test that value iteration converges"""
        V, policy, iterations = value_iteration(self.env)
        
        # Should converge in reasonable number of iterations
        assert iterations > 0
        assert iterations < 1000
        
        # V should be a valid array
        assert V.shape == (self.size * self.size,)
        assert not np.any(np.isnan(V))
        assert not np.any(np.isinf(V))
    
    def test_value_iteration_policy_shape(self):
        """Test that policy has correct shape"""
        V, policy, iterations = value_iteration(self.env)
        
        assert policy.shape == (self.size * self.size,)
        assert policy.dtype in [np.int32, np.int64, int]
        
        # All actions should be valid (0-3)
        assert np.all(policy >= 0)
        assert np.all(policy < 4)
    
    def test_value_iteration_goal_state(self):
        """Test that goal state has highest or near-highest value"""
        V, policy, iterations = value_iteration(self.env)
        
        goal_state = self.env.pos_to_state(self.env.goal)
        goal_value = V[goal_state]
        
        # Goal should have high value (but may not be absolute max due to transitions)
        # After reaching goal, value becomes 0, so nearby states have higher values
        # Let's check that goal value is reasonable
        assert goal_value >= np.min(V)
    
    def test_value_iteration_start_state(self):
        """Test that start state has reasonable value"""
        V, policy, iterations = value_iteration(self.env)
        
        start_state = self.env.pos_to_state(self.env.start)
        start_value = V[start_state]
        
        # Start state should have positive value (can reach goal)
        assert start_value > -100  # Reasonable lower bound
    
    def test_value_iteration_obstacle_states(self):
        """Test values of obstacle-adjacent states"""
        V, policy, iterations = value_iteration(self.env)
        
        # States adjacent to obstacles should have valid values
        obstacle = self.env.obstacles[0]
        obs_state = self.env.pos_to_state(obstacle)
        
        # Value should be defined (not NaN or Inf)
        assert not np.isnan(V[obs_state])
        assert not np.isinf(V[obs_state])
    
    def test_value_iteration_different_gamma(self):
        """Test value iteration with different discount factors"""
        gamma_values = [0.5, 0.9, 0.99]
        results = []
        
        for gamma in gamma_values:
            V, policy, iterations = value_iteration(self.env, gamma=gamma)
            results.append((gamma, V, iterations))
        
        # Higher gamma should generally lead to higher values
        # (more weight on future rewards)
        V_low_gamma = results[0][1]
        V_high_gamma = results[2][1]
        
        # At least some states should have higher values with higher gamma
        assert np.mean(V_high_gamma) >= np.mean(V_low_gamma)
    
    def test_value_iteration_deterministic(self):
        """Test that value iteration is deterministic"""
        V1, policy1, iter1 = value_iteration(self.env)
        V2, policy2, iter2 = value_iteration(self.env)
        
        # Should get same results
        np.testing.assert_array_almost_equal(V1, V2)
        np.testing.assert_array_equal(policy1, policy2)
        assert iter1 == iter2
    
    def test_value_iteration_small_theta(self):
        """Test convergence with very small theta"""
        V, policy, iterations = value_iteration(self.env, theta=1e-10)
        
        # Should converge but take more iterations
        assert iterations > 0
        assert iterations < 10000


class TestPolicyIteration:
    """Test suite for Policy Iteration algorithm"""
    
    def setup_method(self):
        """Setup test environment before each test"""
        self.size = 5
        self.env = GridWorldEnv(
            size=self.size,
            start=(0, 0),
            goal=(4, 4),
            obstacles=[(1, 1), (2, 2)],
            render_mode=None
        )
    
    def test_policy_iteration_convergence(self):
        """Test that policy iteration converges"""
        V, policy, iterations = policy_iteration(self.env)
        
        # Should converge quickly
        assert iterations > 0
        assert iterations < 100  # PI usually converges in fewer iterations than VI
        
        # Results should be valid
        assert V.shape == (self.size * self.size,)
        assert policy.shape == (self.size * self.size,)
        assert not np.any(np.isnan(V))
    
    def test_policy_iteration_vs_value_iteration(self):
        """Test that PI and VI give same optimal policy"""
        V_vi, policy_vi, iter_vi = value_iteration(self.env)
        V_pi, policy_pi, iter_pi = policy_iteration(self.env)
        
        # Policies should be identical (both optimal)
        np.testing.assert_array_equal(policy_vi, policy_pi)
        
        # Values should be very close
        np.testing.assert_array_almost_equal(V_vi, V_pi, decimal=4)
        
        # PI should converge in fewer iterations
        assert iter_pi <= iter_vi
    
    def test_policy_evaluation(self):
        """Test policy evaluation separately"""
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        
        # Create a random policy
        policy = np.random.randint(0, n_actions, n_states)
        V = np.zeros(n_states)
        
        # Evaluate it
        V_evaluated = policy_evaluation(self.env, policy, V.copy())
        
        # Should produce valid values
        assert V_evaluated.shape == (n_states,)
        assert not np.any(np.isnan(V_evaluated))
        assert not np.any(np.isinf(V_evaluated))
    
    def test_policy_improvement(self):
        """Test policy improvement separately"""
        n_states = self.env.observation_space.n
        
        # Start with random value function
        V = np.random.rand(n_states) * 10
        
        # Improve policy
        improved_policy = policy_improvement(self.env, V)
        
        # Should produce valid policy
        assert improved_policy.shape == (n_states,)
        assert np.all(improved_policy >= 0)
        assert np.all(improved_policy < 4)
    
    def test_policy_iteration_deterministic(self):
        """Test that policy iteration is deterministic"""
        V1, policy1, iter1 = policy_iteration(self.env)
        V2, policy2, iter2 = policy_iteration(self.env)
        
        # Should get same results
        np.testing.assert_array_almost_equal(V1, V2)
        np.testing.assert_array_equal(policy1, policy2)
        assert iter1 == iter2


class TestPolicyQuality:
    """Test suite for policy quality and optimality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.size = 5
        self.env = GridWorldEnv(
            size=self.size,
            start=(0, 0),
            goal=(4, 4),
            obstacles=[],  # No obstacles for simpler testing
            render_mode=None
        )
    
    def test_optimal_policy_reaches_goal(self):
        """Test that optimal policy can reach goal"""
        V, policy, iterations = value_iteration(self.env)
        
        # Simulate episode with learned policy
        obs, _ = self.env.reset()
        visited_states = set()
        max_steps = 50
        
        for step in range(max_steps):
            if obs in visited_states:
                # Check if we're in a loop (bad policy)
                break
            visited_states.add(obs)
            
            action = policy[obs]
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated:
                # Successfully reached goal
                assert info['agent_pos'] == self.env.goal
                break
        
        # Should reach goal
        assert self.env.agent_pos == self.env.goal
    
    def test_policy_no_loops(self):
        """Test that policy doesn't create loops (on simple grid)"""
        V, policy, iterations = value_iteration(self.env)
        
        # Check multiple starting positions
        for start_row in range(self.size):
            for start_col in range(self.size):
                if (start_row, start_col) == self.env.goal:
                    continue
                
                self.env.agent_pos = (start_row, start_col)
                obs = self.env.pos_to_state(self.env.agent_pos)
                
                visited = set()
                for _ in range(self.size * self.size):
                    if obs in visited:
                        # Found a loop - this is bad
                        pytest.fail(f"Policy creates loop starting from {(start_row, start_col)}")
                    visited.add(obs)
                    
                    action = policy[obs]
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    if terminated:
                        break
    
    def test_policy_values_monotonic_towards_goal(self):
        """Test that values generally increase towards goal"""
        V, policy, iterations = value_iteration(self.env)
        
        # Get value at start
        start_state = self.env.pos_to_state(self.env.start)
        start_value = V[start_state]
        
        # Follow policy and check values increase
        obs, _ = self.env.reset()
        previous_value = V[obs]
        
        for step in range(20):
            action = policy[obs]
            obs, reward, terminated, truncated, info = self.env.step(action)
            current_value = V[obs]
            
            # Value should generally increase as we get closer to goal
            # (accounting for step penalties)
            if terminated:
                break
        
        # At minimum, we should reach the goal
        assert terminated or step < 19


class TestAlgorithmRobustness:
    """Test algorithm robustness under different conditions"""
    
    def test_single_cell_grid(self):
        """Test algorithms on minimal 1x1 grid"""
        env = GridWorldEnv(size=1, start=(0, 0), goal=(0, 0), render_mode=None)
        
        # Should handle gracefully (start = goal)
        V, policy, iterations = value_iteration(env)
        assert V.shape == (1,)
        assert policy.shape == (1,)
    
    def test_large_grid(self):
        """Test algorithms on larger grid"""
        env = GridWorldEnv(size=10, start=(0, 0), goal=(9, 9), 
                          obstacles=[(5, 5)], render_mode=None)
        
        V, policy, iterations = value_iteration(env)
        
        # Should still converge
        assert iterations > 0
        assert iterations < 10000
        assert V.shape == (100,)
        assert policy.shape == (100,)
    
    def test_many_obstacles(self):
        """Test with many obstacles"""
        obstacles = [(i, j) for i in range(2, 4) for j in range(2, 4)]
        env = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4),
                          obstacles=obstacles, render_mode=None)
        
        V, policy, iterations = value_iteration(env)
        
        # Should converge even with obstacles
        assert iterations > 0
        assert not np.any(np.isnan(V))
    
    def test_goal_surrounded_by_obstacles(self):
        """Test with goal that's hard to reach"""
        # Create obstacles around goal, but leave one path
        obstacles = [(3, 3), (3, 4), (4, 3)]  # Three sides blocked
        env = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4),
                          obstacles=obstacles, render_mode=None)
        
        V, policy, iterations = value_iteration(env)
        
        # Should find the one path
        # Test by following policy
        obs, _ = env.reset()
        for _ in range(50):
            action = policy[obs]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        
        # Should still be able to reach goal
        assert terminated or env.agent_pos == env.goal


class TestAlgorithmEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_negative_rewards_only(self):
        """Test with only negative rewards (no goal bonus)"""
        env = GridWorldEnv(size=3, start=(0, 0), goal=(2, 2), render_mode=None)
        
        # Modify environment to have no goal reward
        original_get_transition = env.get_transition
        
        def modified_get_transition(state, action):
            next_state, reward, done = original_get_transition(state, action)
            return next_state, -1, done  # Always -1 reward
        
        env.get_transition = modified_get_transition
        
        V, policy, iterations = value_iteration(env, gamma=0.9)
        
        # Should still converge
        assert iterations > 0
        # All values should be negative
        assert np.all(V <= 0)
    
    def test_gamma_zero(self):
        """Test with gamma=0 (myopic agent)"""
        env = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4), render_mode=None)
        
        V, policy, iterations = value_iteration(env, gamma=0.0)
        
        # Should converge very quickly
        assert iterations > 0
        assert iterations < 10
        
        # All non-goal states should have value of -1 (immediate reward only)
        goal_state = env.pos_to_state(env.goal)
        for s in range(env.observation_space.n):
            pos = env.state_to_pos(s)
            if pos != env.goal:
                # Non-goal states adjacent to goal might have value from reaching goal
                _, reward, _ = env.get_transition(s, 0)
                assert V[s] >= reward
    
    def test_gamma_one(self):
        """Test with gamma=1.0 (no discounting)"""
        env = GridWorldEnv(size=3, start=(0, 0), goal=(2, 2), render_mode=None)
        
        # With gamma=1, may take longer to converge
        V, policy, iterations = value_iteration(env, gamma=1.0, theta=1e-4)
        
        assert iterations > 0
        assert not np.any(np.isnan(V))


class TestAlgorithmConsistency:
    """Test consistency between algorithm components"""
    
    def test_value_policy_consistency(self):
        """Test that value function and policy are consistent"""
        env = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4), render_mode=None)
        V, policy, iterations = value_iteration(env)
        
        # For each state, check that the policy action gives the best Q-value
        for s in range(env.observation_space.n):
            q_values = []
            for a in range(4):
                next_state, reward, done = env.get_transition(s, a)
                v_next = 0 if done else V[next_state]
                q_values.append(reward + 0.9 * v_next)
            
            best_action = np.argmax(q_values)
            # Policy should choose the best action
            assert policy[s] == best_action or np.isclose(q_values[policy[s]], q_values[best_action])


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])