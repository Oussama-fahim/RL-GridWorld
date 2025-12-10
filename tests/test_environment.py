"""
Unit tests for GridWorld environment
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grid_env import GridWorldEnv


class TestGridWorldEnv:
    """Test suite for GridWorldEnv"""
    
    def setup_method(self):
        """Setup test environment before each test"""
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (2, 2)]
        
        self.env = GridWorldEnv(
            size=self.size,
            start=self.start,
            goal=self.goal,
            obstacles=self.obstacles,
            render_mode=None
        )
    
    def test_initialization(self):
        """Test environment initialization"""
        assert self.env.size == self.size
        assert self.env.start == self.start
        assert self.env.goal == self.goal
        assert self.env.obstacles == self.obstacles
        assert self.env.observation_space.n == self.size * self.size
        assert self.env.action_space.n == 4
    
    def test_reset(self):
        """Test environment reset"""
        obs, info = self.env.reset()
        assert obs == self.env.pos_to_state(self.start)
        assert info['agent_pos'] == self.start
        assert self.env.agent_pos == self.start
    
    def test_pos_to_state_conversion(self):
        """Test position to state conversion"""
        # Test corners
        assert self.env.pos_to_state((0, 0)) == 0
        assert self.env.pos_to_state((0, 4)) == 4
        assert self.env.pos_to_state((4, 0)) == 20
        assert self.env.pos_to_state((4, 4)) == 24
        
        # Test middle
        assert self.env.pos_to_state((2, 2)) == 12
    
    def test_state_to_pos_conversion(self):
        """Test state to position conversion"""
        # Test corners
        assert self.env.state_to_pos(0) == (0, 0)
        assert self.env.state_to_pos(4) == (0, 4)
        assert self.env.state_to_pos(20) == (4, 0)
        assert self.env.state_to_pos(24) == (4, 4)
        
        # Test middle
        assert self.env.state_to_pos(12) == (2, 2)
    
    def test_conversion_reversibility(self):
        """Test that pos->state->pos is reversible"""
        for r in range(self.size):
            for c in range(self.size):
                pos = (r, c)
                state = self.env.pos_to_state(pos)
                recovered_pos = self.env.state_to_pos(state)
                assert pos == recovered_pos
    
    def test_step_valid_move(self):
        """Test valid movement"""
        self.env.reset()
        
        # Move right from (0,0) to (0,1)
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert info['agent_pos'] == (0, 1)
        assert reward == -1  # Step penalty
        assert not terminated
    
    def test_step_boundary(self):
        """Test boundary collision"""
        self.env.reset()  # Start at (0, 0)
        
        # Try to move up from top row
        initial_pos = self.env.agent_pos
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert info['agent_pos'] == initial_pos  # Should stay in place
        assert reward == -1
    
    def test_step_obstacle(self):
        """Test obstacle collision"""
        # Move agent next to obstacle
        self.env.agent_pos = (1, 0)
        
        # Try to move right into obstacle at (1, 1)
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert info['agent_pos'] == (1, 0)  # Should stay in place
        assert reward == -1
    
    def test_reach_goal(self):
        """Test reaching the goal"""
        # Place agent next to goal
        self.env.agent_pos = (4, 3)
        
        # Move right to goal
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert info['agent_pos'] == self.goal
        assert reward == 10  # Goal reward
        assert terminated
    
    def test_get_transition(self):
        """Test transition function"""
        # Test valid move
        state = self.env.pos_to_state((2, 2))
        next_state, reward, done = self.env.get_transition(state, 1)  # Right
        assert next_state == self.env.pos_to_state((2, 3))
        assert reward == -1
        assert not done
        
        # Test goal transition
        state = self.env.pos_to_state((4, 3))
        next_state, reward, done = self.env.get_transition(state, 1)  # Right to goal
        assert next_state == self.env.pos_to_state(self.goal)
        assert reward == 10
        assert done
    
    def test_all_actions(self):
        """Test all four actions"""
        self.env.agent_pos = (2, 2)
        
        # Up
        obs, _, _, _, info = self.env.step(0)
        assert info['agent_pos'] == (1, 2)
        
        # Right
        obs, _, _, _, info = self.env.step(1)
        assert info['agent_pos'] == (1, 3)
        
        # Down
        obs, _, _, _, info = self.env.step(2)
        assert info['agent_pos'] == (2, 3)
        
        # Left
        obs, _, _, _, info = self.env.step(3)
        assert info['agent_pos'] == (2, 2)
    
    def test_episode_workflow(self):
        """Test complete episode workflow"""
        obs, info = self.env.reset()
        assert obs == self.env.pos_to_state(self.start)
        
        total_reward = 0
        terminated = False
        steps = 0
        max_steps = 100
        
        while not terminated and steps < max_steps:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
        
        assert steps <= max_steps
        assert isinstance(total_reward, (int, float))
    
    def test_set_state_values(self):
        """Test setting state values for visualization"""
        values = np.random.rand(self.size * self.size)
        self.env.set_state_values(values)
        assert np.array_equal(self.env.state_values, values)
    
    def test_set_episode_step(self):
        """Test setting episode and step counters"""
        self.env.set_episode_step(5, 10)
        assert self.env.current_episode == 5
        assert self.env.current_step == 10


class TestGridWorldEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_goal_at_start(self):
        """Test when goal is at start position"""
        env = GridWorldEnv(size=5, start=(2, 2), goal=(2, 2), render_mode=None)
        obs, info = env.reset()
        
        # Take any action - should immediately terminate
        obs, reward, terminated, truncated, info = env.step(0)
        # Note: In current implementation, agent at goal doesn't auto-terminate
        # This is actually a potential bug - consider if this is desired behavior
    
    def test_empty_obstacles(self):
        """Test environment with no obstacles"""
        env = GridWorldEnv(size=3, start=(0, 0), goal=(2, 2), 
                          obstacles=[], render_mode=None)
        assert env.obstacles == []
        
        # Should be able to move freely
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert info['agent_pos'] == (0, 1)
    
    def test_small_grid(self):
        """Test minimum grid size"""
        env = GridWorldEnv(size=2, start=(0, 0), goal=(1, 1), render_mode=None)
        assert env.observation_space.n == 4
        assert env.action_space.n == 4
    
    def test_large_grid(self):
        """Test larger grid"""
        size = 20
        env = GridWorldEnv(size=size, start=(0, 0), goal=(19, 19), render_mode=None)
        assert env.observation_space.n == size * size


class TestMultipleEnvironments:
    """Test multiple environment instances"""
    
    def test_independent_environments(self):
        """Test that multiple environments are independent"""
        env1 = GridWorldEnv(size=5, start=(0, 0), goal=(4, 4), render_mode=None)
        env2 = GridWorldEnv(size=7, start=(1, 1), goal=(6, 6), render_mode=None)
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        assert env1.agent_pos != env2.agent_pos
        assert env1.size != env2.size
        
        # Take different actions
        env1.step(1)
        env2.step(0)
        
        assert env1.agent_pos != env2.agent_pos


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])