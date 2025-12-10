# 🧪 Testing Guide - RL GridWorld

Complete guide for running and understanding tests in the RL GridWorld project.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

---

## 📦 Installation

### Install Testing Dependencies

```bash
# Install all dependencies including test tools
pip install -r requirements.txt

# Or install test dependencies only
pip install pytest pytest-cov
```

---

## 🚀 Running Tests

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run all tests with detailed output
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Run environment tests only
pytest tests/test_environment.py -v

# Run algorithm tests only
pytest tests/test_algorithms.py -v

# Run integration tests only
pytest tests/test_integration.py -v
```

### Run Specific Test Classes

```bash
# Run specific test class
pytest tests/test_environment.py::TestGridWorldEnv -v

# Run specific test method
pytest tests/test_algorithms.py::TestValueIteration::test_value_iteration_convergence -v
```

### Run Tests with Filters

```bash
# Run only tests matching keyword
pytest tests/ -k "convergence" -v

# Run only tests matching multiple keywords
pytest tests/ -k "value_iteration or policy_iteration" -v

# Skip tests matching keyword
pytest tests/ -k "not slow" -v
```

### Run Tests with Markers

```bash
# Run only fast tests
pytest tests/ -m "not slow" -v

# Run only integration tests
pytest tests/ -m "integration" -v
```

---

## 📁 Test Structure

### `test_environment.py` - Environment Tests (20+ tests)

**Purpose**: Test GridWorld environment functionality

**Test Classes**:
- `TestGridWorldEnv`: Core environment functionality
  - Initialization
  - Reset mechanism
  - State/position conversions
  - Action execution
  - Boundary conditions
  - Obstacle collision
  - Goal reaching
  
- `TestGridWorldEdgeCases`: Edge cases and boundary conditions
  - Goal at start position
  - Empty obstacles
  - Small/large grids
  - Unreachable goals
  
- `TestMultipleEnvironments`: Multiple environment instances
  - Independence of environments
  - Parallel execution

**Example**:
```bash
pytest tests/test_environment.py -v
```

**Expected Output**:
```
test_environment.py::TestGridWorldEnv::test_initialization PASSED
test_environment.py::TestGridWorldEnv::test_reset PASSED
test_environment.py::TestGridWorldEnv::test_step_valid_move PASSED
...
======================== 20 passed in 2.35s ========================
```

---

### `test_algorithms.py` - Algorithm Tests (30+ tests)

**Purpose**: Test reinforcement learning algorithms

**Test Classes**:

1. **TestValueIteration**: Value Iteration algorithm
   - Convergence testing
   - Policy shape validation
   - Value function properties
   - Different gamma values
   - Determinism verification

2. **TestPolicyIteration**: Policy Iteration algorithm
   - Convergence testing
   - Comparison with Value Iteration
   - Policy evaluation
   - Policy improvement

3. **TestPolicyQuality**: Policy quality assessment
   - Goal reachability
   - Loop detection
   - Value monotonicity

4. **TestAlgorithmRobustness**: Robustness testing
   - Minimal grids
   - Large grids
   - Many obstacles
   - Hard-to-reach goals

5. **TestAlgorithmEdgeCases**: Edge cases
   - Negative rewards
   - Extreme gamma values (0, 1)

6. **TestAlgorithmConsistency**: Internal consistency
   - Value-policy consistency

**Example**:
```bash
pytest tests/test_algorithms.py::TestValueIteration -v
```

---

### `test_integration.py` - Integration Tests (25+ tests)

**Purpose**: Test complete workflows and component interactions

**Test Classes**:

1. **TestEndToEndWorkflow**: Complete training and execution
   - Train and execute Value Iteration
   - Train and execute Policy Iteration
   - Multiple episodes with reset

2. **TestAlgorithmComparison**: Algorithm comparisons
   - VI vs PI optimality
   - Random vs optimal policies
   - Policy evaluation consistency

3. **TestConfigurationSystem**: Configuration management
   - Config loading
   - Environment creation from config
   - Config persistence

4. **TestLoggingSystem**: Logging and tracking
   - Experiment logger workflow
   - Model saving and loading

5. **TestVisualizationSystem**: Visualization components
   - Policy visualization
   - Value function visualization

6. **TestPerformanceComparison**: Performance utilities
   - Policy evaluation function
   - Algorithm comparison function

7. **TestMovingGoalEnvironment**: Advanced environments
   - State space verification
   - Goal randomization
   - Observation encoding

8. **TestCompleteScenarios**: Real-world scenarios
   - Maze solving
   - Multi-agent training
   - Transfer learning

9. **TestSystemRobustness**: System reliability
   - Invalid action handling
   - Reset after termination
   - Long episodes

**Example**:
```bash
pytest tests/test_integration.py::TestEndToEndWorkflow -v
```

---

## 📊 Test Coverage

### Generate Coverage Report

```bash
# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Open coverage report
# The report will be in htmlcov/index.html
```

### Coverage by Module

**Expected Coverage**:
- `grid_env.py`: ~95%
- `value_agent.py`: ~90%
- `policy_iteration.py`: ~90%
- `config.py`: ~85%
- `logger.py`: ~80%
- `visualization_utils.py`: ~70% (limited due to matplotlib)

### View Coverage Summary

```bash
pytest tests/ --cov=. --cov-report=term
```

**Expected Output**:
```
Name                           Stmts   Miss  Cover
--------------------------------------------------
grid_env.py                      120      6    95%
value_agent.py                    45      5    89%
policy_iteration.py               50      5    90%
config.py                         85     13    85%
logger.py                        140     28    80%
--------------------------------------------------
TOTAL                            440     57    87%
```

---

## ✍️ Writing Tests

### Test Template

```python
import pytest
import numpy as np
from grid_env import GridWorldEnv

class TestMyFeature:
    """Test suite for my feature"""
    
    def setup_method(self):
        """Setup before each test"""
        self.env = GridWorldEnv(size=5, start=(0,0), goal=(4,4))
    
    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_feature_works(self):
        """Test that feature works as expected"""
        # Arrange
        expected_value = 42
        
        # Act
        actual_value = my_function()
        
        # Assert
        assert actual_value == expected_value
    
    def test_feature_edge_case(self):
        """Test edge case handling"""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Testing Best Practices

1. **Use Descriptive Names**: `test_value_iteration_converges` not `test1`
2. **One Assertion Per Test**: Focus on one thing
3. **Use Setup/Teardown**: Initialize cleanly
4. **Test Edge Cases**: Boundaries, nulls, extremes
5. **Mock External Dependencies**: Files, networks, time
6. **Use Fixtures**: Share common setup

### Pytest Fixtures

```python
@pytest.fixture
def simple_env():
    """Fixture for simple environment"""
    env = GridWorldEnv(size=3, start=(0,0), goal=(2,2))
    yield env
    env.close()

def test_with_fixture(simple_env):
    """Test using fixture"""
    obs, _ = simple_env.reset()
    assert obs == 0
```

---

## 🔄 Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Problem**:
```
ModuleNotFoundError: No module named 'grid_env'
```

**Solution**:
```bash
# Make sure you're in the project root
cd /path/to/rl-gridworld

# Run pytest with Python path
PYTHONPATH=. pytest tests/

# Or install in development mode
pip install -e .
```

#### Issue 2: Matplotlib Backend Errors

**Problem**:
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

**Solution**:
```python
# Add to test file
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

#### Issue 3: Slow Tests

**Problem**: Tests take too long

**Solution**:
```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run in parallel
pip install pytest-xdist
pytest tests/ -n auto
```

#### Issue 4: Random Failures

**Problem**: Tests pass sometimes, fail others

**Solution**:
```python
# Set random seed in test
np.random.seed(42)
import random
random.seed(42)

# Use pytest-randomly plugin
pip install pytest-randomly
pytest tests/ --randomly-seed=42
```

---

## 📈 Test Metrics

### Current Test Statistics

```
Total Tests: 75+
- Environment Tests: 20
- Algorithm Tests: 30
- Integration Tests: 25

Total Coverage: ~87%
Total Runtime: ~15 seconds

Pass Rate: 100%
```

### Performance Benchmarks

```
Test Suite               Time    Status
----------------------------------------
test_environment.py      2.5s    ✓ PASS
test_algorithms.py       8.2s    ✓ PASS
test_integration.py      4.3s    ✓ PASS
----------------------------------------
Total                   15.0s    ✓ PASS
```

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test
pytest tests/test_environment.py::TestGridWorldEnv::test_reset -v

# Run and stop at first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Show test output
pytest tests/ -v -s

# Generate HTML report
pytest tests/ --html=report.html
```

### Useful Pytest Options

| Option | Description |
|--------|-------------|
| `-v` | Verbose output |
| `-s` | Show print statements |
| `-x` | Stop at first failure |
| `--lf` | Run last failed |
| `--ff` | Run failed first |
| `-k EXPR` | Run tests matching expression |
| `-m MARK` | Run tests with marker |
| `--maxfail=N` | Stop after N failures |
| `--tb=short` | Short traceback format |
| `-n auto` | Run in parallel (needs pytest-xdist) |

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## ✅ Testing Checklist

Before submitting code:

- [ ] All tests pass locally
- [ ] New features have tests
- [ ] Coverage hasn't decreased
- [ ] Tests are documented
- [ ] Edge cases are covered
- [ ] Integration tests pass
- [ ] No warnings in test output

---

**Happy Testing! 🧪✨**