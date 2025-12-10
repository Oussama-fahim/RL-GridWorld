"""
Logging and model saving utilities
"""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from pathlib import Path


class ExperimentLogger:
    """Logger for tracking experiments and results"""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{self.timestamp}"
        
        # Create run directory
        self.run_dir = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Storage for metrics
        self.metrics = {
            'episodes': [],
            'steps': [],
            'rewards': [],
            'success': [],
            'training_time': 0,
            'hyperparameters': {}
        }
        
        self.logger.info(f"Initialized experiment: {self.run_name}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.run_dir / "experiment.log"
        
        # Create logger
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters"""
        self.metrics['hyperparameters'] = hyperparameters
        self.logger.info(f"Hyperparameters: {hyperparameters}")
        
        # Save to JSON
        hp_file = self.run_dir / "hyperparameters.json"
        with open(hp_file, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
    
    def log_episode(self, episode: int, steps: int, reward: float, 
                   success: bool, extra_info: Optional[Dict] = None):
        """Log episode results"""
        self.metrics['episodes'].append(episode)
        self.metrics['steps'].append(steps)
        self.metrics['rewards'].append(reward)
        self.metrics['success'].append(1 if success else 0)
        
        info_str = f"Episode {episode}: Steps={steps}, Reward={reward:.2f}, Success={success}"
        if extra_info:
            info_str += f", Extra={extra_info}"
        
        self.logger.info(info_str)
    
    def log_training_time(self, time_seconds: float):
        """Log total training time"""
        self.metrics['training_time'] = time_seconds
        self.logger.info(f"Total training time: {time_seconds:.2f}s")
    
    def log_convergence(self, iterations: int, final_delta: float):
        """Log convergence information"""
        self.logger.info(f"Converged in {iterations} iterations with delta={final_delta:.6f}")
    
    def save_metrics(self):
        """Save all metrics to file"""
        metrics_file = self.run_dir / "metrics.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, list):
                serializable_metrics[key] = [
                    float(x) if isinstance(x, (np.integer, np.floating)) else x 
                    for x in value
                ]
            else:
                serializable_metrics[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics['episodes']:
            return {}
        
        return {
            'total_episodes': len(self.metrics['episodes']),
            'success_rate': np.mean(self.metrics['success']),
            'avg_steps': np.mean(self.metrics['steps']),
            'avg_reward': np.mean(self.metrics['rewards']),
            'training_time': self.metrics['training_time']
        }
    
    def print_summary(self):
        """Print experiment summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print("="*60)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
        print("="*60 + "\n")


class ModelSaver:
    """Utility for saving and loading trained models"""
    
    def __init__(self, save_dir: str = "saved_models"):
        """
        Initialize model saver
        
        Args:
            save_dir: Directory to save models
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model_data: Dict[str, Any], model_name: str,
                  metadata: Optional[Dict] = None):
        """
        Save model and metadata
        
        Args:
            model_data: Dictionary containing model arrays (V, policy, Q, etc.)
            model_name: Name for the saved model
            metadata: Additional metadata to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.save_dir / model_filename
        
        # Prepare data to save
        save_data = {
            'model': model_data,
            'metadata': metadata or {},
            'timestamp': timestamp
        }
        
        # Save using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Model saved to: {model_path}")
        
        # Also save metadata as JSON for easy inspection
        if metadata:
            json_path = self.save_dir / f"{model_name}_{timestamp}_metadata.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load saved model
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Dictionary containing model data and metadata
        """
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")
        return data
    
    def list_saved_models(self) -> list:
        """List all saved models"""
        models = list(self.save_dir.glob("*.pkl"))
        return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def save_policy(self, policy: np.ndarray, policy_name: str, 
                   env_config: Dict):
        """
        Save policy with environment configuration
        
        Args:
            policy: Policy array
            policy_name: Name for the policy
            env_config: Environment configuration
        """
        model_data = {
            'policy': policy,
            'env_config': env_config
        }
        
        metadata = {
            'type': 'policy',
            'shape': policy.shape,
            'env_size': env_config.get('size', 'unknown')
        }
        
        self.save_model(model_data, policy_name, metadata)
    
    def save_value_function(self, V: np.ndarray, V_name: str, 
                           env_config: Dict):
        """
        Save value function
        
        Args:
            V: Value function array
            V_name: Name for the value function
            env_config: Environment configuration
        """
        model_data = {
            'V': V,
            'env_config': env_config
        }
        
        metadata = {
            'type': 'value_function',
            'shape': V.shape,
            'max_value': float(np.max(V)),
            'min_value': float(np.min(V))
        }
        
        self.save_model(model_data, V_name, metadata)
    
    def save_q_function(self, Q: np.ndarray, Q_name: str, 
                       env_config: Dict):
        """
        Save Q-function
        
        Args:
            Q: Q-function array
            Q_name: Name for the Q-function
            env_config: Environment configuration
        """
        model_data = {
            'Q': Q,
            'env_config': env_config
        }
        
        metadata = {
            'type': 'q_function',
            'shape': Q.shape,
            'max_q_value': float(np.max(Q)),
            'min_q_value': float(np.min(Q))
        }
        
        self.save_model(model_data, Q_name, metadata)


def create_experiment_report(logger: ExperimentLogger, save_path: str = None):
    """
    Create comprehensive experiment report
    
    Args:
        logger: ExperimentLogger instance
        save_path: Path to save the report
    """
    if save_path is None:
        save_path = logger.run_dir / "experiment_report.txt"
    
    summary = logger.get_summary()
    
    report = f"""
{'='*70}
EXPERIMENT REPORT
{'='*70}

Experiment Name: {logger.experiment_name}
Run Name: {logger.run_name}
Timestamp: {logger.timestamp}

{'='*70}
HYPERPARAMETERS
{'='*70}
{json.dumps(logger.metrics['hyperparameters'], indent=2)}

{'='*70}
RESULTS SUMMARY
{'='*70}
Total Episodes: {summary.get('total_episodes', 0)}
Success Rate: {summary.get('success_rate', 0):.2%}
Average Steps per Episode: {summary.get('avg_steps', 0):.2f}
Average Reward per Episode: {summary.get('avg_reward', 0):.2f}
Total Training Time: {summary.get('training_time', 0):.2f} seconds

{'='*70}
EPISODE STATISTICS
{'='*70}
"""
    
    if logger.metrics['steps']:
        report += f"""
Steps:
  - Min: {np.min(logger.metrics['steps'])}
  - Max: {np.max(logger.metrics['steps'])}
  - Mean: {np.mean(logger.metrics['steps']):.2f}
  - Std: {np.std(logger.metrics['steps']):.2f}

Rewards:
  - Min: {np.min(logger.metrics['rewards']):.2f}
  - Max: {np.max(logger.metrics['rewards']):.2f}
  - Mean: {np.mean(logger.metrics['rewards']):.2f}
  - Std: {np.std(logger.metrics['rewards']):.2f}
"""
    
    report += f"\n{'='*70}\n"
    
    # Write report
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Experiment report saved to: {save_path}")
    print(report)


if __name__ == "__main__":
    # Example usage
    logger = ExperimentLogger("test_experiment")
    
    # Log hyperparameters
    logger.log_hyperparameters({
        'algorithm': 'Q-Learning',
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'episodes': 1000
    })
    
    # Simulate some episodes
    for episode in range(10):
        steps = np.random.randint(10, 50)
        reward = np.random.uniform(-20, 10)
        success = np.random.random() > 0.5
        logger.log_episode(episode, steps, reward, success)
    
    # Log training time
    logger.log_training_time(120.5)
    
    # Save metrics
    logger.save_metrics()
    
    # Print summary
    logger.print_summary()
    
    # Create report
    create_experiment_report(logger)
    
    # Example of saving models
    saver = ModelSaver()
    policy = np.random.randint(0, 4, 25)
    env_config = {'size': 5, 'start': (0, 0), 'goal': (4, 4)}
    saver.save_policy(policy, "random_policy", env_config)