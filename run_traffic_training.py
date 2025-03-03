"""
Standalone script for traffic control training
"""
import os
import sys
import ray
from verl.trainer.ppo.traffic_trainer import TrafficControlTrainer
import yaml
from omegaconf import OmegaConf

def load_config():
    """Load configuration directly from standard location"""
    # Start with basic config
    config = {
        "env": {
            "name": "traffic_control",
            "path_to_work_directory": os.path.join(os.getcwd(), "data/traffic"),
            "roadnet_file": "roadnet.json",
            "flow_file": "flow.json",
            "min_action_time": 15,
            "max_steps": 300,
            "num_intersections": 1
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "total_epochs": 150,
            "save_freq": 5,
            "test_freq": 2,
            "default_local_dir": "./checkpoints/traffic_control",
            "project_name": "traffic_control",
            "experiment_name": "optimized_1x1"
        },
        "model": "qwen_0_5b",
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "tensor_parallel_size": 1,
                "rollout_tensor_parallel_size": 1
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "temperature": 0.8,
                "top_p": 0.9
            },
            "actor": {
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size": 2,
                "clip_ratio": 0.2,
                "entropy_coeff": 0.01,
                "ppo_epochs": 4,
                "optim": {"lr": 5e-6}
            }
        },
        "critic": {
            "optim": {"lr": 1e-5},
            "cliprange_value": 0.2,
            "tensor_parallel_size": 1
        },
        "algorithm": {
            "gamma": 0.99,
            "lam": 0.95,
            "adv_estimator": "gae",
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.0005
            }
        },
        "data": {
            "train_batch_size": 2,
            "val_batch_size": 2,
            "use_dataset": False,
            "use_dataset_training": False
        }
    }
    
    # Convert to OmegaConf object
    return OmegaConf.create(config)

@ray.remote
def main_task(config):
    """Main training task"""
    print("Starting traffic control training with custom trainer")
    print(f"Environment config: {config.env}")
    
    # Create our custom trainer
    trainer = TrafficControlTrainer(
        config=config,
        env_cls=config.env.name
    )
    
    # Run training
    trainer.fit()
    
    return "Training complete"

def main():
    """Main entry point that uses our custom trainer"""
    # Initialize Ray
    ray.init(
        num_cpus=os.cpu_count(),
        include_dashboard=False,
        _temp_dir="/tmp/ray"
    )
    
    # Load config directly
    config = load_config()
    
    # Run the main task
    result = ray.get(main_task.remote(config))
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
