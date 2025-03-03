"""
Main script for traffic control training that uses our custom trainer
"""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import ray

# Import our custom trainer
from verl.trainer.ppo.traffic_trainer import TrafficControlTrainer

@ray.remote
def main_task(config):
    """Main training task using our custom trainer"""
    print("Starting traffic control training with custom trainer")
    
    # Create our custom trainer
    trainer = TrafficControlTrainer(
        config=config,
        env_cls=config.env.name
    )
    
    # Run training
    trainer.fit()
    
    return "Training complete"

@hydra.main(config_path="../conf", config_name="config")
def main(config: DictConfig):
    """Main entry point that uses our custom trainer"""
    ray.init(
        num_cpus=os.cpu_count(),
        include_dashboard=False,
        _temp_dir="/tmp/ray"
    )
    
    # Run the main task
    result = ray.get(main_task.remote(config))
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
