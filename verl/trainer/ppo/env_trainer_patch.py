# Environment-driven training patch
# This patch will modify the trainer to skip dataset usage for environment-driven training

import torch
import ray
from verl import DataProto
from typing import List, Dict, Any, Tuple, Optional

# Patch functions for the PPO trainer
def patch_init_dataloader(original_init_dataloaders):
    """Patch for initializing dataloaders to skip for environment training"""
    
    def patched_init_dataloaders(self, *args, **kwargs):
        if hasattr(self, 'config') and self.config.env.name == 'traffic_control':
            print("ENVIRONMENT-DRIVEN TRAINING: Skipping dataloader initialization")
            self.train_dataloader = None
            self.val_dataloader = None
            return
        
        return original_init_dataloaders(self, *args, **kwargs)
    
    return patched_init_dataloaders

def patch_validate(original_validate):
    """Patch for the validation method to use environment-driven approach"""
    
    def patched_validate(self, *args, **kwargs):
        if not hasattr(self, 'val_dataloader') or self.val_dataloader is None:
            if hasattr(self, 'val_env') and self.val_env is not None:
                print("ENVIRONMENT-DRIVEN VALIDATION: Using validation environment directly")
                # Create validation environments
                val_envs = [self.val_env.reset() for _ in range(self.config.data.val_batch_size)]
                
                # Get initial observations
                initial_obs = [env.get_obs() for env in val_envs]
                
                # Create dummy batch for environment-driven validation
                dummy_batch = DataProto.from_dict({
                    'input_ids': torch.ones((len(val_envs), 1), dtype=torch.long),
                    'attention_mask': torch.ones((len(val_envs), 1), dtype=torch.long),
                    'position_ids': torch.zeros((len(val_envs), 1), dtype=torch.long)
                })
                
                # Run validation through environment
                results = self.rollout_once(
                    dummy_batch, 
                    val_envs, 
                    is_env_driven=True
                )
                
                # Process validation metrics as needed
                val_metrics = {
                    'val_reward': sum([env.get_reward() for env in val_envs]) / len(val_envs),
                    'val_steps': sum([env.get_step_count() for env in val_envs]) / len(val_envs)
                }
                
                return val_metrics
            else:
                print("WARNING: No validation dataloader or environment found. Skipping validation.")
                return {}
        
        return original_validate(self, *args, **kwargs)
    
    return patched_validate

def patch_rollout_once(original_rollout_once):
    """Patch for the rollout method to handle environment-driven approach"""
    
    def patched_rollout_once(self, batch, envs=None, is_env_driven=False, *args, **kwargs):
        if is_env_driven and envs is not None:
            # For environment-driven training, we'll use the environments directly
            # This approach bypasses the need for dataset loading
            
            # Run the LLM generation loop with the environments
            llm_gen_manager = self.llm_generation_manager
            
            # Get initial input (can be None for environment-driven approach)
            initial_input_ids = None
            
            # Run LLM generation loop
            final_output = llm_gen_manager.run_llm_loop(
                None,  # No batch needed
                envs,
                initial_input_ids,
                self.output_dir,
                self.global_steps
            )
            
            return final_output
        
        return original_rollout_once(self, batch, envs, *args, **kwargs)
    
    return patched_rollout_once

# Apply the patches to the RayPPOTrainer class
def apply_patches():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    
    # Store original methods
    original_init_dataloaders = RayPPOTrainer._init_dataloaders
    original_validate = RayPPOTrainer._validate
    original_rollout_once = RayPPOTrainer.rollout_once
    
    # Apply patches
    RayPPOTrainer._init_dataloaders = patch_init_dataloader(original_init_dataloaders)
    RayPPOTrainer._validate = patch_validate(original_validate)
    RayPPOTrainer.rollout_once = patch_rollout_once(original_rollout_once)
    
    print("Applied environment-driven training patches to RayPPOTrainer") 