# Environment-driven training patch
# This patch will modify the trainer to skip dataset usage for environment-driven training

import torch
import os
import sys
from verl import DataProto
from typing import List, Dict, Any, Tuple, Optional
import inspect

# Patch functions for the PPO trainer
def patch_init_dataloader(original_method):
    """Patch for initializing dataloaders to skip for environment training"""
    
    def patched_method(self, *args, **kwargs):
        if hasattr(self, 'config') and self.config.env.name == 'traffic_control':
            print("ENVIRONMENT-DRIVEN TRAINING: Skipping dataloader initialization")
            self.train_dataloader = None
            self.val_dataloader = None
            return
        
        return original_method(self, *args, **kwargs)
    
    return patched_method

def patch_validate(original_method):
    """Patch for the validation method to use environment-driven approach"""
    
    def patched_method(self, *args, **kwargs):
        if not hasattr(self, 'val_dataloader') or self.val_dataloader is None:
            if hasattr(self, 'val_env') and self.val_env is not None:
                print("ENVIRONMENT-DRIVEN VALIDATION: Using validation environment directly")
                # Create validation environments
                val_envs = [self.val_env.reset() for _ in range(self.config.data.val_batch_size)]
                
                # Create dummy batch for environment-driven validation
                dummy_batch = DataProto.from_dict({
                    'input_ids': torch.ones((len(val_envs), 1), dtype=torch.long),
                    'attention_mask': torch.ones((len(val_envs), 1), dtype=torch.long),
                    'position_ids': torch.zeros((len(val_envs), 1), dtype=torch.long)
                })
                
                # Run validation through environment
                try:
                    # Note: We need to check if the method accepts is_env_driven parameter
                    import inspect
                    rollout_params = inspect.signature(self.rollout_once).parameters
                    if 'is_env_driven' in rollout_params:
                        results = self.rollout_once(
                            dummy_batch, 
                            val_envs, 
                            is_env_driven=True
                        )
                    else:
                        results = self.rollout_once(
                            dummy_batch, 
                            val_envs
                        )
                    
                    # Process validation metrics as needed
                    val_metrics = {
                        'val_reward': sum([env.get_reward() for env in val_envs]) / len(val_envs),
                        'val_steps': sum([env.get_step_count() for env in val_envs]) / len(val_envs)
                    }
                    
                    return val_metrics
                except Exception as e:
                    print(f"ERROR in environment validation: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return empty metrics if validation fails
                    return {}
            else:
                print("WARNING: No validation dataloader or environment found. Skipping validation.")
                return {}
        
        return original_method(self, *args, **kwargs)
    
    return patched_method

def patch_fit(original_method):
    """Patch for the fit method to handle environment-driven approach"""
    
    def patched_method(self, *args, **kwargs):
        if not hasattr(self, 'train_dataloader') or self.train_dataloader is None:
            if not hasattr(self, 'env') or self.env is None:
                print("ERROR: No training dataloader or environment available. Cannot train.")
                return
                
            print("ENVIRONMENT-DRIVEN TRAINING: Using environment directly")
            # Modified training loop for environment-driven approach
            for epoch in range(self.config.trainer.total_epochs):
                print(f"Starting epoch {epoch}")
                
                # Create environments
                train_envs = [self.env.reset() for _ in range(self.config.data.train_batch_size)]
                
                # Create dummy batch
                dummy_batch = DataProto.from_dict({
                    'input_ids': torch.ones((len(train_envs), 1), dtype=torch.long),
                    'attention_mask': torch.ones((len(train_envs), 1), dtype=torch.long),
                    'position_ids': torch.zeros((len(train_envs), 1), dtype=torch.long)
                })
                
                # Run training through environment
                try:
                    # Check method signature to see if it accepts is_env_driven parameter
                    import inspect
                    train_epoch_params = inspect.signature(self.train_epoch).parameters
                    rollout_params = inspect.signature(self.rollout_once).parameters
                    
                    # First try to call train_epoch with is_env_driven
                    if 'is_env_driven' in train_epoch_params:
                        self.train_epoch(dummy_batch, train_envs, is_env_driven=True)
                    elif 'envs' in train_epoch_params:
                        # If it accepts envs but not is_env_driven
                        self.train_epoch(dummy_batch, envs=train_envs)
                    else:
                        # Just do a basic rollout_once if train_epoch doesn't work
                        if 'is_env_driven' in rollout_params:
                            self.rollout_once(dummy_batch, train_envs, is_env_driven=True)
                        else:
                            self.rollout_once(dummy_batch, train_envs)
                    
                    # Run validation if needed
                    if epoch % self.config.trainer.test_freq == 0:
                        val_metrics = self._validate()
                        print(f"Validation metrics: {val_metrics}")
                    
                    # Save checkpoint if needed
                    if epoch % self.config.trainer.save_freq == 0:
                        self.save_checkpoint(epoch)
                except Exception as e:
                    print(f"ERROR in training epoch: {e}")
                    import traceback
                    traceback.print_exc()
            
            return
        
        return original_method(self, *args, **kwargs)
    
    return patched_method

# Apply the patches to the RayPPOTrainer class
def apply_patches():
    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        
        # Find the dataloader initialization method
        dataloader_method = None
        validate_method = None
        fit_method = RayPPOTrainer.fit  # This one we know exists
        
        # Inspect all methods to find the ones we need
        for name, method in inspect.getmembers(RayPPOTrainer, inspect.isfunction):
            if 'dataloader' in name.lower() and 'init' in name.lower():
                print(f"Found dataloader init method: {name}")
                dataloader_method = method
            elif 'validate' in name.lower():
                print(f"Found validation method: {name}")
                validate_method = method
        
        # Apply patches to the methods we found
        if dataloader_method:
            method_name = dataloader_method.__name__
            setattr(RayPPOTrainer, method_name, patch_init_dataloader(dataloader_method))
            print(f"Patched {method_name}")
        
        if validate_method:
            method_name = validate_method.__name__
            setattr(RayPPOTrainer, method_name, patch_validate(validate_method))
            print(f"Patched {method_name}")
        
        # Always patch fit
        setattr(RayPPOTrainer, "fit", patch_fit(fit_method))
        print("Patched fit method")
        
        print("Applied environment-driven training patches to RayPPOTrainer")
        return True
    except Exception as e:
        print(f"Failed to apply patches: {e}")
        import traceback
        traceback.print_exc()
        return False

# Apply patches when imported
apply_patches()
