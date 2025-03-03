# Patch to fix ray_trainer.py by adding safe attribute access
import importlib
import sys
import os
import traceback

def apply_trainer_fix():
    """Apply safer attribute access to the fit method"""
    try:
        # Import the trainer module
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        
        # Get the original fit method
        original_fit = RayPPOTrainer.fit
        
        # Define the patched fit method
        def patched_fit(self):
            """Patched fit method with safe attribute checking"""
            assert hasattr(self, 'val_env'), 'Error: Validation environment not initialized!'

            self.logger = self.init_logger()
            self.actor_rollout_wg.set_logger(self.logger)
            if self.critic_wg:
                self.critic_wg.set_logger(self.logger)

            # Optionally run validation before training starts
            val_before_train = getattr(self.config.trainer, 'val_before_train', False)
            if val_before_train:
                self._validate()

            # Main training loop
            for epoch in range(self.config.trainer.total_epochs):
                metrics = {}
                timing_raw = {}
                avg_timing_raw = {}
                
                with _timer('step', timing_raw):
                    try:
                        ppo_batch = self._rollout()
                        with _timer('train', timing_raw):
                            metrics, results = self._update(ppo_batch)
                    except Exception as e:
                        print(f"Error in training step: {e}")
                        traceback.print_exc()
                        continue

                metrics.update({
                    "train_time_seconds/step": timing_raw.get('step', 0),
                    "train_time_seconds/rollout": timing_raw.get('rollout', 0),
                    "train_time_seconds/train": timing_raw.get('train', 0),
                    "train_time_seconds/process_data": timing_raw.get('process_data', 0),
                    "global_steps": self.global_steps,
                    "epochs": epoch
                })
                
                if hasattr(ppo_batch, 'batch'):
                    if 'reward' in ppo_batch.batch:
                        metrics["reward"] = ppo_batch.batch['reward']
                    if 'env_steps' in ppo_batch.batch:
                        metrics["env_steps"] = ppo_batch.batch['env_steps']

                # Log metrics
                self.logger.log(data=metrics, step=self.global_steps)

                # Reference policy update logic - with safe attribute checking
                has_ref_update_steps = hasattr(self.config.trainer, 'ref_update_steps') and self.config.trainer.ref_update_steps is not None
                if has_ref_update_steps and self.global_steps % self.config.trainer.ref_update_steps == 0:
                    with _timer('update_ref', timing_raw):
                        # move parameters from actor to ref_policy
                        self.actor_rollout_wg.update_ref()
                        metrics.update({
                            "train_time_seconds/update_ref": timing_raw.get('update_ref', 0),
                        })

                # Regular checkpoint saving
                if (self.global_steps + 1) % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()

                # Periodic validation
                if (self.global_steps + 1) % self.config.trainer.test_freq == 0:
                    val_metrics = self._validate()
                    
                self.global_steps += 1

            # Save final checkpoint
            self._save_checkpoint()
            return metrics
            
        # Apply the patched method
        RayPPOTrainer.fit = patched_fit
        print("✓ Successfully patched fit method with safe attribute checking")
        return True
    except Exception as e:
        print(f"✗ Failed to patch fit method: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply_trainer_fix()
