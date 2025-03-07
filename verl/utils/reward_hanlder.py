from verl import DataProto
import torch

def get_token_level_reward(batch: DataProto):
    """
    Convert final outcome rewards to token-level scores.
    
    Args:
        batch: DataProto containing the batch data with rewards
        
    Returns:
        token_level_scores: Tensor of shape [batch_size, response_length] with rewards
                           assigned to the final token of each response
    """
    # Get the responses and attention mask
    responses = batch.batch['responses']
    response_length = responses.size(1)
    attention_mask = batch.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    
    # Create a tensor of zeros with the same shape as responses
    token_level_scores = torch.zeros_like(responses, dtype=torch.float32)
    
    # Get the rewards from the batch
    rewards = batch.non_tensor_batch['reward']
    
    # For each example in the batch, assign the reward to the last valid token
    for i, reward in enumerate(rewards):
        # Find the position of the last valid token in the response
        valid_length = response_mask[i].sum().item()
        if valid_length > 0:
            # Assign the reward to the last valid token
            token_level_scores[i, valid_length - 1] = float(reward)
    
    return token_level_scores