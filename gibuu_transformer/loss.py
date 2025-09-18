"""
Loss functions for GiBUU Transformer.
"""

import torch
import torch.nn.functional as F
from .constants import EOS_STEP_TOKEN, PAD_TOKEN


def particle_gpt_loss(output, batch):
    """
    GPT-style loss for particle sequence generation.
    Each position predicts the next token based on all previous tokens.
    
    Parameters:
    -----------
    output: dict
        Model output with 'particle_types' and 'particle_feats'
    batch: dict
        Contains 'encoded_ids', 'particle_feats', 'padding_mask'
        
    Returns:
    --------
    loss: dict
        Dictionary with 'total_loss', 'type_loss', 'feat_loss'
    """
    particle_types = output['particle_types']  # (batch, seq_len, num_classes)
    particle_feats = output['particle_feats']  # (batch, seq_len, 7)
    
    # Targets (shifted by 1 for next token prediction)
    targets_types = batch['encoded_ids']  # (batch, seq_len)
    targets_feats = batch['particle_feats']  # (batch, seq_len, 7)
    
    # Create shifted targets (next token prediction)
    # Input:  [token1, token2, token3, token4]
    # Target: [token2, token3, token4, token5]
    targets_types_shifted = targets_types[:, 1:]  # (batch, seq_len-1)
    targets_feats_shifted = targets_feats[:, 1:]  # (batch, seq_len-1)
    
    # Shift predictions to match targets
    particle_types_shifted = particle_types[:, :-1]  # (batch, seq_len-1, num_classes)
    particle_feats_shifted = particle_feats[:, :-1]  # (batch, seq_len-1, 7)

    # Create masks to only consider loss starting from the second step
    batch_size, seq_len = targets_types.shape
    valid_loss_mask = torch.ones_like(targets_types_shifted, dtype=torch.bool)
    eos_mask = (targets_types == EOS_STEP_TOKEN)  # (batch, seq_len)
    # Find first EOS_STEP_TOKEN position
    first_eos_positions = eos_mask.float().argmax(dim=1)  # (batch,)
    # Find last EOS_STEP_TOKEN position
    eos_mask_flipped = torch.flip(eos_mask, dims=[1])  # (batch, seq_len)
    last_eos_positions_flipped = eos_mask_flipped.float().argmax(dim=1)  # (batch,)
    last_eos_positions = seq_len - 1 - last_eos_positions_flipped  # (batch,)
    # Create position indices for comparison
    pos_indices = torch.arange(seq_len-1, device=targets_types.device).unsqueeze(0)  # (1, seq_len-1)
    valid_loss_mask = (pos_indices >= first_eos_positions.unsqueeze(1)) & (pos_indices < last_eos_positions.unsqueeze(1))  # (batch, seq_len-1)
    
    no_eos_found = ~eos_mask.any(dim=1)  # (batch,)
    if no_eos_found.any():
        print("[Warning] No EOS_STEP found in the sequence!")
        valid_loss_mask[no_eos_found, 0] = False
    
    # Type loss (cross-entropy)
    type_loss = F.cross_entropy(
        particle_types_shifted.reshape(-1, particle_types_shifted.size(-1)),
        targets_types_shifted.reshape(-1),
        reduction='none'
    )
    
    # Feature loss (MSE)
    feat_loss = F.mse_loss(
        particle_feats_shifted.reshape(-1, 7),
        targets_feats_shifted.reshape(-1, 7),
        reduction='none'
    ).mean(dim=1)
    
    # Apply both padding mask and valid loss mask
    padding_mask = batch['padding_mask']
    if padding_mask is not None:
        padding_mask_shifted = padding_mask[:, 1:]  # (batch, seq_len-1)
        final_mask = (~padding_mask_shifted) & valid_loss_mask
    else:
        final_mask = valid_loss_mask
    
    # Flatten the mask
    final_mask_flat = final_mask.reshape(-1)
    
    # Zero out invalid positions
    type_loss = type_loss * final_mask_flat.float()
    feat_loss = feat_loss * final_mask_flat.float()
    
    # Average over valid positions
    num_valid = final_mask_flat.sum()
    type_loss = type_loss.sum() / max(num_valid, 1)
    feat_loss = feat_loss.sum() / max(num_valid, 1)
    
    # Combine losses with weights
    total_loss = type_loss + feat_loss
    
    return {
        'total_loss': total_loss,
        'type_loss': type_loss,
        'feat_loss': feat_loss
    }
