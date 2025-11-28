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


def gibuu_propagation_loss(
    output, 
    input_features,
    target_features, 
    target_interaction, 
    padding_mask=None,
    feature_loss_weight=1.0,
    position_loss_weight=1.0,
    em_zero_loss_weight=1.0,
    em_value_loss_weight=1.0,
    interaction_loss_weight=1.0,
    apply_feature_loss_only_when_no_interaction=True,
    pos_weight=None,
    use_huber_loss=True,
    huber_delta=1.0,
    zero_threshold=1e-10,
    use_zero_inflation=True
):
    """
    Loss function for GiBUU propagation model with optional zero-inflated E/p prediction.
    
    Loss Components:
    ----------------
    1. Position loss: Huber loss on Δx, Δy, Δz
    2. E/p zero classification loss: BCE on whether ALL E/p are zero (if zero-inflated)
    3. E/p value regression loss: Huber loss on non-zero E/p values (if zero-inflated)
    4. Interaction loss: BCE on step-level interaction flag
    
    If use_zero_inflation=False, components 2 and 3 are replaced by direct delta prediction loss.
    
    Total loss = feature_loss_weight × (position_loss_weight × pos_loss 
                                       + em_zero_loss_weight × em_zero_loss
                                       + em_value_loss_weight × em_value_loss)
                + interaction_loss_weight × interaction_loss
    
    Parameters:
    -----------
    output : dict
        Model output with 'interaction_prediction' and either:
        - If zero-inflated: 'position_deltas', 'em_prob_all_zero', 'em_delta_values'
        - If simple: 'predicted_deltas'
    input_features : torch.Tensor
        Normalized input features, shape (batch, N, 7)
    target_features : torch.Tensor
        Normalized target features, shape (batch, N, 7)
    target_interaction : torch.Tensor
        Binary interaction flags, shape (batch,)
    padding_mask : torch.Tensor, optional
        Boolean mask for padded particles
    feature_loss_weight : float
        Weight for combined feature loss
    position_loss_weight : float
        Weight for position loss component
    em_zero_loss_weight : float
        Weight for E/p zero classification loss
    em_value_loss_weight : float
        Weight for E/p value regression loss
    interaction_loss_weight : float
        Weight for interaction loss
    apply_feature_loss_only_when_no_interaction : bool
        If True, only apply feature losses on non-interaction steps
    pos_weight : float, optional
        Positive class weight for interaction loss (for imbalanced data)
    use_huber_loss : bool
        If True, use Huber loss for regression, else MSE
    huber_delta : float
        Delta parameter for Huber loss
    zero_threshold : float
        Threshold for determining if E/p values are zero
    use_zero_inflation : bool
        If True, use zero-inflated loss; if False, use simple direct loss
        
    Returns:
    --------
    dict with keys:
        - 'total_loss': Combined loss
        - 'feature_loss': Combined feature loss
        - 'position_loss': Position delta loss
        - 'em_zero_loss': E/p zero classification loss (0 if not zero-inflated)
        - 'em_value_loss': E/p value regression loss (0 if not zero-inflated)
        - 'interaction_loss': Interaction prediction loss
    """
    from .constants import FEATS_MEAN, FEATS_SIGMA, FEATS_DELTA_MEAN, FEATS_DELTA_SIGMA
    
    interaction_prediction = output['interaction_prediction']  # (batch,)
    device = interaction_prediction.device
    
    # Get normalization constants
    feats_mean = torch.tensor(FEATS_MEAN, device=device, dtype=torch.float32)
    feats_sigma = torch.tensor(FEATS_SIGMA, device=device, dtype=torch.float32)
    
    # ===================================================================
    # 1. Interaction loss (per step)
    # ===================================================================
    if pos_weight is not None:
        target_float = target_interaction.float()
        if isinstance(pos_weight, (int, float)):
            pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=torch.float32)
        else:
            pos_weight_tensor = pos_weight.to(device)
        
        loss_positive = -target_float * torch.log(interaction_prediction + 1e-8) * pos_weight_tensor
        loss_negative = -(1 - target_float) * torch.log(1 - interaction_prediction + 1e-8)
        interaction_loss = (loss_positive + loss_negative).mean()
    else:
        interaction_loss = F.binary_cross_entropy(interaction_prediction, target_interaction.float())
    
    # ===================================================================
    # 2. Feature losses (only on non-interaction steps)
    # ===================================================================
    if apply_feature_loss_only_when_no_interaction:
        no_interaction_mask = (target_interaction == 0)
        
        if no_interaction_mask.sum() > 0:
            # Compute target deltas (unnormalized)
            input_features_unnorm = input_features * feats_sigma + feats_mean
            target_features_unnorm = target_features * feats_sigma + feats_mean
            target_deltas_unnorm = target_features_unnorm - input_features_unnorm
            
            if use_zero_inflation:
                # Zero-inflated loss
                position_deltas = output['position_deltas']  # (batch, N, 3)
                em_prob_all_zero = output['em_prob_all_zero']  # (batch, N)
                em_delta_values = output['em_delta_values']  # (batch, N, 4)
                
                # Use separate normalization stats
                position_delta_mean = torch.tensor(FEATS_DELTA_MEAN[:3], device=device, dtype=torch.float32)
                position_delta_sigma = torch.tensor(FEATS_DELTA_SIGMA[:3], device=device, dtype=torch.float32)
                em_delta_mean = torch.tensor(FEATS_DELTA_MEAN[3:], device=device, dtype=torch.float32)
                em_delta_sigma = torch.tensor(FEATS_DELTA_SIGMA[3:], device=device, dtype=torch.float32)
                
                # Split target deltas
                target_position_deltas_unnorm = target_deltas_unnorm[:, :, :3]
                target_em_deltas_unnorm = target_deltas_unnorm[:, :, 3:]  # (batch, N, 4)
                
                # Normalize separately
                target_position_deltas_norm = (target_position_deltas_unnorm - position_delta_mean) / position_delta_sigma
                target_em_deltas_norm = (target_em_deltas_unnorm - em_delta_mean) / em_delta_sigma
                
                # ------------------------------------------------------------
                # 2a. Position loss
                # ------------------------------------------------------------
                position_diff = position_deltas - target_position_deltas_norm
                
                if use_huber_loss:
                    position_loss_per_particle = F.huber_loss(
                        position_diff, 
                        torch.zeros_like(position_diff),
                        reduction='none',
                        delta=huber_delta
                    ).mean(dim=-1)
                else:
                    position_loss_per_particle = (position_diff ** 2).mean(dim=-1)
                
                # ------------------------------------------------------------
                # 2b. E/p zero classification loss
                # ------------------------------------------------------------
                # Target: Are ALL 4 E/p components zero? (single binary per particle)
                em_all_components_zero = (target_em_deltas_unnorm.abs() < zero_threshold).all(dim=-1).float()  # (batch, N)
                
                # BCE loss on "all zero" classification
                em_zero_loss_per_particle = F.binary_cross_entropy(
                    em_prob_all_zero,  # (batch, N)
                    em_all_components_zero,  # (batch, N)
                    reduction='none'
                )  # (batch, N)
                
                # ------------------------------------------------------------
                # 2c. E/p value regression loss
                # ------------------------------------------------------------
                # Apply loss to ALL particles where NOT all components are zero
                em_has_nonzero = ~em_all_components_zero.bool()  # (batch, N)
                
                if em_has_nonzero.any():
                    # Compute loss on predicted values vs target (normalized with NON-ZERO stats)
                    em_value_diff = em_delta_values - target_em_deltas_norm  # (batch, N, 4)
                    
                    if use_huber_loss:
                        em_value_loss_per_component = F.huber_loss(
                            em_value_diff,
                            torch.zeros_like(em_value_diff),
                            reduction='none',
                            delta=huber_delta
                        )  # (batch, N, 4)
                    else:
                        em_value_loss_per_component = em_value_diff ** 2
                    
                    # Average over 4 components
                    em_value_loss_all_particles = em_value_loss_per_component.mean(dim=-1)  # (batch, N)
                    
                    # Only apply where target has non-zero components
                    em_value_loss_per_particle = em_value_loss_all_particles * em_has_nonzero.float()
                else:
                    em_value_loss_per_particle = torch.zeros_like(position_loss_per_particle)
                
                # ------------------------------------------------------------
                # Aggregate losses with padding mask
                # ------------------------------------------------------------
                if padding_mask is not None:
                    mask_float = (~padding_mask).float()
                    position_loss_per_particle = position_loss_per_particle * mask_float
                    em_zero_loss_per_particle = em_zero_loss_per_particle * mask_float
                    em_value_loss_per_particle = em_value_loss_per_particle * mask_float
                    
                    # Position: average over all valid particles
                    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
                    position_loss_per_step = position_loss_per_particle.sum(dim=1) / num_valid.squeeze(-1)
                    
                    # E/M zero: average over all valid particles
                    em_zero_loss_per_step = em_zero_loss_per_particle.sum(dim=1) / num_valid.squeeze(-1)
                    
                    # E/M value: average over non-zero particles ONLY
                    em_has_nonzero_masked = em_has_nonzero.float() * mask_float  # (batch, N)
                    num_nonzero_per_step = em_has_nonzero_masked.sum(dim=1).clamp(min=1.0)  # (batch,)
                    em_value_loss_per_step = em_value_loss_per_particle.sum(dim=1) / num_nonzero_per_step
                else:
                    position_loss_per_step = position_loss_per_particle.mean(dim=1)
                    em_zero_loss_per_step = em_zero_loss_per_particle.mean(dim=1)
                    
                    # E/M value: average over non-zero particles ONLY
                    num_nonzero_per_step = em_has_nonzero.float().sum(dim=1).clamp(min=1.0)
                    em_value_loss_per_step = em_value_loss_per_particle.sum(dim=1) / num_nonzero_per_step
                
                # Only average over non-interaction steps
                position_loss = (position_loss_per_step * no_interaction_mask.float()).sum() / no_interaction_mask.sum()
                em_zero_loss = (em_zero_loss_per_step * no_interaction_mask.float()).sum() / no_interaction_mask.sum()
                em_value_loss = (em_value_loss_per_step * no_interaction_mask.float()).sum() / no_interaction_mask.sum()
            else:
                # Simple direct delta prediction loss
                predicted_deltas = output['predicted_deltas']  # (batch, N, 7)
                
                # Use single normalization stats for all deltas
                delta_mean = torch.tensor(FEATS_DELTA_MEAN, device=device, dtype=torch.float32)
                delta_sigma = torch.tensor(FEATS_DELTA_SIGMA, device=device, dtype=torch.float32)
                
                # Normalize target deltas
                target_deltas_norm = (target_deltas_unnorm - delta_mean) / delta_sigma
                
                # Compute loss
                delta_diff = predicted_deltas - target_deltas_norm
                
                if use_huber_loss:
                    loss_per_particle = F.huber_loss(
                        delta_diff,
                        torch.zeros_like(delta_diff),
                        reduction='none',
                        delta=huber_delta
                    ).mean(dim=-1)  # Average over 7 features
                else:
                    loss_per_particle = (delta_diff ** 2).mean(dim=-1)
                
                # Apply padding mask
                if padding_mask is not None:
                    mask_float = (~padding_mask).float()
                    loss_per_particle = loss_per_particle * mask_float
                    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
                    loss_per_step = loss_per_particle.sum(dim=1) / num_valid.squeeze(-1)
                else:
                    loss_per_step = loss_per_particle.mean(dim=1)
                
                # Only average over non-interaction steps
                position_loss = (loss_per_step * no_interaction_mask.float()).sum() / no_interaction_mask.sum()
                em_zero_loss = torch.tensor(0.0, device=device)
                em_value_loss = torch.tensor(0.0, device=device)
        else:
            position_loss = torch.tensor(0.0, device=device)
            em_zero_loss = torch.tensor(0.0, device=device)
            em_value_loss = torch.tensor(0.0, device=device)
    else:
        raise NotImplementedError("Loss for all steps not implemented")
    
    # Combined feature loss
    feature_loss = (position_loss_weight * position_loss + 
                   em_zero_loss_weight * em_zero_loss + 
                   em_value_loss_weight * em_value_loss)
    
    # Total loss
    total_loss = (feature_loss_weight * feature_loss + 
                 interaction_loss_weight * interaction_loss)
    
    return {
        'total_loss': total_loss,
        'feature_loss': feature_loss,
        'position_loss': position_loss,
        'em_zero_loss': em_zero_loss,
        'em_value_loss': em_value_loss,
        'interaction_loss': interaction_loss
    }


def gibuu_interaction_loss(output, batch, type_loss_weight=1.0, feat_loss_weight=1.0):
    """
    Loss function for time step sequence prediction (encoder-decoder with teacher forcing).
    
    Following standard translation model practice:
    - Decoder input: [START, P1, P2, ..., PN, EOS] (includes EOS, length = N+2)
    - Model predictions: [pred_after_START, pred_after_P1, ..., pred_after_PN, pred_after_EOS]
    - Targets: [P1, P2, ..., PN, EOS] = decoder_input[:, 1:] (shifted by 1)
    
    Parameters:
    -----------
    output: dict
        Model output with 'particle_types', 'particle_feats'
    batch: dict
        Contains 'output_encoded_ids' (decoder input), 'output_particle_feats', 'output_padding_mask'
    type_loss_weight: float
        Weight for particle type classification loss
    feat_loss_weight: float
        Weight for particle feature regression loss
        
    Returns:
    --------
    loss: dict
        Dictionary with 'total_loss', 'type_loss', 'feat_loss'
    """
    particle_types = output['particle_types']  # (batch, seq_len, num_classes)
    particle_feats = output['particle_feats']  # (batch, seq_len, 7)
    
    # Decoder input: [START, P1, P2, ..., PN, EOS] (includes EOS)
    decoder_input_ids = batch['output_encoded_ids']  # (batch, seq_len)
    decoder_input_feats = batch['output_particle_feats']  # (batch, seq_len, 7)
    output_padding_mask = batch['output_padding_mask']  # (batch, seq_len)
    
    # Compute targets by shifting decoder input (remove START token)
    target_types = decoder_input_ids[:, 1:]  # (batch, seq_len-1)
    target_feats = decoder_input_feats[:, 1:]  # (batch, seq_len-1, 7)
    target_padding_mask = output_padding_mask[:, 1:]  # (batch, seq_len-1)
    valid_mask = ~target_padding_mask  # (batch, seq_len-1) - True for valid positions
    
    # Align predictions with targets
    particle_types_aligned = particle_types[:, :target_types.size(1)]  # (batch, seq_len-1, num_classes)
    particle_feats_aligned = particle_feats[:, :target_feats.size(1)]  # (batch, seq_len-1, 7)
    
    # Type loss (cross-entropy) - includes EOS
    type_loss = F.cross_entropy(
        particle_types_aligned.reshape(-1, particle_types_aligned.size(-1)),
        target_types.reshape(-1),
        reduction='none'
    )
    type_loss = type_loss.reshape(particle_types_aligned.shape[:2])
    
    # Apply padding mask (EOS is included in type loss, PAD positions are excluded)
    type_loss = (type_loss * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)
    
    # Feature loss (MSE) - exclude EOS (only on particle positions)
    eos_positions = (target_types == EOS_STEP_TOKEN)  # (batch, seq_len-1)
    valid_feat_mask = valid_mask & (~eos_positions)  # Valid positions AND not EOS
    
    feat_loss = F.mse_loss(
        particle_feats_aligned.reshape(-1, 7),
        target_feats.reshape(-1, 7),
        reduction='none'
    ).mean(dim=1)
    feat_loss = feat_loss.reshape(particle_feats_aligned.shape[:2])
    
    # Apply mask (exclude EOS from feature loss)
    feat_loss = (feat_loss * valid_feat_mask.float()).sum() / valid_feat_mask.sum().clamp(min=1)
    
    # Combine losses
    total_loss = (
        type_loss_weight * type_loss +
        feat_loss_weight * feat_loss
    )
    
    return {
        'total_loss': total_loss,
        'type_loss': type_loss,
        'feat_loss': feat_loss
    }
