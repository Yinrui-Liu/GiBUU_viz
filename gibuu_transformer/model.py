"""
Model architecture components for GiBUU Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import EOS_STEP_TOKEN


def get_mlp(in_features, hidden_features, hidden_layers, out_features, 
            Activation=nn.ReLU, bias=True):
    """
    Return an MLP with fully connected layers.
    """
    dims = [in_features] + [hidden_features] * hidden_layers + [out_features]
    actn = Activation()

    net = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i+1], bias=bias))
        if i < len(dims) - 2:
            net.append(actn)
    return nn.Sequential(*net)


class ParticleEncoder(nn.Module):
    """
    Encoded particle ID -> dense vector embedding
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.embedding = nn.Embedding(**cfg['embedding'])
        self.encoder = get_mlp(**cfg['particle_encoder'])

    def forward(self, encoded_ids, feats):
        """
        The embedding captures categorical identity (particle type)
        The MLP encodes continuous features (kinematics)
        """
        x_out = self.embedding(encoded_ids) + self.encoder(feats)
        return x_out  # (batch_size, seq_len, embedding_dim)


class ParticleDecoder(nn.Module):
    """
    Decode embeddings back to particle types and features.
    """
    def __init__(self, cfg):
        super().__init__()
        
        dec_cfg = cfg['particle_decoder'].copy()
        num_classes = dec_cfg.pop('num_classes')

        # for backward compatibility, simply use linear prediction
        dec_cfg.setdefault('hidden_features', 0)
        dec_cfg.setdefault('hidden_layers', 0)
        
        self.part_type_cls = nn.Linear(dec_cfg['in_features'], num_classes)
        self.part_feat_dec = get_mlp(**dec_cfg)
    
    def forward(self, x):
        part_type = self.part_type_cls(x)
        part_feat = self.part_feat_dec(x)
        return part_type, part_feat


def create_batch_timestep_masks(encoded_ids_batch, num_heads=1):
    """
    Create timestep-based causal masks for a batch of sequences.
    encoded_ids_batch: tensor of shape (batch_size, seq_len)
    Returns: mask of shape (batch_size, seq_len, seq_len) where True means masked
    """
    batch_size, seq_len = encoded_ids_batch.shape
    
    # Find all EOS_STEP_TOKEN positions for each sequence
    eos_mask = (encoded_ids_batch == EOS_STEP_TOKEN)  # (batch_size, seq_len)
    
    # Create cumulative sum and shift by 1
    timestep_for_pos = eos_mask.cumsum(dim=1)  # (batch_size, seq_len)
    zeros = torch.zeros(batch_size, 1, dtype=timestep_for_pos.dtype, device=timestep_for_pos.device)
    timestep_for_pos = torch.cat([zeros, timestep_for_pos[:, :-1]], dim=1)
    
    # Expand timestep arrays for broadcasting
    timestep_i = timestep_for_pos.unsqueeze(2)  # (batch_size, seq_len, 1)
    timestep_j = timestep_for_pos.unsqueeze(1)  # (batch_size, 1, seq_len)
    
    # Create masks for current and previous timestep
    current_timestep_mask = (timestep_j == timestep_i)
    prev_timestep_mask = (timestep_j == timestep_i - 1)
    
    # Combine masks: can attend to current OR previous timestep
    allowed_mask = current_timestep_mask | prev_timestep_mask
    mask = ~allowed_mask
    
    # Expand for multiple attention heads
    mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    mask = mask.reshape(batch_size * num_heads, seq_len, seq_len)

    return mask


class GiBUUTransformer(nn.Module):
    """
    GPT-style Transformer for GiBUU particle sequence generation.
    Input: [encoded_id, x, y, z, E, Px, Py, Pz]
    Autoregressive model that generates particles one by one.
    """
    def __init__(self, cfg):
        super().__init__()
        
        # Particle Encoder + Decoder
        self.particle_encoder = ParticleEncoder(cfg)
        self.particle_decoder = ParticleDecoder(cfg)

        # Transformer Decoder (GPT-style, no encoder)
        layer_cfg = cfg['transformer']['layer']
        decoder_layer = nn.TransformerDecoderLayer(**layer_cfg)
        
        d_model = layer_cfg['d_model']
        decoder_norm = nn.LayerNorm(d_model)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer, norm=decoder_norm,
            **cfg['transformer']['decoder']
        )
        
    def create_causal_mask(self, encoded_ids, device=None):
        """
        Create a mask for each sequence so each token can only attend to tokens 
        in the previous time step and the current time step.
        encoded_ids: tensor of shape (batch_size, seq_len)
        Returns: mask of shape (batch_size, seq_len, seq_len)
        """
        # Standard causal mask
        #seq_len = encoded_ids.shape[1]
        #causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(encoded_ids.device)

        num_heads = self.decoder.layers[0].self_attn.num_heads
        causal_mask = create_batch_timestep_masks(encoded_ids, num_heads)
        return causal_mask
        
    def forward(
        self, encoded_ids, particle_feats, 
        padding_mask=None, causal_mask=None, return_aux=False
    ):
        """
        Forward pass through the GPT-style Transformer.
        
        Parameters:
        -----------
        encoded_ids: tensor
            Encoded particle IDs of shape (batch, seq_len).
        particle_feats: tensor
            Particle features of shape (batch, seq_len, 7) - [x, y, z, E, Px, Py, Pz].
        padding_mask: tensor, optional
            Boolean padding mask of shape (batch, seq_len).
        causal_mask: tensor, optional
            Causal mask for autoregressive attention of shape (seq_len, seq_len).
            If None, will be created automatically.
        return_aux: bool, optional
            Return auxiliary outputs from intermediate layers.
            
        Returns:
        --------
        output: dict
            Dictionary containing:
            - 'embeddings': Final embeddings (batch, seq_len, d_model)
            - 'particle_types': Logits for particle type prediction
            - 'particle_feats': Decoded particle features
            - 'aux_outputs': Auxiliary outputs (if return_aux=True)
        """
        # Encode particles
        embeddings = self.particle_encoder(encoded_ids, particle_feats)
        
        # Create dummy memory for decoder (since we're not using encoder)
        # In GPT-style, the decoder attends to its own input
        memory = embeddings
        
        # Use pre-computed mask if available, otherwise fall back to standard causal mask
        if causal_mask is None:
            causal_mask = self.create_causal_mask(encoded_ids)
        else:
            print("[Info] Non-None causal_mask provided.")
        
        if return_aux:
            # Get auxiliary outputs from intermediate layers
            aux_outputs = self.decode_with_aux(embeddings, memory, padding_mask, causal_mask)
            embeddings = aux_outputs[-1]  # Final layer output
            output = {
                'embeddings': embeddings,
                'aux_outputs': aux_outputs[:-1]  # All but last
            }
        else:
            # Standard forward pass
            embeddings = self.decoder(
                embeddings, memory,
                tgt_key_padding_mask=padding_mask,
                tgt_mask=causal_mask
            )
            output = {'embeddings': embeddings}
        
        # Decode to particle types and features
        particle_types, particle_feats = self.particle_decoder(embeddings)
        output.update({
            'particle_types': particle_types,
            'particle_feats': particle_feats
        })
        
        return output

    def decode_with_aux(self, embeddings, memory, padding_mask=None, causal_mask=None):
        """
        Decoder forward pass with auxiliary outputs from intermediate layers.
        """
        x = embeddings
        aux_outputs = []

        # Create causal mask if not provided
        if causal_mask is None:
            # Use embeddings to determine device and create mask
            batch_size, seq_len = embeddings.shape[:2]
            dummy_encoded_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=embeddings.device)
            causal_mask = self.create_causal_mask(dummy_encoded_ids)
        
        for layer in self.decoder.layers:
            x = layer(
                x, memory,
                tgt_key_padding_mask=padding_mask,
                tgt_mask=causal_mask
            )
            aux_outputs.append(x)
        
        return aux_outputs


class GiBUUPropagationModel(nn.Module):
    """
    Particle propagation model with zero-inflated architecture for predicting feature changes.
    
    This model predicts:
    1. Position deltas (Δx, Δy, Δz) - always predicted
    2. Energy/momentum changes with zero-inflation:
       - Binary flag: "Are ALL E/p components unchanged?" (per particle)
       - If not all zero, predicts all 4 values: ΔKE, ΔPx, ΔPy, ΔPz
    3. Interaction flag: Step-level binary indicating whether interactions will occur
    
    Parameters:
    -----------
    num_particle_types : int
        Size of particle type vocabulary (default: 4096)
    feature_dim : int
        Number of input features per particle (default: 7 for [x,y,z,KE,Px,Py,Pz])
    hidden_dims : list of int
        Hidden layer dimensions for MLP encoder (default: [256, 128, 64])
    dropout : float
        Dropout rate (default: 0.1)
    aggregation_method : str
        Method to aggregate particle representations for step-level prediction
        Options: 'mean', 'max', 'attention' (default: 'mean')
    use_zero_inflation : bool
        If True, uses zero-inflated architecture for E/p predictions
        If False, directly predicts all 7 deltas (simpler model)
        (default: True)
    """
    
    def __init__(
        self,
        num_particle_types=4096,
        feature_dim=7,
        hidden_dims=[256, 128, 64],
        dropout=0.1,
        aggregation_method='mean',
        use_zero_inflation=True
    ):
        super().__init__()
        
        from .constants import FEATS_MEAN, FEATS_SIGMA, FEATS_DELTA_MEAN, FEATS_DELTA_SIGMA
        
        self.feature_dim = feature_dim
        self.aggregation_method = aggregation_method
        self.use_zero_inflation = use_zero_inflation
        
        # Particle type embedding
        self.particle_type_embedding = nn.Embedding(
            num_particle_types, 
            embedding_dim=16,
            padding_idx=None
        )
        
        # Shared encoder
        input_dim = 16 + feature_dim
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*encoder_layers)
        encoder_output_dim = hidden_dims[-1]
        
        if aggregation_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(encoder_output_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        if use_zero_inflation:
            # Zero-inflated architecture
            # Position head (Δx, Δy, Δz)
            self.position_head = nn.Sequential(
                nn.Linear(encoder_output_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 3)
            )
            
            # E/p zero classifier: Single binary per particle
            # "Are ALL 4 E/p components unchanged?"
            self.em_zero_classifier = nn.Sequential(
                nn.Linear(encoder_output_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),  # Single output per particle
                nn.Sigmoid()
            )
            
            # E/p value predictor: If not all zero, predict all 4 values
            self.em_value_predictor = nn.Sequential(
                nn.Linear(encoder_output_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 4)  # 4 values: ΔKE, ΔPx, ΔPy, ΔPz
            )
            
            # Initialize
            nn.init.xavier_uniform_(self.position_head[-1].weight, gain=0.1)
            nn.init.zeros_(self.position_head[-1].bias)
            nn.init.xavier_uniform_(self.em_value_predictor[-1].weight, gain=0.1)
            nn.init.zeros_(self.em_value_predictor[-1].bias)
            
            # Normalization buffers (separate for position and E/p)
            self.register_buffer('position_delta_mean', torch.tensor(FEATS_DELTA_MEAN[:3], dtype=torch.float32))
            self.register_buffer('position_delta_sigma', torch.tensor(FEATS_DELTA_SIGMA[:3], dtype=torch.float32))
            self.register_buffer('em_delta_mean_nonzero', torch.tensor(FEATS_DELTA_MEAN[3:], dtype=torch.float32))
            self.register_buffer('em_delta_sigma_nonzero', torch.tensor(FEATS_DELTA_SIGMA[3:], dtype=torch.float32))
        else:
            # Simple architecture: Direct prediction of all 7 deltas
            self.delta_predictor = nn.Sequential(
                nn.Linear(encoder_output_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 7)  # All 7 deltas: Δx, Δy, Δz, ΔKE, ΔPx, ΔPy, ΔPz
            )
            
            # Initialize
            nn.init.xavier_uniform_(self.delta_predictor[-1].weight, gain=0.1)
            nn.init.zeros_(self.delta_predictor[-1].bias)
            
            # Normalization buffers (all deltas use same stats)
            self.register_buffer('delta_mean', torch.tensor(FEATS_DELTA_MEAN, dtype=torch.float32))
            self.register_buffer('delta_sigma', torch.tensor(FEATS_DELTA_SIGMA, dtype=torch.float32))
        
        # Interaction head (shared for both architectures)
        self.interaction_head = nn.Sequential(
            nn.Linear(encoder_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Feature normalization buffers (shared)
        self.register_buffer('feats_mean', torch.tensor(FEATS_MEAN, dtype=torch.float32))
        self.register_buffer('feats_sigma', torch.tensor(FEATS_SIGMA, dtype=torch.float32))
    
    def forward(self, particle_type, features, padding_mask=None):
        """
        Forward pass.
        
        Parameters:
        -----------
        particle_type : torch.Tensor
            Encoded particle type IDs, shape (batch, num_particles)
        features : torch.Tensor
            Normalized particle features, shape (batch, num_particles, feature_dim)
        padding_mask : torch.Tensor, optional
            Boolean mask for padded particles, shape (batch, num_particles)
            True for padded positions
        
        Returns:
        --------
        dict with keys:
            - 'interaction_prediction': Step-level interaction flag, shape (batch,)
            - If use_zero_inflation=True:
                - 'position_deltas': Position deltas, shape (batch, N, 3)
                - 'em_prob_all_zero': Probability all E/p are zero, shape (batch, N)
                - 'em_delta_values': E/p delta values, shape (batch, N, 4)
                - 'em_deltas': Final E/p deltas (weighted), shape (batch, N, 4)
                - 'predicted_deltas': All 7 deltas, shape (batch, N, 7)
            - If use_zero_inflation=False:
                - 'predicted_deltas': All 7 deltas, shape (batch, N, 7)
        """
        batch_size, num_particles = features.size(0), features.size(1)
        
        # Encode
        type_embedding = self.particle_type_embedding(particle_type)
        combined = torch.cat([type_embedding, features], dim=-1)
        combined_flat = combined.reshape(-1, combined.size(-1))
        encoded_flat = self.shared_encoder(combined_flat)
        encoded = encoded_flat.reshape(batch_size, num_particles, -1)
        
        # Aggregate for interaction prediction
        if self.aggregation_method == 'mean':
            if padding_mask is not None:
                valid_mask = ~padding_mask
                encoded_masked = encoded * valid_mask.unsqueeze(-1).float()
                num_valid = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                step_representation = encoded_masked.sum(dim=1) / num_valid
            else:
                step_representation = encoded.mean(dim=1)
        elif self.aggregation_method == 'max':
            if padding_mask is not None:
                encoded_masked = encoded.clone()
                encoded_masked[padding_mask] = float('-inf')
                step_representation = encoded_masked.max(dim=1)[0]
            else:
                step_representation = encoded.max(dim=1)[0]
        elif self.aggregation_method == 'attention':
            attention_scores = self.attention(encoded)
            if padding_mask is not None:
                attention_scores = attention_scores.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            step_representation = (encoded * attention_weights).sum(dim=1)
        
        interaction_prediction = self.interaction_head(step_representation).squeeze(-1)
        
        if self.use_zero_inflation:
            # Zero-inflated predictions
            # Position deltas
            position_deltas_flat = self.position_head(encoded_flat)
            position_deltas = position_deltas_flat.reshape(batch_size, num_particles, 3)
            
            # E/p: Single binary per particle
            em_prob_all_zero_flat = self.em_zero_classifier(encoded_flat)  # (batch*N, 1)
            em_prob_all_zero = em_prob_all_zero_flat.reshape(batch_size, num_particles)  # (batch, N)
            
            # E/p: All 4 values if not zero
            em_delta_values_flat = self.em_value_predictor(encoded_flat)  # (batch*N, 4)
            em_delta_values = em_delta_values_flat.reshape(batch_size, num_particles, 4)
            
            # Final E/p deltas: weighted by (1 - prob_all_zero)
            em_deltas = (1 - em_prob_all_zero).unsqueeze(-1) * em_delta_values  # (batch, N, 4)
            
            # Combine
            predicted_deltas = torch.cat([position_deltas, em_deltas], dim=-1)
            
            return {
                'interaction_prediction': interaction_prediction,
                'position_deltas': position_deltas,
                'em_prob_all_zero': em_prob_all_zero,
                'em_delta_values': em_delta_values,
                'em_deltas': em_deltas,
                'predicted_deltas': predicted_deltas
            }
        else:
            # Simple direct prediction
            predicted_deltas_flat = self.delta_predictor(encoded_flat)
            predicted_deltas = predicted_deltas_flat.reshape(batch_size, num_particles, 7)
            
            return {
                'interaction_prediction': interaction_prediction,
                'predicted_deltas': predicted_deltas
            }
