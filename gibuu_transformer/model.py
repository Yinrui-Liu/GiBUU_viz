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
