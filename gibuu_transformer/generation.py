"""
Event generation using trained GiBUU models.

This module provides the GiBUUEventGenerator class for generating physics events
using trained interaction and/or propagation models.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import numpy as np

from .data_processing import encode_id, decode_id
from .constants import EOS_STEP_TOKEN, PAD_TOKEN, START_TOKEN, FEATS_MEAN, FEATS_SIGMA


class GiBUUEventGenerator:
    """
    Event generator that uses interaction and/or propagation models.
    
    Workflow (with both models):
    1. Feed current step to propagation model
    2. Check interaction prediction (0 or 1)
    3a. If 0: Use propagation model's predicted features
    3b. If 1: Switch to interaction model for new particles
    4. Continue to next time step
    
    Workflow (interaction-only mode):
    1. Feed initial step to interaction model
    2. Generate final step directly (autoregressive)
    3. Position features can be ignored if needed
    """
    
    def __init__(
        self,
        interaction_model_path: str,
        interaction_model_class,
        propagation_model_path: Optional[str] = None,
        propagation_model_class = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        interaction_threshold: float = 0.5,
        feats_mean: Optional[List[float]] = None,
        feats_sigma: Optional[List[float]] = None
    ):
        """
        Initialize generator with trained models.
        
        Parameters:
        -----------
        interaction_model_path: str
            Path to interaction model checkpoint
        interaction_model_class: class
            Lightning module class for interaction model
        propagation_model_path: str, optional
            Path to propagation model checkpoint (None for interaction-only mode)
        propagation_model_class: class, optional
            Lightning module class for propagation model
        device: str
            Device to run models on ('cuda' or 'cpu')
        interaction_threshold: float
            Threshold for interaction prediction (default: 0.5)
        feats_mean: List[float], optional
            Feature normalization means (if None, uses FEATS_MEAN from constants)
        feats_sigma: List[float], optional
            Feature normalization stds (if None, uses FEATS_SIGMA from constants)
        """
        self.device = torch.device(device)
        self.interaction_threshold = interaction_threshold
        self.interaction_only_mode = (propagation_model_path is None)
        
        print(f"Loading models on device: {self.device}")
        print(f"Mode: {'Interaction-only' if self.interaction_only_mode else 'Interaction + Propagation'}")
        
        # Load interaction model
        print(f"Loading interaction model from: {interaction_model_path}")
        self.interaction_model = interaction_model_class.load_from_checkpoint(
            interaction_model_path,
            map_location=self.device
        )
        self.interaction_model.eval()
        self.interaction_model.to(self.device)
        print("✓ Interaction model loaded")
        
        # Load propagation model (optional)
        self.propagation_model = None
        if not self.interaction_only_mode:
            print(f"Loading propagation model from: {propagation_model_path}")
            self.propagation_model = propagation_model_class.load_from_checkpoint(
                propagation_model_path,
                map_location=self.device
            )
            self.propagation_model.eval()
            self.propagation_model.to(self.device)
            print("✓ Propagation model loaded")
        
        # Normalization constants
        # Use provided values, or fall back to constants module
        if feats_mean is not None and feats_sigma is not None:
            self.feats_mean = torch.tensor(feats_mean, device=self.device, dtype=torch.float32)
            self.feats_sigma = torch.tensor(feats_sigma, device=self.device, dtype=torch.float32)
        elif FEATS_MEAN is not None and FEATS_SIGMA is not None:
            self.feats_mean = torch.tensor(FEATS_MEAN, device=self.device, dtype=torch.float32)
            self.feats_sigma = torch.tensor(FEATS_SIGMA, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(
                "Feature normalization constants not provided. "
                "Either pass feats_mean/feats_sigma or set FEATS_MEAN/FEATS_SIGMA in constants.py"
            )
        
        print("\n✓ Generator initialized successfully!")
    
    def normalize_features(self, features):
        """Normalize features using training statistics."""
        return (features - self.feats_mean) / self.feats_sigma
    
    def denormalize_features(self, normalized_features):
        """Denormalize features back to original scale."""
        return normalized_features * self.feats_sigma + self.feats_mean
    
    def prepare_step_input(self, particles):
        """
        Prepare particles for model input.
        
        Parameters:
        -----------
        particles: List of tuples
            Each tuple: (gibuu_id, charge, x, y, z, m, E, Px, Py, Pz)
            
        Returns:
        --------
        encoded_ids: torch.Tensor (1, num_particles)
        features: torch.Tensor (1, num_particles, 7)
        """
        encoded_ids = []
        features = []
        
        for particle in particles:
            gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = particle
            
            # Encode particle ID
            encoded_id = encode_id(gibuu_id, charge)
            encoded_ids.append(encoded_id)
            
            # Features: [x, y, z, KE, Px, Py, Pz] where KE = E - m
            feat = torch.tensor([x, y, z, E-m, Px, Py, Pz], dtype=torch.float32)
            features.append(feat)
        
        # Convert to tensors and add batch dimension
        encoded_ids = torch.tensor(encoded_ids, dtype=torch.long).unsqueeze(0)  # (1, N)
        features = torch.stack(features).unsqueeze(0)  # (1, N, 7)
        
        # Normalize features
        features = self.normalize_features(features)
        
        return encoded_ids.to(self.device), features.to(self.device)
    
    @torch.no_grad()
    def generate_step(self, current_particles):
        """
        Generate next time step given current particles.
        
        Parameters:
        -----------
        current_particles: List of tuples
            Each tuple: (gibuu_id, charge, x, y, z, m, E, Px, Py, Pz)
            
        Returns:
        --------
        next_particles: List of tuples
            Particles at next time step
        interaction_occurred: bool
            Whether an interaction occurred
        interaction_prob: float
            Interaction probability from propagation model (None if interaction-only mode)
        """
        if self.interaction_only_mode:
            # In interaction-only mode, always use interaction model
            return self._generate_with_interaction_only(current_particles)
        
        # Prepare input
        encoded_ids, features = self.prepare_step_input(current_particles)
        
        # Step 1: Feed to propagation model
        prop_output = self.propagation_model(
            particle_type=encoded_ids,
            features=features,
            padding_mask=None
        )
        
        interaction_prob = prop_output['interaction_prediction'].item()
        interaction_occurred = interaction_prob > self.interaction_threshold
        
        # Step 2: Branch based on interaction prediction
        if not interaction_occurred:
            # No interaction: Use propagation model's predicted features
            predicted_features = prop_output['predicted_features'][0]  # (N, 7)
            
            # Denormalize features
            predicted_features = self.denormalize_features(predicted_features)
            
            # Build output particles (same types, new features)
            next_particles = []
            for i, (gibuu_id, charge, x, y, z, m, E_old, Px_old, Py_old, Pz_old) in enumerate(current_particles):
                x_new, y_new, z_new, KE_new, Px_new, Py_new, Pz_new = predicted_features[i].cpu().numpy()
                E_new = KE_new + m  # Reconstruct E from KE and m
                next_particles.append((gibuu_id, charge, x_new, y_new, z_new, m, E_new, Px_new, Py_new, Pz_new))
        
        else:
            # Interaction: Use interaction model to predict new particles
            # Add EOS token to encoder input
            eos_token = torch.tensor([[EOS_STEP_TOKEN]], dtype=torch.long, device=self.device)
            eos_feat = torch.zeros((1, 1, 7), dtype=torch.float32, device=self.device)
            eos_feat = self.normalize_features(eos_feat)  # Normalize EOS features
            
            encoder_ids = torch.cat([encoded_ids, eos_token], dim=1)  # (1, N+1)
            encoder_feats = torch.cat([features, eos_feat], dim=1)  # (1, N+1, 7)
            
            # Generate output particles autoregressively
            next_particles = self._generate_with_interaction_model(
                encoder_ids, encoder_feats, current_particles
            )
        
        return next_particles, interaction_occurred, interaction_prob
    
    @torch.no_grad()
    def _generate_with_interaction_only(self, current_particles):
        """
        Generate using only interaction model (for interaction-only mode).
        
        Parameters:
        -----------
        current_particles: List of tuples
            Current particles
            
        Returns:
        --------
        next_particles: List of tuples
            Generated particles
        interaction_occurred: bool
            Always True in this mode
        interaction_prob: None
            None in interaction-only mode
        """
        encoded_ids, features = self.prepare_step_input(current_particles)
        
        # Add EOS token to encoder input
        eos_token = torch.tensor([[EOS_STEP_TOKEN]], dtype=torch.long, device=self.device)
        eos_feat = torch.zeros((1, 1, 7), dtype=torch.float32, device=self.device)
        eos_feat = self.normalize_features(eos_feat)
        
        encoder_ids = torch.cat([encoded_ids, eos_token], dim=1)
        encoder_feats = torch.cat([features, eos_feat], dim=1)
        
        # Generate particles
        next_particles = self._generate_with_interaction_model(
            encoder_ids, encoder_feats, current_particles
        )
        
        return next_particles, True, None
    
    @torch.no_grad()
    def _generate_with_interaction_model(
        self,
        encoder_ids,
        encoder_feats,
        current_particles,
        max_particles=64
    ):
        """
        Generate particles using interaction model (autoregressive).
        
        Parameters:
        -----------
        encoder_ids: torch.Tensor (1, input_len)
            Encoded input particle IDs (includes EOS)
        encoder_feats: torch.Tensor (1, input_len, 7)
            Input particle features (includes EOS)
        current_particles: List of tuples
            Current particles (for mass lookup)
        max_particles: int
            Maximum number of particles to generate
            
        Returns:
        --------
        particles: List of tuples
            Generated particles
        """
        # Initialize decoder sequence with START token
        decoder_ids = torch.tensor([[START_TOKEN]], dtype=torch.long, device=self.device)
        decoder_feats = torch.zeros((1, 1, 7), dtype=torch.float32, device=self.device)
        decoder_feats = self.normalize_features(decoder_feats)
        
        generated_particles = []
        
        # Autoregressive generation
        for step in range(max_particles):
            # Forward pass through model
            output = self.interaction_model(
                input_encoded_ids=encoder_ids,
                input_particle_feats=encoder_feats,
                input_padding_mask=None,
                output_encoded_ids=decoder_ids,
                output_particle_feats=decoder_feats,
                output_padding_mask=None
            )
            
            # Get predictions for last position
            particle_type_logits = output['particle_types'][0, -1]  # (num_classes,)
            particle_feat = output['particle_feats'][0, -1]  # (7,)
            
            # Sample particle type (greedy decoding)
            pred_encoded_id = torch.argmax(particle_type_logits).item()
            
            # Check if EOS token
            if pred_encoded_id == EOS_STEP_TOKEN:
                break
            
            # Decode particle ID
            gibuu_id, charge = decode_id(pred_encoded_id)
            
            # Denormalize features
            feat_denorm = self.denormalize_features(particle_feat.unsqueeze(0))[0]
            x, y, z, KE, Px, Py, Pz = feat_denorm.cpu().numpy()
            
            # Get mass (lookup from particle properties or use a default)
            # TODO: Add proper particle database lookup
            m = 0.938  # Placeholder (proton mass)
            E = KE + m
            
            generated_particles.append((gibuu_id, charge, x, y, z, m, E, Px, Py, Pz))
            
            # Append to decoder sequence for next step
            next_id = torch.tensor([[pred_encoded_id]], dtype=torch.long, device=self.device)
            next_feat = particle_feat.unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
            
            decoder_ids = torch.cat([decoder_ids, next_id], dim=1)
            decoder_feats = torch.cat([decoder_feats, next_feat], dim=1)
        
        return generated_particles
    
    def generate_event(
        self,
        initial_particles,
        num_steps=10,
        verbose=True
    ):
        """
        Generate a complete event starting from initial particles.
        
        Parameters:
        -----------
        initial_particles: List of tuples
            Initial particles at t=0
        num_steps: int
            Number of time steps to generate
        verbose: bool
            Whether to print progress
            
        Returns:
        --------
        event: Dict
            Dictionary containing:
            - 'particles': List of particle lists (one per time step)
            - 'interactions': List of booleans (whether interaction occurred)
            - 'interaction_probs': List of floats (interaction probabilities, None if interaction-only)
        """
        event = {
            'particles': [initial_particles],
            'interactions': [],
            'interaction_probs': []
        }
        
        current_particles = initial_particles
        
        for step in range(num_steps):
            if verbose:
                print(f"\nStep {step+1}/{num_steps}:")
                print(f"  Current particles: {len(current_particles)}")
            
            # Generate next step
            next_particles, interaction_occurred, interaction_prob = self.generate_step(
                current_particles
            )
            
            if verbose:
                if interaction_prob is not None:
                    print(f"  Interaction prob: {interaction_prob:.4f}")
                print(f"  Interaction occurred: {interaction_occurred}")
                print(f"  Next particles: {len(next_particles)}")
            
            # Store results
            event['particles'].append(next_particles)
            event['interactions'].append(interaction_occurred)
            event['interaction_probs'].append(interaction_prob)
            
            # Update current particles
            current_particles = next_particles
            
            # Stop if no particles remain
            if len(current_particles) == 0:
                if verbose:
                    print("  No particles remaining - stopping generation")
                break
        
        return event

