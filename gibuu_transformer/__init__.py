"""
GiBUU Transformer Package

A PyTorch-based transformer model for particle sequence generation from GiBUU simulation data.
"""

__version__ = "1.0.0"
__author__ = ""

from .model import GiBUUTransformer, ParticleEncoder, ParticleDecoder
from .training import GiBUUSeqOfSetsModel, train_model
from .loss import particle_gpt_loss
from .data_processing import extract_particle_sequences, prepare_sequence_for_training
from .visualization import (
    visualize_particles_with_slider, 
    save_particles_gif, 
    evaluate_model,
    extract_visualization_lists_from_output_sequence
)
from .utils import calculate_feature_statistics, load_feature_statistics, save_sequence_data, load_sequence_data

__all__ = [
    "GiBUUTransformer",
    "ParticleEncoder", 
    "ParticleDecoder",
    "GiBUUSeqOfSetsModel",
    "train_model",
    "particle_gpt_loss",
    "extract_particle_sequences",
    "prepare_sequence_for_training",
    "visualize_particles_with_slider",
    "save_particles_gif",
    "evaluate_model",
    "extract_visualization_lists_from_output_sequence",
    "calculate_feature_statistics",
    "load_feature_statistics",
    "save_sequence_data",
    "load_sequence_data"
]
