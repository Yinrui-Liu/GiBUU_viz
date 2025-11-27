"""
GiBUU Transformer Package

A PyTorch-based transformer model for particle sequence generation from GiBUU simulation data.
"""

__version__ = "1.0.0"
__author__ = ""

from .model import GiBUUTransformer, GiBUUPropagationModel, ParticleEncoder, ParticleDecoder
from .training import GiBUUSeqOfSetsModel, GiBUUPropagationLightning, train_model
from .loss import particle_gpt_loss, gibuu_propagation_loss
from .data_processing import (
    extract_particle_sequences, 
    prepare_sequence_for_training,
    extract_timestep_pairs,
    filter_pairs_with_changes,
    build_gibuu_h5_from_root,
    prepare_interaction_data,
    prepare_propagation_data
)
from .visualization import (
    visualize_particles_with_slider, 
    save_particles_gif, 
    evaluate_model,
    extract_visualization_lists_from_output_sequence
)
from .utils import calculate_feature_statistics, load_feature_statistics, save_sequence_data, load_sequence_data
from .generation import GiBUUEventGenerator

__all__ = [
    "GiBUUTransformer",
    "GiBUUPropagationModel",
    "ParticleEncoder", 
    "ParticleDecoder",
    "GiBUUSeqOfSetsModel",
    "GiBUUPropagationLightning",
    "train_model",
    "particle_gpt_loss",
    "gibuu_propagation_loss",
    "extract_particle_sequences",
    "prepare_sequence_for_training",
    "extract_timestep_pairs",
    "filter_pairs_with_changes",
    "build_gibuu_h5_from_root",
    "prepare_interaction_data",
    "prepare_propagation_data",
    "visualize_particles_with_slider",
    "save_particles_gif",
    "evaluate_model",
    "extract_visualization_lists_from_output_sequence",
    "calculate_feature_statistics",
    "load_feature_statistics",
    "save_sequence_data",
    "load_sequence_data",
    "GiBUUEventGenerator"
]
