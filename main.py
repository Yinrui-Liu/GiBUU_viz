#!/usr/bin/env python3
"""
Main script for GiBUU Transformer training and evaluation.

This script provides a command-line interface for:
1. Data preparation (prep mode)
2. Training the GiBUU Transformer model
3. Evaluating model performance
4. Generating new particle sequences
5. Visualizing results

Usage:
    # Prepare data from H5 file (one-time)
    python main.py --mode prep --data_path /exp/dune/data/users/yinrui/GiBUU/GPTdata/GiBUU_FSI_particles.h5

    # Prepare data from ROOT files (automatically converts to H5)
    python main.py --mode prep --data_path /path/to/root/files/directory

    # Force conversion of ROOT files even if H5 file exists
    python main.py --mode prep --data_path /path/to/root/files/directory --force_convert

    # Train a new model
    python main.py --mode train --config config.json --data_path /exp/dune/data/users/yinrui/GiBUU/GPTdata/GiBUU_FSI_particles.h5

    # Train using preprocessed data (faster)
    python main.py --mode train --config config.json --seqdata_path processed_seqdata.npz

    # Evaluate a trained model
    python main.py --mode eval --checkpoint_path checkpoints/model.ckpt --data_path GiBUU_FSI_particles.h5

    # Generate new sequences with visualization
    python main.py --mode generate --checkpoint_path checkpoints/model.ckpt --data_path GiBUU_FSI_particles.h5 --visualize

    # Plot training curves
    python main.py --mode plot --log_dir lightning_logs/exp_name
"""

import argparse
import os
import glob
import torch
import gc
from pathlib import Path

from gibuu_transformer import (
    extract_particle_sequences, 
    prepare_sequence_for_training,
    GiBUUSeqOfSetsModel,
    train_model,
    evaluate_model,
    visualize_particles_with_slider,
    save_particles_gif,
    extract_visualization_lists_from_output_sequence
)
from gibuu_transformer.utils import save_sequence_data, load_sequence_data, detect_data_format, convert_root_to_h5
from gibuu_transformer.constants import EOS_STEP_TOKEN, EOS_TOKEN, GIBUU_CHARGE_TO_PDG
from gibuu_transformer.data_processing import create_dataloaders, ParticleSequenceDataset
from gibuu_transformer.visualization import load_and_plot_loss_curves


def load_config(config_path):
    """Load configuration from JSON file."""
    import json
    from pathlib import Path
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from {config_path}")
    return config


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint in a directory."""
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    # Sort by modification time (latest)
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0]


def prep_mode(args):
    """Prepare and save sequence data."""
    print("=== Data Preparation Mode ===")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Detect data format and handle ROOT to H5 conversion if needed
    format_type, h5_file_path = detect_data_format(args.data_path)
    
    if format_type == 'unknown':
        raise ValueError(f"Could not detect data format in {args.data_path}. "
                        "Expected either H5 files or ROOT files with pattern 'EventOutput.Pert.00000*.root'")
    
    if format_type == 'root':
        print(f"Detected ROOT files in {args.data_path}")
        print(f"Converting ROOT files to H5 format: {h5_file_path}")
        
        # Check if H5 file already exists
        if Path(h5_file_path).exists():
            print(f"H5 file already exists: {h5_file_path}")
            print("Skipping conversion. Use --force_convert to overwrite.")
            if not getattr(args, 'force_convert', False):
                data_path = h5_file_path
            else:
                print("Force converting ROOT files to H5...")
                convert_root_to_h5(args.data_path, h5_file_path)
                data_path = h5_file_path
        else:
            convert_root_to_h5(args.data_path, h5_file_path)
            data_path = h5_file_path
    else:
        print(f"Detected H5 file: {h5_file_path}")
        data_path = h5_file_path
    
    # Load and process data
    print(f"Loading data from {data_path}...")
    data = extract_particle_sequences(data_path)
    print(f"Loaded {len(data)} events")
    
    # Prepare sequences for training
    print("Preparing sequences for training...")
    stats_path = getattr(args, 'stats_path', None)
    save_stats_path = getattr(args, 'save_stats_path', 'feature_stats.json')
    encoded_ids, particle_feats, padding_mask = prepare_sequence_for_training(
        data, cfg["trainer"]["max_seq_len"], recursive_truncate=True,
        stats_path=stats_path, save_stats_path=save_stats_path
    )
    
    seqdata = {
        "encoded_ids": encoded_ids,
        "particle_feats": particle_feats,
        "padding_mask": padding_mask,
        "causal_mask": None
    }
    
    # Save processed data
    seqdata_path = getattr(args, 'seqdata_path', 'processed_seqdata.npz')
    save_sequence_data(seqdata, seqdata_path)
    
    print("Data preparation completed!")
    return seqdata


def train_mode(args):
    """Train the model."""
    print("=== Training Mode ===")
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Load processed data or prepare from scratch
    seqdata_path = getattr(args, 'seqdata_path', 'processed_seqdata.npz')
    
    if args.seqdata_path and Path(args.seqdata_path).exists():
        print(f"Loading preprocessed data from {seqdata_path}...")
        seqdata = load_sequence_data(seqdata_path)
    else:
        print("No preprocessed data found. Running data preparation...")
        # Call prep_mode to prepare and save data
        seqdata = prep_mode(args)
    
    # Train model
    log_dir = args.log_dir or "lightning_logs"
    exp_name = args.exp_name or "gibuu_transformer"
    
    if args.resume and args.checkpoint_path:
        checkpoint_path = find_best_checkpoint(args.checkpoint_path)
        model, trainer = train_model(
            cfg, seqdata, checkpoint_path, resume_training=True,
            log_dir=log_dir, exp_name=exp_name
        )
    else:
        model, trainer = train_model(
            cfg, seqdata, log_dir=log_dir, exp_name=exp_name
        )
    
    print("Training completed!")
    return model, trainer


def eval_mode(args):
    """Evaluate the model."""
    print("=== Evaluation Mode ===")
    
    # Load model
    if args.checkpoint_path:
        model = GiBUUSeqOfSetsModel.load_from_checkpoint(args.checkpoint_path)
    else:
        raise ValueError("Checkpoint path required for evaluation")
    
    # Load test data
    if args.data_path or args.seqdata_path:
        # Try to load preprocessed data first
        if args.seqdata_path and Path(args.seqdata_path).exists():
            print(f"Loading preprocessed data from {args.seqdata_path}...")
            test_seqdata = load_sequence_data(args.seqdata_path)
        elif args.data_path:
            # Try to load preprocessed test data
            test_seqdata_path = getattr(args, 'test_seqdata_path', 'processed_test_seqdata.npz')
            
            if args.test_seqdata_path and Path(args.test_seqdata_path).exists():
                print(f"Loading preprocessed test data from {test_seqdata_path}...")
                test_seqdata = load_sequence_data(test_seqdata_path)
            else:
                print("No preprocessed data found. Preparing from raw data...")
                print(f"Loading test data from {args.data_path}...")
                test_data = extract_particle_sequences(args.data_path)
                # Use first 1000 events for testing
                test_data = test_data[:1000]
                
                # Prepare test sequences
                stats_path = getattr(args, 'stats_path', 'feature_stats.json')
                encoded_ids, particle_feats, padding_mask = prepare_sequence_for_training(
                    test_data, 1000, recursive_truncate=True, stats_path=stats_path
                )
                
                test_seqdata = {
                    "encoded_ids": encoded_ids,
                    "particle_feats": particle_feats,
                    "padding_mask": padding_mask,
                    "causal_mask": None
                }
        else:
            raise ValueError("No preprocessed data found. Please run 'python main.py --mode prep' first to prepare data.")
        
        # Create test dataloader
        test_dataset = ParticleSequenceDataset(test_seqdata, 1000)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Evaluate
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = evaluate_model(model, test_loader, device)
        
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Feature MSE: {results['feature_mse']:.6f}")
        print(f"Type Loss: {results['type_loss']:.4f}")
        print(f"Total Loss: {results['total_loss']:.4f}")
        
        return results
    else:
        print("No test data provided. Skipping evaluation.")
        return None


def generate_mode(args):
    """Generate new particle sequences."""
    print("=== Generation Mode ===")
    
    # Load model
    if args.checkpoint_path:
        model = GiBUUSeqOfSetsModel.load_from_checkpoint(args.checkpoint_path)
    else:
        raise ValueError("Checkpoint path required for generation")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Load reference data for initial context
    if args.data_path or args.seqdata_path:
        if args.seqdata_path and Path(args.seqdata_path).exists():
            print(f"Loading preprocessed data from {args.seqdata_path}...")
            seqdata = load_sequence_data(args.seqdata_path)
            
            # Use first event as reference
            ievt = args.event_idx or 0
            event_tokens = seqdata['encoded_ids'][ievt]
            event_feats = seqdata['particle_feats'][ievt]
        elif args.data_path:
            print(f"Loading reference data from {args.data_path}...")
            data = extract_particle_sequences(args.data_path)
            
            # Use first event as reference
            ievt = args.event_idx or 0
            event_tokens = torch.tensor([token[1:] for token in data[ievt]], dtype=torch.long)  # Remove time_step
            event_feats = torch.tensor([token[3:] for token in data[ievt]], dtype=torch.float32)  # [x, y, z, E, Px, Py, Pz]
            
            # Normalize features
            from gibuu_transformer.constants import FEATS_MEAN, FEATS_SIGMA
            event_feats = (event_feats - torch.tensor(FEATS_MEAN)) / torch.tensor(FEATS_SIGMA)
        else:
            raise ValueError("No preprocessed data found. Please run 'python main.py --mode prep' first to prepare data.")
        
        # Prepare initial context (first n time steps)
        input_n_step = args.input_steps or 3
        time_step_indices = (event_tokens == EOS_STEP_TOKEN).nonzero(as_tuple=True)[0]
        if len(time_step_indices) < input_n_step:
            raise ValueError(f"Event has fewer than {input_n_step} time steps!")
        cutoff_idx = time_step_indices[input_n_step-1] + 1
        
        tokens_in = event_tokens[:cutoff_idx].unsqueeze(0).to(device)
        feats_in = event_feats[:cutoff_idx].unsqueeze(0).to(device)
        mask_in = torch.zeros_like(tokens_in, dtype=torch.bool).to(device)
        
        print(f"Using first {input_n_step} time steps as context...")
        
        # Generate new sequence
        output_sequence = []
        max_gen_len = args.max_length or 500
        
        print(f"Generating up to {max_gen_len} tokens...")
        
        for step in range(max_gen_len):
            # Forward pass
            output = model(tokens_in, feats_in, mask_in)
            logits = output['particle_types']
            feats_pred = output['particle_feats']
            
            # Get next token and features
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            next_feat = feats_pred[:, -1:, :]
            
            # Append to sequence
            tokens_in = torch.cat([tokens_in, next_token], dim=1)
            feats_in = torch.cat([feats_in, next_feat], dim=1)
            mask_in = torch.cat([mask_in, torch.zeros_like(next_token, dtype=torch.bool)], dim=1)
            
            # Denormalize features
            next_feat_denorm = (next_feat[0, 0] * torch.tensor(FEATS_SIGMA).to(device) + 
                               torch.tensor(FEATS_MEAN).to(device))
            
            # Decode token
            from gibuu_transformer.data_processing import decode_id
            gibuu_id, charge, _ = decode_id(next_token.item())
            
            # Add to output sequence
            output_sequence.append([
                step, gibuu_id, charge,
                next_feat_denorm[0].item(),  # x
                next_feat_denorm[1].item(),  # y
                next_feat_denorm[2].item(),  # z
                next_feat_denorm[3].item(),  # E
                next_feat_denorm[4].item(),  # Px
                next_feat_denorm[5].item(),  # Py
                next_feat_denorm[6].item(),  # Pz
            ])
            
            # Clean up memory
            del output, logits, feats_pred, next_token, next_feat
            if device == "cuda":
                torch.cuda.empty_cache()
            
            if next_token.item() == EOS_TOKEN:
                print("EOS_TOKEN generated. Stopping generation.")
                break
        
        print(f"Generated {len(output_sequence)} particles")
        
        # Save results
        if args.output_path:
            import json
            with open(args.output_path, 'w') as f:
                json.dump(output_sequence, f, indent=2)
            print(f"Generated sequence saved to {args.output_path}")
        
        # Create visualization
        if args.visualize:
            positions_list, momenta_list, pdg_codes_list, timestep_list = extract_visualization_lists_from_output_sequence(
                output_sequence, GIBUU_CHARGE_TO_PDG
            )
            
            # Save GIF
            gif_path = args.output_path.replace('.json', '.gif') if args.output_path else 'generated_particles.gif'
            save_particles_gif(
                positions_list, momenta_list, pdg_codes_list,
                timestep_list=timestep_list,
                filename=gif_path,
                xlim=[-10, 10], ylim=[-10, 10], zlim=[-10, 10],
                fps=5
            )
            print(f"Visualization saved to {gif_path}")
        
        return output_sequence
    else:
        print("No reference data provided. Cannot generate sequences.")
        return None


def plot_mode(args):
    """Plot training curves."""
    print("=== Plot Mode ===")
    
    if not args.log_dir:
        raise ValueError("Log directory required for plotting")
    
    df = load_and_plot_loss_curves(args.log_dir)
    if df is not None:
        print("Loss curves plotted successfully!")
    return df


def main():
    parser = argparse.ArgumentParser(description="GiBUU Transformer")
    parser.add_argument("--mode", choices=["prep", "train", "eval", "generate", "plot"], 
                       required=True, help="Mode to run")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to H5 data file or directory containing ROOT files")
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration JSON file")
    parser.add_argument("--force_convert", action="store_true", 
                       help="Force conversion of ROOT files to H5 even if H5 file already exists")
    
    # Training arguments
    parser.add_argument("--log_dir", type=str, default="lightning_logs", 
                       help="Directory for logging")
    parser.add_argument("--exp_name", type=str, default="gibuu_transformer",
                       help="Experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--stats_path", type=str, help="Path to load pre-computed feature statistics")
    parser.add_argument("--save_stats_path", type=str, default="feature_stats.json",
                       help="Path to save computed feature statistics")
    parser.add_argument("--seqdata_path", type=str, default="processed_seqdata.npz",
                       help="Path to load/save processed sequence data")
    parser.add_argument("--test_seqdata_path", type=str, default="processed_test_seqdata.npz",
                       help="Path to load/save processed test sequence data")
    
    # Generation arguments
    parser.add_argument("--event_idx", type=int, default=0, 
                       help="Event index to use as reference for generation")
    parser.add_argument("--input_steps", type=int, default=3,
                       help="Number of initial time steps to use as context")
    parser.add_argument("--max_length", type=int, default=500,
                       help="Maximum length of generated sequence")
    parser.add_argument("--output_path", type=str, help="Path to save generated sequence")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == "prep":
        prep_mode(args)
    elif args.mode == "train":
        train_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "generate":
        generate_mode(args)
    elif args.mode == "plot":
        plot_mode(args)


if __name__ == "__main__":
    main()
