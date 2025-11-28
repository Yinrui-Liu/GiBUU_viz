"""
Utility functions for GiBUU Transformer.
"""

from pathlib import Path


def calculate_feature_statistics(raw_sequences, save_path=None):
    """
    Calculate mean and standard deviation for feature normalization from raw particle sequences.
    
    Parameters:
    -----------
    raw_sequences: List of lists
        Each inner list is a sequence of particle tokens: [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
    save_path: str, optional
        Path to save the calculated statistics as JSON file
        
    Returns:
    --------
    feats_mean: list
        Mean values for [x, y, z, KE, Px, Py, Pz] where KE = E - m
    feats_sigma: list
        Standard deviation values for [x, y, z, KE, Px, Py, Pz]
    """
    import numpy as np
    import json
    
    all_features = []
    
    for sequence in raw_sequences:
        for token in sequence:
            # token = [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
            # Extract features: [x, y, z, KE, Px, Py, Pz] where KE = E - m
            x, y, z, m, E, Px, Py, Pz = token[3:11]  # Skip ts, gibuuID, chg
            KE = E - m  # Calculate kinetic energy
            features = [x, y, z, KE, Px, Py, Pz]
            all_features.append(features)
    
    # Convert to numpy array for easier computation
    all_features = np.array(all_features)
    
    # Calculate mean and standard deviation
    feats_mean = np.mean(all_features, axis=0).tolist()
    feats_sigma = np.std(all_features, axis=0).tolist()
    
    print(f"Calculated feature statistics from {len(all_features)} particles:")
    print(f"Mean: {feats_mean}")
    print(f"Std: {feats_sigma}")
    
    # Save statistics if path provided
    if save_path:
        stats = {
            'feats_mean': feats_mean,
            'feats_sigma': feats_sigma
        }
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Feature statistics saved to {save_path}")
    
    return feats_mean, feats_sigma


def load_feature_statistics(load_path):
    """
    Load feature statistics from a JSON file.
    
    Parameters:
    -----------
    load_path: str
        Path to the JSON file containing feature statistics
        
    Returns:
    --------
    feats_mean: list
        Mean values for [x, y, z, KE, Px, Py, Pz] where KE = E - m
    feats_sigma: list
        Standard deviation values for [x, y, z, KE, Px, Py, Pz]
    """
    import json
    
    with open(load_path, 'r') as f:
        stats = json.load(f)
    
    feats_mean = stats['feats_mean']
    feats_sigma = stats['feats_sigma']
    
    print(f"Loaded feature statistics from {load_path}")
    print(f"Mean: {feats_mean}")
    print(f"Std: {feats_sigma}")
    
    return feats_mean, feats_sigma


def save_sequence_data(seqdata, save_path):
    """
    Save processed sequence data to a file.
    
    Parameters:
    -----------
    seqdata: dict
        Dictionary containing 'encoded_ids', 'particle_feats', 'padding_mask', 'causal_mask'
    save_path: str
        Path to save the sequence data
    """
    import torch
    import json
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to numpy arrays for serialization
    seqdata_to_save = {
        'encoded_ids': seqdata['encoded_ids'].numpy(),
        'particle_feats': seqdata['particle_feats'].numpy(),
        'padding_mask': seqdata['padding_mask'].numpy(),
    }
    
    # Add causal_mask if it exists
    if 'causal_mask' in seqdata and seqdata['causal_mask'] is not None:
        seqdata_to_save['causal_mask'] = seqdata['causal_mask'].numpy()
    else:
        seqdata_to_save['causal_mask'] = None
    
    # Save as numpy file (more efficient for large arrays)
    import numpy as np
    np.savez_compressed(save_path, **seqdata_to_save)
    
    # Also save metadata as JSON
    metadata_path = save_path.replace('.npz', '_metadata.json')
    metadata = {
        'num_sequences': len(seqdata['encoded_ids']),
        'seq_len': seqdata['encoded_ids'].shape[1],
        'num_features': seqdata['particle_feats'].shape[2],
        'has_causal_mask': seqdata['causal_mask'] is not None
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sequence data saved to {save_path}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Saved {metadata['num_sequences']} sequences of length {metadata['seq_len']}")


def load_sequence_data(load_path):
    """
    Load processed sequence data from a file.
    
    Parameters:
    -----------
    load_path: str
        Path to load the sequence data from
        
    Returns:
    --------
    seqdata: dict
        Dictionary containing 'encoded_ids', 'particle_feats', 'padding_mask', 'causal_mask'
    """
    import torch
    import numpy as np
    
    # Load the data with allow_pickle=True to handle None values
    data = np.load(load_path, allow_pickle=True)
    
    # Convert back to tensors
    seqdata = {
        'encoded_ids': torch.tensor(data['encoded_ids'], dtype=torch.long),
        'particle_feats': torch.tensor(data['particle_feats'], dtype=torch.float32),
        'padding_mask': torch.tensor(data['padding_mask'], dtype=torch.bool),
    }
    
    # Add causal_mask if it exists
    if 'causal_mask' in data:
        # Check if causal_mask is an object array (contains None) or is None
        if data['causal_mask'] is None or data['causal_mask'].dtype == np.object_:
            seqdata['causal_mask'] = None
        else:
            seqdata['causal_mask'] = torch.tensor(data['causal_mask'], dtype=torch.bool)
    else:
        seqdata['causal_mask'] = None
    
    # Load and print metadata
    metadata_path = load_path.replace('.npz', '_metadata.json')
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded sequence data from {load_path}")
        print(f"Loaded {metadata['num_sequences']} sequences of length {metadata['seq_len']}")
    except FileNotFoundError:
        print(f"Loaded sequence data from {load_path}")
        print(f"Warning: Metadata file not found at {metadata_path}")
    
    return seqdata


def detect_data_format(data_path):
    """
    Detect whether the data_path contains H5 files or ROOT files.
    
    Returns:
        tuple: (format_type, h5_file_path)
        - format_type: 'h5' if H5 file found, 'root' if ROOT files found, 'unknown' otherwise
        - h5_file_path: path to H5 file if found, or suggested path for conversion
    """
    data_path = Path(data_path)
    
    # Check if it's a direct H5 file
    if data_path.is_file() and data_path.suffix.lower() in ['.h5', '.hdf5']:
        return 'h5', str(data_path)
    
    # Check if it's a directory
    if data_path.is_dir():
        # Look for H5 files in the directory
        h5_files = list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5"))
        if h5_files:
            return 'h5', str(h5_files[0])
        
        # Look for ROOT files with the expected pattern
        root_files = list(data_path.glob("EventOutput.Pert.00000*.root"))
        if root_files:
            # Generate suggested H5 output path
            h5_output = data_path / "GiBUU_FSI_particles.h5"
            return 'root', str(h5_output)
    
    return 'unknown', None


def convert_root_to_h5(root_data_path, h5_output_path):
    """
    Convert ROOT files to H5 format using build_gibuu_h5_from_root.
    
    Parameters:
    -----------
    root_data_path: str or Path
        Directory containing ROOT files
    h5_output_path: str or Path
        Output H5 file path
    """
    from .data_processing import build_gibuu_h5_from_root
    
    root_data_path = Path(root_data_path)
    
    # Find all ROOT files to determine the timestep range
    root_files = sorted(list(root_data_path.glob("EventOutput.Pert.00000*.root")))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in {root_data_path}")
    
    # Extract timesteps from filenames
    timesteps = []
    for root_file in root_files:
        # Extract timestep from filename like "EventOutput.Pert.00000123.root"
        filename = root_file.name
        if "EventOutput.Pert.00000" in filename and filename.endswith(".root"):
            timestep_str = filename.replace("EventOutput.Pert.00000", "").replace(".root", "")
            try:
                timestep = int(timestep_str)
                timesteps.append(timestep)
            except ValueError:
                print(f"Warning: Could not parse timestep from {filename}")
    
    if not timesteps:
        raise ValueError("No valid timesteps found in ROOT filenames")
    
    timesteps = sorted(timesteps)
    print(f"Found {len(timesteps)} timesteps: {timesteps[0]} to {timesteps[-1]}")
    
    # Convert ROOT files to H5
    build_gibuu_h5_from_root(
        data_path=str(root_data_path),
        out_path=h5_output_path,
        timesteps=timesteps,
        group_key="perturbative",
        filename_pattern="EventOutput.Pert.00000{ttt:03d}.root"
    )
    
    print(f"Successfully converted ROOT files to H5: {h5_output_path}")


def list_checkpoints(experiment_name, log_dir="lightning_logs"):
    """
    List all available checkpoints for an experiment.
    
    Parameters:
    -----------
    experiment_name: str
        Name of the experiment
    log_dir: str
        Directory where logs are saved (default: "lightning_logs")
        
    Returns:
    --------
    checkpoint_paths: list of str
        List of checkpoint paths, sorted by modification time (newest first)
    """
    checkpoint_dir = Path(log_dir) / experiment_name / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"No checkpoint directory found for experiment: {experiment_name}")
        return []
    
    checkpoint_files = sorted(checkpoint_dir.glob('*.ckpt'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    print(f"Available checkpoints for '{experiment_name}':")
    for i, ckpt in enumerate(checkpoint_files, 1):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        modified = ckpt.stat().st_mtime
        import datetime
        mod_time = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i}. {ckpt.name} ({size_mb:.1f} MB, {mod_time})")
    
    return [str(f) for f in checkpoint_files]


def find_checkpoint(experiment_name, checkpoint_path=None, checkpoint_name="last.ckpt", log_dir="lightning_logs"):
    """
    Find a checkpoint for resuming training.
    
    Parameters:
    -----------
    experiment_name: str
        Name of the experiment
    checkpoint_path: str, optional
        Specific checkpoint path to use. If provided, returns this path if it exists.
    checkpoint_name: str
        Name of checkpoint file to look for (default: "last.ckpt")
        If the named checkpoint doesn't exist, falls back to most recent checkpoint
    log_dir: str
        Directory where logs are saved (default: "lightning_logs")
        
    Returns:
    --------
    checkpoint_path: str or None
        Path to checkpoint file, or None if not found
    """
    # If specific checkpoint provided, verify it exists
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            print(f"✓ Using specified checkpoint: {checkpoint_path}")
            return str(checkpoint_path)
        else:
            print(f"⚠ Specified checkpoint not found: {checkpoint_path}")
            return None
    
    # Look in experiment directory
    checkpoint_dir = Path(log_dir) / experiment_name / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"⚠ Checkpoint directory doesn't exist: {checkpoint_dir}")
        return None
    
    # Try to find the named checkpoint
    named_ckpt = checkpoint_dir / checkpoint_name
    if named_ckpt.exists():
        print(f"✓ Found checkpoint: {named_ckpt}")
        return str(named_ckpt)
    
    # Fall back to most recent checkpoint
    checkpoint_files = sorted(checkpoint_dir.glob('*.ckpt'), key=lambda x: x.stat().st_mtime, reverse=True)
    if checkpoint_files:
        print(f"✓ Using most recent checkpoint: {checkpoint_files[0]}")
        return str(checkpoint_files[0])
    
    print(f"⚠ No checkpoints found in {checkpoint_dir}")
    return None


def plot_loss_curves(csv_path, mode='propagation', figsize=(15, 10), title=None, 
                     logy=True, window_size=None, custom_epoch_ticks=None):
    """
    Load and plot training/validation loss curves from PyTorch Lightning CSV logs.
    
    Parameters:
    -----------
    csv_path: str or Path
        Path to metrics.csv file (e.g., "lightning_logs/gibuu_propagation/version_0/metrics.csv")
        or "lightning_logs/gibuu_propagation/combined.csv" for combined versions
    mode: str
        Model type: 'propagation' or 'interaction'
        - 'propagation': plots feature, position, em_zero, em_value, interaction losses
        - 'interaction': plots total, type, feat losses
    figsize: tuple
        Figure size (width, height)
    title: str, optional
        Custom title for the plot. If None, uses the csv filename
    logy: bool
        Whether to use log scale for y-axis (default: True)
    window_size: int, optional
        Window size for rolling average. If None, auto-computed as train_steps/val_steps
    custom_epoch_ticks: array-like or integer, optional
        Custom epoch tick marks for secondary x-axis (e.g., np.arange(0, 200, 10))
        
    Returns:
    --------
    fig, axes: matplotlib figure and axes objects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"⚠ CSV file not found: {csv_path}")
        return None, None
    
    # Load CSV
    print(f"Loading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Columns: {list(df.columns)}")
    
    # Define loss terms based on mode
    if mode == 'propagation':
        loss_terms = {
            'interaction_loss': 'Interaction Loss (Binary Classification)',
            'position_loss': 'Position Loss (Δx, Δy, Δz)',
            'em_zero_loss': 'E/p Zero Classification Loss',
            'em_value_loss': 'E/p Value Regression Loss',
            'feature_loss': 'Feature Loss (Combined)'
        }
        n_rows, n_cols = 2, 3
    elif mode == 'interaction':
        loss_terms = {
            'loss': 'Total Loss',
            'type_loss': 'Type Loss (Particle Classification)',
            'feat_loss': 'Feature Loss (x, y, z, KE, p)'
        }
        n_rows, n_cols = 2, 2
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'propagation' or 'interaction'")
    
    # Create subplots
    n_plots = len(loss_terms)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    def plot_loss_component(ax, train_col, val_col, loss_name):
        """Helper to plot a single loss component with train/val curves."""
        plotted = False
        
        # Plot train loss (raw + rolling average)
        if train_col in df.columns:
            train_mask = ~df[train_col].isna()
            if 'step' in df.columns:
                train_steps = df['step'][train_mask]
                train_losses = df[train_col][train_mask]
            else:
                train_steps = df.index[train_mask]
                train_losses = df[train_col][train_mask]
            
            if len(train_steps) > 0:
                # Plot raw train (thin, transparent)
                ax.plot(train_steps, train_losses, label='Train (raw)', 
                       alpha=0.3, linewidth=0.5, color='blue')
                plotted = True
        
        # Plot val loss
        if val_col in df.columns:
            val_mask = ~df[val_col].isna()
            if 'step' in df.columns:
                val_steps = df['step'][val_mask]
                val_losses = df[val_col][val_mask]
            else:
                val_steps = df.index[val_mask]
                val_losses = df[val_col][val_mask]
            
            if len(val_steps) > 0:
                ax.plot(val_steps, val_losses, '.-', label='Validation', 
                       alpha=0.7, markersize=5, color='orange')
                plotted = True
                
                # Calculate rolling average for train
                if train_col in df.columns and len(train_steps) > 0:
                    # Auto-compute window size if not provided
                    if window_size is None:
                        window = max(1, len(train_steps) // len(val_steps))
                    else:
                        window = window_size
                    
                    train_rolling = train_losses.rolling(window=window, center=True).mean()
                    ax.plot(train_steps, train_rolling, label='Train (rolling avg)', 
                           alpha=0.8, linewidth=1.5, color='green')
        
        if not plotted:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_title(loss_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        
        if logy:
            ax.set_yscale('log')
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add secondary x-axis for epoch
        if 'epoch' in df.columns and val_col in df.columns and plotted:
            val_mask = ~df[val_col].isna()
            if val_mask.any() and 'step' in df.columns:
                ax2 = ax.twiny()
                ax2.set_xlim(ax.get_xlim())
                
                all_epochs = df['epoch'][val_mask].unique()
                all_epochs = np.sort(all_epochs)
                
                if custom_epoch_ticks is not None:
                    if isinstance(custom_epoch_ticks, int):
                        # If int, use it as desired number of ticks
                        num_ticks = custom_epoch_ticks
                        if len(all_epochs) > num_ticks:
                            step_size = max(1, len(all_epochs) // num_ticks)
                            epoch_indices = np.arange(0, len(all_epochs), step_size)
                            selected_epochs = all_epochs[epoch_indices]
                        else:
                            selected_epochs = all_epochs
                        
                        step_ticks = []
                        epoch_labels = []
                        for e in selected_epochs:
                            steps = df.loc[df['epoch'] == e, 'step']
                            if not steps.empty:
                                step_ticks.append(steps.iloc[0])
                                epoch_labels.append(int(e))
                        
                        if step_ticks:
                            ax2.set_xticks(step_ticks)
                            ax2.set_xticklabels(epoch_labels)
                    else:
                        # If array-like, use as explicit epoch values
                        step_ticks = []
                        for e in custom_epoch_ticks:
                            steps = df.loc[df['epoch'] == e, 'step']
                            if not steps.empty:
                                step_ticks.append(steps.iloc[0])
                            elif not df.empty and 'epoch' in df.columns:
                                step_ticks.append(np.interp(e, df['epoch'].dropna(), 
                                                            df['step'].dropna()))
                        if step_ticks:
                            ax2.set_xticks(step_ticks)
                            ax2.set_xticklabels(custom_epoch_ticks)
                else:
                    # Auto-select sparse epoch ticks (max 20 ticks)
                    if len(all_epochs) > 20:
                        # Sample every N epochs to get ~20 ticks
                        step_size = max(1, len(all_epochs) // 20)
                        epoch_indices = np.arange(0, len(all_epochs), step_size)
                        selected_epochs = all_epochs[epoch_indices]
                    else:
                        selected_epochs = all_epochs
                    
                    step_ticks = []
                    epoch_labels = []
                    for e in selected_epochs:
                        steps = df.loc[df['epoch'] == e, 'step']
                        if not steps.empty:
                            step_ticks.append(steps.iloc[0])
                            epoch_labels.append(int(e))
                    
                    if step_ticks:
                        ax2.set_xticks(step_ticks)
                        ax2.set_xticklabels(epoch_labels)
                
                ax2.set_xlabel('Epoch', fontsize=11)
    
    # Plot each loss term
    for idx, (loss_key, loss_name) in enumerate(loss_terms.items()):
        ax = axes[idx]
        train_col = f'train_{loss_key}'
        val_col = f'val_{loss_key}'
        plot_loss_component(ax, train_col, val_col, loss_name)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    # Set title
    if title is None:
        title = f'{csv_path.stem} - Loss Curves'
    plt.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    
    print(f"✓ Loss curves plotted for {mode} model")
    
    # Print final loss values
    max_epoch = int(df['epoch'].max()) if 'epoch' in df.columns else None
    print("\n" + "="*70)
    if max_epoch is not None:
        print(f"FINAL LOSS VALUES (Epoch {max_epoch}):")
    else:
        print("FINAL LOSS VALUES:")
    print("="*70)
    for loss_key in loss_terms.keys():
        val_col = f'val_{loss_key}'
        if val_col in df.columns:
            final_val = df[val_col].dropna()
            if len(final_val) > 0:
                print(f"  {val_col:30s}: {final_val.iloc[-1]:.6f}")
    print("="*70)
    
    return fig, axes
