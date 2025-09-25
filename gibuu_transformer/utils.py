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
