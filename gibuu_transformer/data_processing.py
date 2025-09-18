"""
Data processing utilities for GiBUU particle sequences.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from .constants import EOS_STEP_TOKEN, EOS_TOKEN, PAD_TOKEN, FEATS_MEAN, FEATS_SIGMA
from .utils import calculate_feature_statistics, load_feature_statistics


def extract_particle_sequences(h5_file_path, gr_key="perturbative"):
    """
    Extracts the full sequence of particles for each event, flattened as a list of tokens. 
    Sort by time_step, then by E (descending) within each time step.
    Each token: [time_step, gibuuID, charge, x, y, z, E, Px, Py, Pz]
    Returns: List of lists (one per event), each inner list is the sequence of tokens.
    """
    sequences_per_event = []

    with h5py.File(h5_file_path, 'r') as f:
        events = f[gr_key]

        for i in range(len(events)):
            event_data = events[i]
            if len(event_data) == 0:
                sequences_per_event.append([])
                continue

            # Find all unique time steps, sorted
            time_steps = np.unique(event_data['time_step'])
            sequence = []
            for t in time_steps:
                mask = event_data['time_step'] == t
                particles = event_data[mask]
                # Sort by E descending
                sorted_idx = np.argsort(-particles['E'])
                particles = particles[sorted_idx]
                for p in particles:
                    token = [
                        int(p['time_step']),
                        int(p['gibuuID']),
                        int(p['charge']),
                        float(p['x']),
                        float(p['y']),
                        float(p['z']),
                        float(p['E']),
                        float(p['Px']),
                        float(p['Py']),
                        float(p['Pz']),
                    ]
                    if int(p['time_step']) > 1: # do not include time_step#1 as it seems always the same with time_step#2 (TB confirmed with GiBUU experts)
                        # TB considered, only include particles whose position is within the range of argon nucleus (r ≈ 5 fm)
                        sequence.append(token)
            
            sequences_per_event.append(sequence)
    return sequences_per_event


def encode_id(gibuu_id, charge, is_real=0):
    """
    Encode particle class to a 12-bit ID
     --------------------------------
    |   11    | 10 9 8 7 | 6 5 ... 0 | : bit
     --------------------------------
    | is_real |   Q + 8  | GiBUU ID  | : content
     --------------------------------
    """
    bits = gibuu_id \
        + ((charge+8) << 7) \
        + is_real * (1 << 11)
    return bits


def decode_id(bits):
    """
    Decode particle class from a 12-bit ID
    """
    gibuu_id = bits & 0x7f
    charge = ((bits >> 7) & 0xf) - 8
    is_real = (bits >> 11) & 0x1
    return gibuu_id, charge, is_real


def prepare_sequence_for_training(raw_sequences, max_seq_len=6000, recursive_truncate=False, 
                                 stats_path=None, save_stats_path=None):
    """
    Prepare raw particle sequences for training:
    - Encode particle IDs using encode_id()
    - Insert EOS_STEP_TOKEN after each time step
    - Insert EOS_TOKEN at the end
    - Pad to max length
    - Remove time_step (not needed in model input)
    
    Parameters:
    -----------
    raw_sequences: List of lists
        Each inner list is a sequence of particle tokens: [ts, gibuuID, chg, x, y, z, E, Px, Py, Pz]
    max_seq_len: int
        Maximum sequence length after padding
    recursive_truncate: bool
        If True, create additional sequences from truncated sequences
    stats_path: str, optional
        Path to load pre-computed feature statistics
    save_stats_path: str, optional
        Path to save computed feature statistics
        
    Returns:
    --------
    encoded_ids: tensor (batch, seq_len)
    particle_feats: tensor (batch, seq_len, num_features)
    padding_mask: tensor (batch, seq_len)
    """
    print(f"### Start prepare_sequence_for_training: {len(raw_sequences)} raw event sequences")
    
    # Handle feature statistics
    global FEATS_MEAN, FEATS_SIGMA
    
    if stats_path:
        # Load pre-computed statistics
        FEATS_MEAN, FEATS_SIGMA = load_feature_statistics(stats_path)
    elif FEATS_MEAN is None or FEATS_SIGMA is None:
        # Calculate statistics from data
        print("FEATS_MEAN and FEATS_SIGMA are None. Calculating from input data...")
        FEATS_MEAN, FEATS_SIGMA = calculate_feature_statistics(raw_sequences, save_stats_path)
    
    processed_sequences = []

    i_seq = 1
    for sequence in raw_sequences:
        if len(sequence) == 0:
            # Empty event: just EOS token
            processed_seq = [[EOS_TOKEN, 0, 0, 0, 0, 0, 0, 0]]
        else:
            processed_seq = []
            current_time_step = None
            
            for token in sequence:
                ts, gibuu_id, charge, x, y, z, E, Px, Py, Pz = token
                
                # Check if we've moved to a new time step
                if current_time_step is not None and ts != current_time_step:
                    # Insert EOS_STEP_TOKEN after the previous time step
                    processed_seq.append([EOS_STEP_TOKEN, 0, 0, 0, 0, 0, 0, 0])
                
                current_time_step = ts
                
                # Encode particle ID
                encoded_id = encode_id(gibuu_id, charge)
                
                # Create token: [encoded_id, x, y, z, E, Px, Py, Pz]
                processed_token = [encoded_id, x, y, z, E, Px, Py, Pz]
                processed_seq.append(processed_token)
            
            # Insert EOS_STEP_TOKEN after the last time step
            if len(processed_seq) > 0:
                processed_seq.append([EOS_STEP_TOKEN, 0, 0, 0, 0, 0, 0, 0])
            
            # Insert EOS_TOKEN at the end
            processed_seq.append([EOS_TOKEN, 0, 0, 0, 0, 0, 0, 0])

        if not recursive_truncate:
            # Pad to max length
            if len(processed_seq) < max_seq_len:
                pad_length = max_seq_len - len(processed_seq)
                padding = [[PAD_TOKEN, 0, 0, 0, 0, 0, 0, 0]] * pad_length
                processed_seq.extend(padding)
            else:
                processed_seq = processed_seq[:max_seq_len]
                
            processed_sequences.append(processed_seq)
        else:
            complete_recursive_truncate = False
            while not complete_recursive_truncate:
                if len(processed_seq) <= max_seq_len:
                    pad_length = max_seq_len - len(processed_seq)
                    padding = [[PAD_TOKEN, 0, 0, 0, 0, 0, 0, 0]] * pad_length
                    processed_seq.extend(padding)
                    processed_sequences.append(processed_seq)
                    complete_recursive_truncate = True
                else:
                    # truncate the sequence at max_seq_len
                    append_seq = processed_seq[:max_seq_len]
                    processed_sequences.append(append_seq)
                    # start the remaining sequence from the last complete time step
                    indices = [i for i, token in enumerate(append_seq) if token[0] == EOS_STEP_TOKEN]
                    if len(indices) < 2:
                        raise Exception("The step may be as long as max_seq_len. Increase max_seq_len if it's too small.")
                    start_idx = indices[-2]
                    processed_seq = processed_seq[start_idx+1:]
        
        if i_seq % 10000 == 0:
            print(f"Done loading {i_seq}/{len(raw_sequences)} raw event sequences...")
        i_seq += 1

    print("Converting to tensors...")
    batch_size = len(processed_sequences)
    seq_len = len(processed_sequences[0])
    
    # Extract encoded IDs and features
    encoded_ids = []
    particle_feats = []
    
    for seq in processed_sequences:
        seq_ids = []
        seq_feats = []
        for token in seq:
            encoded_id = token[0]
            features = token[1:]  # x, y, z, E, Px, Py, Pz
            # Convert to numpy arrays for normalization
            features = np.array(features)
            feats_mean = np.array(FEATS_MEAN)
            feats_sigma = np.array(FEATS_SIGMA)
            features = (features - feats_mean) / feats_sigma  # normalization
            
            seq_ids.append(encoded_id)
            seq_feats.append(features.tolist())
        
        encoded_ids.append(seq_ids)
        particle_feats.append(seq_feats)
    
    # Convert to tensors
    encoded_ids = torch.tensor(encoded_ids, dtype=torch.long)
    particle_feats = torch.tensor(particle_feats, dtype=torch.float32)
    
    # Create padding mask
    padding_mask = (encoded_ids == PAD_TOKEN)
    
    print(f"### Complete prepare_sequence_for_training: {len(encoded_ids)} sequences generated")
    return encoded_ids, particle_feats, padding_mask


class ParticleSequenceDataset(Dataset):
    """
    Dataset for particle sequences.
    """
    def __init__(self, seqdata, max_seq_len=6000):
        self.encoded_ids = seqdata["encoded_ids"]
        self.particle_feats = seqdata["particle_feats"]
        self.padding_mask = seqdata["padding_mask"]
        self.causal_mask = seqdata.get("causal_mask", None)
    
    def __len__(self):
        return len(self.encoded_ids)
    
    def __getitem__(self, idx):
        item = {
            'encoded_ids': self.encoded_ids[idx],
            'particle_feats': self.particle_feats[idx],
            'padding_mask': self.padding_mask[idx],
        }
        if hasattr(self, 'causal_mask') and self.causal_mask is not None:
            item['causal_mask'] = self.causal_mask[idx]
        return item


def create_dataloaders(seqdata, batch_size=32, max_seq_len=6000, train_split=0.8):
    """
    Create training and validation dataloaders.
    """
    # Create full dataset
    full_dataset = ParticleSequenceDataset(seqdata, max_seq_len)
    
    # Split into train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
