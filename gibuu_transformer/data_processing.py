"""
Data processing utilities for GiBUU particle sequences.
"""

import numpy as np
import h5py
import torch
import os
import uproot
from typing import Sequence
from torch.utils.data import Dataset, DataLoader
from .constants import EOS_STEP_TOKEN, EOS_TOKEN, PAD_TOKEN, FEATS_MEAN, FEATS_SIGMA
from .utils import calculate_feature_statistics, load_feature_statistics


def extract_particle_sequences(h5_file_path, gr_key="perturbative"):
    """
    Extracts the full sequence of particles for each event, flattened as a list of tokens. 
    Sort by time_step, then by E (descending) within each time step.
    Each token: [time_step, gibuuID, charge, x, y, z, m, E, Px, Py, Pz]
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
                        float(p['m']),
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
        Each inner list is a sequence of particle tokens: [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
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
                ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
                
                # Check if we've moved to a new time step
                if current_time_step is not None and ts != current_time_step:
                    # Insert EOS_STEP_TOKEN after the previous time step
                    processed_seq.append([EOS_STEP_TOKEN, 0, 0, 0, 0, 0, 0, 0])
                
                current_time_step = ts
                
                # Encode particle ID
                encoded_id = encode_id(gibuu_id, charge)
                
                # Create token: [encoded_id, x, y, z, E, Px, Py, Pz]
                processed_token = [encoded_id, x, y, z, E-m, Px, Py, Pz]
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
            features = token[1:]  # x, y, z, KE, Px, Py, Pz
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


def build_gibuu_h5_from_root(
    data_path: str,
    out_path: str,
    timesteps: Sequence[int],
    group_key: str = "perturbative",
    filename_pattern: str = "EventOutput.Pert.00000{ttt:03d}.root",
) -> None:
    """
    Read GiBUU ROOT files across timesteps and write an HDF5 file compatible 
    with GiBUU_Transformer extract_particle_sequences().
    
    Output HDF5 schema:
      - a single VLEN dataset at f[group_key] with length n_events
      - each element is a structured array with fields:
        ('time_step','i4'), ('gibuuID','i4'), ('charge','i4'),
        ('x','f4'), ('y','f4'), ('z','f4'), ('m','f4'), ('E','f4'),
        ('Px','f4'), ('Py','f4'), ('Pz','f4')
    
    Parameters:
    - data_path: directory containing ROOT files
    - out_path: output HDF5 path (e.g., "GiBUU_FSI_particles.h5")
    - timesteps: iterable of timestep integers (e.g., range(1, 203))
    - group_key: HDF5 key (default "perturbative") expected by downstream code
    - filename_pattern: template for ROOT filenames; default matches your notebook
    """
    # 1) Build weight-based event remapping (eID_map) to fix event order across timesteps
    weight_list = []
    for ttt in timesteps:
        file_name = os.path.join(data_path, filename_pattern.format(ttt=ttt))
        with uproot.open(file_name) as f:
            tree = f["RootTuple"]
            weight_list.append(tree["weight"].array(library="np"))
    eWID_arr = np.asarray(weight_list)  # shape: (n_steps, n_events)
    if eWID_arr.ndim != 2:
        raise RuntimeError(f"Unexpected weight array shape: {eWID_arr.shape}")
    n_steps, n_events = eWID_arr.shape
    if True: # TB checked if the WID is still necessary anymore
        ref_eWID = eWID_arr[0]
        eID_map = np.empty_like(eWID_arr, dtype=int)
        for i in range(n_steps):
            ref_sort = np.argsort(ref_eWID)
            cur_sort = np.argsort(eWID_arr[i])
            inverse_map = np.empty_like(ref_sort)
            inverse_map[ref_sort] = cur_sort
            eID_map[i] = inverse_map
            if (eWID_arr[i][inverse_map] != ref_eWID).any():
                print(f"[Warning] weight mismatch at step index {i} (t={timesteps[i]})")

    # 3) Read all steps, collect per-event structured arrays
    dtype_particle = np.dtype([
        ('time_step', 'i4'),
        ('gibuuID',   'i4'),
        ('charge',    'i4'), #('PDG', 'i4'),
        ('x',         'f4'), ('y',   'f4'), ('z', 'f4'),
        ('m',         'f4'), ('E',   'f4'),
        ('Px',        'f4'), ('Py',  'f4'), ('Pz', 'f4'),
    ])
    per_event_records = [[] for _ in range(n_events)]

    print("Start looping files")
    for step_idx, ttt in enumerate(timesteps):
        file_name = os.path.join(data_path, filename_pattern.format(ttt=ttt))
        with uproot.open(file_name) as f:
            tree = f["RootTuple"]
            x_all  = tree["x"].array(library="np")
            y_all  = tree["y"].array(library="np")
            z_all  = tree["z"].array(library="np")
            px_all = tree["Px"].array(library="np")
            py_all = tree["Py"].array(library="np")
            pz_all = tree["Pz"].array(library="np")
            pdg_all = tree["pdg_id"].array(library="np")
            gibuuID_all = tree["gibuu_id"].array(library="np")
            charge_all = tree["charge"].array(library="np")
            E_all = tree["E"].array(library="np")
            m_all = tree["mass"].array(library="np")

        # Remap events using eID_map to keep the same physical event across timesteps
        for event_idx in range(n_events):
            eid = event_idx#eID_map[step_idx, event_idx]
            x  = x_all[eid]; y  = y_all[eid]; z  = z_all[eid]
            px = px_all[eid]; py = py_all[eid]; pz = pz_all[eid]
            E  = E_all[eid]; m = m_all[eid]
            pdg = pdg_all[eid]
            gibuu_id = pdg_all[eid]
            charge = charge_all[eid]

            n_particles = len(pdg_all[eid])
            for i in range(n_particles):
                per_event_records[event_idx].append((
                    int(ttt),
                    int(gibuu_id[i]), int(charge[i]), #int(pdg[i]),
                    float(x[i]), float(y[i]), float(z[i]),
                    float(m[i]), float(E[i]),
                    float(px[i]), float(py[i]), float(pz[i]),
                ))

        if (step_idx + 1) % 10 == 0:
            print(f"### Loaded step {step_idx+1}/{n_steps}")

    # 4) Write HDF5: VLEN structured dataset at key `group_key`
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vlen_dtype = h5py.vlen_dtype(dtype_particle)
    with h5py.File(out_path, "w") as f:
        dset = f.create_dataset(group_key, shape=(n_events,), dtype=vlen_dtype)
        for i, recs in enumerate(per_event_records):
            if len(recs) == 0:
                dset[i] = np.array([], dtype=dtype_particle)
            else:
                dset[i] = np.array(recs, dtype=dtype_particle)
    print(f"[Saved] {out_path} with key '{group_key}' and {n_events} events")
