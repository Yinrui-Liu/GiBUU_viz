"""
Data processing utilities for GiBUU particle sequences.
"""

import numpy as np
import h5py
import torch
import os
import uproot
from typing import Sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset
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


def load_pairs_from_npz(npz_files, max_pairs=None, random_subset=False, random_seed=42):
    """
    Load timestep pairs from NPZ files with flattened format.

    Shared helper for GiBUU_propagation.ipynb and GiBUU_interaction.ipynb.

    Parameters
    ----------
    npz_files : list of str
        Paths to NPZ files produced by GiBUU_dataprep.ipynb.
    max_pairs : int, optional
        Maximum number of pairs to load across all files (for quick testing).
        If None, loads all pairs.
    random_subset : bool, optional
        If False (default), takes the first max_pairs pairs in file order
        (fast, streaming-friendly).
        If True, randomly selects up to max_pairs pairs (approximately
        uniform over all pairs in the provided files).
    random_seed : int, optional
        Random seed used when random_subset=True.

    Returns
    -------
    pairs : list of tuples
        Each tuple is (input_particles, output_particles) where each is a list
        of particle tokens:
        [ts, gibuuID, charge, x, y, z, m, E, Px, Py, Pz]
    """
    pairs = []
    if not npz_files:
        return pairs

    remaining = max_pairs if max_pairs is not None else None
    rng = np.random.RandomState(random_seed) if random_subset else None

    for npz_file in npz_files:
        data = np.load(npz_file)

        input_lengths = data["input_lengths"]
        output_lengths = data["output_lengths"]
        input_particles = data["input_particles"]
        output_particles = data["output_particles"]

        num_pairs_in_file = len(input_lengths)
        if remaining is not None:
            num_pairs_in_file = min(num_pairs_in_file, remaining)

        if random_subset:
            # Sample indices within this file
            indices = np.arange(len(input_lengths))
            rng.shuffle(indices)
            indices = indices[:num_pairs_in_file]

            # Pre-compute offsets for random access
            inp_offsets = np.concatenate([[0], np.cumsum(input_lengths[:-1])])
            out_offsets = np.concatenate([[0], np.cumsum(output_lengths[:-1])])
        else:
            # Sequential indices
            indices = np.arange(num_pairs_in_file)
            inp_offsets = np.concatenate([[0], np.cumsum(input_lengths[:-1])])
            out_offsets = np.concatenate([[0], np.cumsum(output_lengths[:-1])])

        for idx in indices:
            inp_len = input_lengths[idx]
            out_len = output_lengths[idx]
            inp_offset = int(inp_offsets[idx])
            out_offset = int(out_offsets[idx])

            inp = input_particles[inp_offset: inp_offset + inp_len].tolist()
            out = output_particles[out_offset: out_offset + out_len].tolist()
            pairs.append((inp, out))

        if remaining is not None:
            remaining -= num_pairs_in_file
            if remaining <= 0:
                break

    return pairs


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
                
                # Encode particle ID (convert to int to handle float inputs from NPZ)
                encoded_id = encode_id(int(gibuu_id), int(charge))
                
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


class NPZStreamingDataset(IterableDataset):
    """
    Memory-efficient IterableDataset that streams items from one or more NPZ files.
    Yields all items with a 'split' field indicating train/val assignment.
    Uses deterministic hashing for consistent train/val split and optional shuffling.
    """

    def __init__(self, npz_file_paths, split_ratio: float = 0.8, shuffle: bool = True):
        self.npz_file_paths = list(npz_file_paths)
        self.split_ratio = float(split_ratio)
        self.shuffle = shuffle

    @staticmethod
    def _get_split(file_path: str, idx: int, split_ratio: float) -> str:
        # Stable hash independent of Python's randomized hash
        import hashlib
        key = f"{file_path}:{idx}".encode("utf-8")
        h = hashlib.md5(key).hexdigest()
        bucket = int(h[:8], 16) % 100  # 0..99
        return "train" if bucket < int(split_ratio * 100) else "val"

    def __iter__(self):
        from time import time
        
        # Check if we have any files to process
        if not self.npz_file_paths:
            print("NPZStreamingDataset: No NPZ files provided, yielding nothing")
            return
        
        # Seed per worker to change order across epochs
        worker_info = torch.utils.data.get_worker_info()
        base_seed = torch.initial_seed() if torch.initial_seed is not None else 0
        rng = np.random.RandomState(base_seed % (2**32 - 1))
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        print(f"NPZStreamingDataset: Worker {worker_id}/{num_workers}, shuffle={self.shuffle}")
        print(f"NPZStreamingDataset: Processing {len(self.npz_file_paths)} files: {self.npz_file_paths}")

        for file_idx, fp in enumerate(self.npz_file_paths):
            file_start = time()
            print(f"NPZStreamingDataset: Loading file {file_idx+1}/{len(self.npz_file_paths)}: {fp}")
            
            load_start = time()
            with np.load(fp, allow_pickle=True) as data:
                load_time = time() - load_start
                num_items = len(data['encoded_ids'])
                print(f"NPZStreamingDataset: File loaded in {load_time:.2f}s, has {num_items} items")
                
                # Convert to tensors once (like original method)
                tensor_start = time()
                encoded_ids = torch.tensor(data['encoded_ids'], dtype=torch.long)
                particle_feats = torch.tensor(data['particle_feats'], dtype=torch.float32)
                padding_mask = torch.tensor(data['padding_mask'], dtype=torch.bool)
                tensor_time = time() - tensor_start
                print(f"NPZStreamingDataset: Converted to tensors in {tensor_time:.2f}s")
                
                indices_start = time()
                indices = np.arange(num_items)
                if self.shuffle:
                    print(f"NPZStreamingDataset: Shuffling indices")
                    rng.shuffle(indices)
                indices_time = time() - indices_start
                print(f"NPZStreamingDataset: Indices prepared in {indices_time:.2f}s")
                
                # Shard indices across workers
                if num_workers > 1:
                    indices = indices[worker_id::num_workers]
                    print(f"NPZStreamingDataset: Worker {worker_id} processing {len(indices)} items")
                
                items_yielded = 0
                yield_start = time()
                for i in indices:
                    split = self._get_split(fp, int(i), self.split_ratio)
                    item = {
                        'encoded_ids': encoded_ids[i],  # Just indexing, no tensor creation
                        'particle_feats': particle_feats[i],
                        'padding_mask': padding_mask[i],
                        'split': split,
                    }
                    items_yielded += 1
                    yield item
                    if items_yielded % 1000 == 0:
                        yield_time = time() - yield_start
                        rate = items_yielded / yield_time if yield_time > 0 else 0
                        #print(f"NPZStreamingDataset: Yielded {items_yielded}/{len(indices)} items from {fp} ({rate:.1f} items/s)")
                        yield_start = time()  # Reset timer
                
                file_time = time() - file_start
                print(f"NPZStreamingDataset: Completed {fp} in {file_time:.2f}s, yielded {items_yielded} items")


class SplitFilterDataset(IterableDataset):
    """
    Wrapper that filters items from NPZStreamingDataset by split type.
    """
    
    def __init__(self, base_dataset: NPZStreamingDataset, split: str):
        assert split in ("train", "val")
        self.base_dataset = base_dataset
        self.split = split
    
    def __iter__(self):
        from time import time
        
        filter_start = time()
        items_seen = 0
        items_yielded = 0
        for item in self.base_dataset:
            items_seen += 1
            if item['split'] == self.split:
                # Remove split field before yielding
                del item['split']
                items_yielded += 1
                yield item
                if items_yielded % 1000 == 0:
                    filter_time = time() - filter_start
                    rate = items_yielded / filter_time if filter_time > 0 else 0
                    #print(f"SplitFilterDataset({self.split}): Yielded {items_yielded} items (seen {items_seen}) at {rate:.1f} items/s")
            if items_seen % 5000 == 0:
                filter_time = time() - filter_start
                rate = items_seen / filter_time if filter_time > 0 else 0
                #print(f"SplitFilterDataset({self.split}): Processed {items_seen} items at {rate:.1f} items/s")
        
        total_time = time() - filter_start
        print(f"SplitFilterDataset({self.split}): Final - yielded {items_yielded} items from {items_seen} total in {total_time:.2f}s")


def create_dataloaders(seqdata, batch_size=32, max_seq_len=6000, train_split=0.8):
    """
    Create training and validation dataloaders.

    Supports two inputs:
    - In-memory seqdata dict with tensors (existing behavior)
    - List/tuple of NPZ file paths for streaming from disk
    """
    # Case 1: list/tuple of NPZ file paths → streaming IterableDatasets
    if isinstance(seqdata, (list, tuple)) and all(isinstance(p, str) for p in seqdata):
        npz_files = list(seqdata)
        print(f"create_dataloaders: Received {len(npz_files)} NPZ files: {npz_files}")
        if len(npz_files) == 0:
            raise ValueError("Empty NPZ file list provided to create_dataloaders")

        # Single dataset loads all files once, then filter by split
        base_dataset = NPZStreamingDataset(npz_files, split_ratio=train_split, shuffle=True)
        train_dataset = SplitFilterDataset(base_dataset, "train")
        val_dataset = SplitFilterDataset(base_dataset, "val")

        # For IterableDataset, DataLoader shuffle has no effect; handled in-dataset.
        # Keep workers low to reduce concurrent file loads; increase if storage can handle it.
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        return train_loader, val_loader

    # Case 2: original dict of tensors → original behavior
    full_dataset = ParticleSequenceDataset(seqdata, max_seq_len)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

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
            gibuu_id = gibuuID_all[eid]
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


def extract_timestep_pairs(raw_sequences, max_particles=200):
    """
    Extract pairs of consecutive time steps from raw sequences.
    No padding - sequences keep their natural length (truncated if exceeding max_particles).
    
    Parameters:
    -----------
    raw_sequences: List of lists
        Each inner list is a sequence of particle tokens: [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
    max_particles: int
        Maximum number of particles per time step (truncate if exceeded, default: 200)
        
    Returns:
    --------
    pairs: List of tuples
        Each tuple is (input_tokens, output_tokens) where:
        - input_tokens: List of tokens at time step t (variable length, max max_particles)
        - output_tokens: List of tokens at time step t+1 (variable length, max max_particles)
    """
    pairs = []
    
    for sequence in raw_sequences:
        if len(sequence) == 0:
            continue
            
        # Group particles by time step
        time_steps = {}
        for token in sequence:
            ts = token[0]
            if ts not in time_steps:
                time_steps[ts] = []
            time_steps[ts].append(token)
        
        # Sort time steps
        sorted_ts = sorted(time_steps.keys())
        
        # Create pairs of consecutive time steps
        for i in range(len(sorted_ts) - 1):
            ts_t = sorted_ts[i]
            ts_t1 = sorted_ts[i + 1]
            
            input_tokens = time_steps[ts_t]
            output_tokens = time_steps[ts_t1]
            
            # Truncate if exceeding max_particles (no padding here)
            if len(input_tokens) > max_particles:
                input_tokens = input_tokens[:max_particles]
            if len(output_tokens) > max_particles:
                output_tokens = output_tokens[:max_particles]
            
            pairs.append((input_tokens, output_tokens))
    
    return pairs


def filter_pairs_with_changes(pairs):
    """
    Filter time step pairs to only keep those where particles change.
    
    A pair is kept if:
    1. The number of particles changes between time steps, OR
    2. If the number matches, the particle types (gibuuID, charge) don't match in order
    
    Parameters:
    -----------
    pairs: List of tuples
        Each tuple is (input_tokens, output_tokens) where tokens are:
        [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
        
    Returns:
    --------
    pairs_with_changes: List of tuples
        Pairs that have particle changes
    pairs_without_changes: List of tuples
        Pairs that don't have particle changes (for reference/analysis)
    """
    pairs_with_changes = []
    pairs_without_changes = []
    
    # Pre-extract all particle counts for vectorized comparison
    input_counts = np.array([len(input_tokens) for input_tokens, _ in pairs])
    output_counts = np.array([len(output_tokens) for _, output_tokens in pairs])
    
    # Find pairs where count changes (vectorized)
    count_changes = input_counts != output_counts
    
    # Process pairs in batches for efficiency
    for idx, (input_tokens, output_tokens) in enumerate(pairs):
        # If count changed, keep it
        if count_changes[idx]:
            pairs_with_changes.append((input_tokens, output_tokens))
            continue
        
        # If count is same, check if types match in order (vectorized)
        if len(input_tokens) == 0:
            # Empty sequences - skip
            continue
        
        # Extract gibuuID and charge for both input and output
        input_gibuu = np.array([token[1] for token in input_tokens], dtype=int)
        input_charge = np.array([token[2] for token in input_tokens], dtype=int)
        output_gibuu = np.array([token[1] for token in output_tokens], dtype=int)
        output_charge = np.array([token[2] for token in output_tokens], dtype=int)
        
        # Create composite keys: gibuuID * 1000 + (charge + 8) to handle negative charges
        input_keys = input_gibuu * 1000 + (input_charge + 8)
        output_keys = output_gibuu * 1000 + (output_charge + 8)
        
        # Vectorized comparison: check if types match in order
        if not np.array_equal(input_keys, output_keys):
            # Types don't match - keep this pair
            pairs_with_changes.append((input_tokens, output_tokens))
        else:
            # Types match exactly - no change
            pairs_without_changes.append((input_tokens, output_tokens))
    
    return pairs_with_changes, pairs_without_changes


def prepare_interaction_data(pairs, stats_path=None, save_stats_path=None):
    """
    Prepare interaction pairs for transformer training (encoder-decoder architecture).
    
    Prepares pairs where particle types/counts change between time steps.
    Returns variable-length sequences (padding happens in collate_fn).
    
    Parameters:
    -----------
    pairs: List of tuples
        Each tuple is (input_tokens, output_tokens) where tokens are:
        [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
    stats_path: str, optional
        Path to load pre-computed feature statistics
    save_stats_path: str, optional
        Path to save computed feature statistics
        
    Returns:
    --------
    data_dict: dict
        Dictionary containing:
        - 'input_encoded_ids': List of lists (encoder input: particles + EOS)
        - 'input_particle_feats': List of lists (encoder input features)
        - 'output_encoded_ids': List of lists (decoder input: START + particles + EOS)
        - 'output_particle_feats': List of lists (decoder input features)
    """
    from .constants import EOS_STEP_TOKEN, START_TOKEN
    from .utils import calculate_feature_statistics, load_feature_statistics
    
    print(f"Preparing {len(pairs)} interaction pairs for training...")
    
    # Handle feature statistics
    global FEATS_MEAN, FEATS_SIGMA
    from . import constants
    
    if stats_path:
        FEATS_MEAN, FEATS_SIGMA = load_feature_statistics(stats_path)
        constants.FEATS_MEAN = FEATS_MEAN
        constants.FEATS_SIGMA = FEATS_SIGMA
    elif constants.FEATS_MEAN is None or constants.FEATS_SIGMA is None:
        print("Calculating feature statistics...")
        all_tokens = []
        for input_tokens, _ in pairs:
            all_tokens.extend(input_tokens)
        FEATS_MEAN, FEATS_SIGMA = calculate_feature_statistics([all_tokens], save_stats_path)
        constants.FEATS_MEAN = FEATS_MEAN
        constants.FEATS_SIGMA = FEATS_SIGMA
    else:
        FEATS_MEAN = constants.FEATS_MEAN
        FEATS_SIGMA = constants.FEATS_SIGMA
    
    # Define START token
    START_TOKEN = 0x800
    
    # Process pairs
    input_encoded_ids_list = []
    input_particle_feats_list = []
    output_encoded_ids_list = []
    output_particle_feats_list = []
    
    eos_feats = [0.0] * 7
    start_feats = [0.0] * 7
    
    for input_tokens, output_tokens in pairs:
        # Encoder input: [P1, P2, ..., EOS]
        input_ids = []
        input_feats = []
        for token in input_tokens:
            ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
            encoded_id = encode_id(int(gibuu_id), int(charge))
            features = np.array([x, y, z, E-m, Px, Py, Pz])
            features = (features - np.array(FEATS_MEAN)) / np.array(FEATS_SIGMA)
            input_ids.append(encoded_id)
            input_feats.append(features.tolist())
        
        input_ids.append(EOS_STEP_TOKEN)
        input_feats.append(eos_feats)
        
        # Decoder input: [START, P1, P2, ..., EOS]
        output_ids = [START_TOKEN]
        output_feats = [start_feats]
        
        for token in output_tokens:
            ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
            encoded_id = encode_id(int(gibuu_id), int(charge))
            features = np.array([x, y, z, E-m, Px, Py, Pz])
            features = (features - np.array(FEATS_MEAN)) / np.array(FEATS_SIGMA)
            output_ids.append(encoded_id)
            output_feats.append(features.tolist())
        
        output_ids.append(EOS_STEP_TOKEN)
        output_feats.append(eos_feats)
        
        input_encoded_ids_list.append(input_ids)
        input_particle_feats_list.append(input_feats)
        output_encoded_ids_list.append(output_ids)
        output_particle_feats_list.append(output_feats)
    
    print(f"✓ Prepared {len(pairs)} interaction pairs")
    print(f"  Features: [x, y, z, KE, Px, Py, Pz] where KE = E - m")
    print(f"  Normalization: (feature - FEATS_MEAN) / FEATS_SIGMA")
    
    return {
        'input_encoded_ids': input_encoded_ids_list,
        'input_particle_feats': input_particle_feats_list,
        'output_encoded_ids': output_encoded_ids_list,
        'output_particle_feats': output_particle_feats_list
    }


def prepare_propagation_data(pairs_without_changes, pairs_with_changes, stats_path=None):
    """
    Prepare data for propagation model training.
    
    Uses both non-interaction pairs (for feature prediction) and interaction pairs (for classification).
    
    Parameters:
    -----------
    pairs_without_changes: List of tuples
        Pairs where particles don't change (interaction=0)
    pairs_with_changes: List of tuples
        Pairs where particles DO change (interaction=1)
    stats_path: str, optional
        Path to load pre-computed feature statistics
        
    Returns:
    --------
    data_dict: dict
        Dictionary containing:
        - 'particle_types': List of lists - encoded particle IDs
        - 'input_features': List of arrays - features at time t
        - 'target_features': List of arrays or None - features at time t+1
        - 'target_interaction': List of binary flags (0=no interaction, 1=interaction)
        - 'has_target_features': List of bools
    """
    from .utils import calculate_feature_statistics, load_feature_statistics
    from . import constants
    
    print(f"Preparing propagation data...")
    print(f"  Non-interaction pairs: {len(pairs_without_changes)}")
    print(f"  Interaction pairs: {len(pairs_with_changes)}")
    
    # Handle feature statistics
    if stats_path:
        FEATS_MEAN, FEATS_SIGMA = load_feature_statistics(stats_path)
        constants.FEATS_MEAN = FEATS_MEAN
        constants.FEATS_SIGMA = FEATS_SIGMA
    elif constants.FEATS_MEAN is None or constants.FEATS_SIGMA is None:
        print("Calculating feature statistics...")
        all_tokens = []
        for input_tokens, _ in pairs_without_changes + pairs_with_changes:
            all_tokens.extend(input_tokens)
        FEATS_MEAN, FEATS_SIGMA = calculate_feature_statistics([all_tokens])
        constants.FEATS_MEAN = FEATS_MEAN
        constants.FEATS_SIGMA = FEATS_SIGMA
    else:
        FEATS_MEAN = constants.FEATS_MEAN
        FEATS_SIGMA = constants.FEATS_SIGMA
    
    feats_mean = np.array(FEATS_MEAN)
    feats_sigma = np.array(FEATS_SIGMA)
    
    # Process pairs
    particle_types_list = []
    input_features_list = []
    target_features_list = []
    target_interaction_list = []
    has_target_features_list = []
    
    # Process non-interaction pairs
    for input_tokens, output_tokens in pairs_without_changes:
        if len(input_tokens) == 0 or len(input_tokens) != len(output_tokens):
            continue
        
        input_ids = [encode_id(int(token[1]), int(token[2])) for token in input_tokens]
        input_feats_unnorm = np.array([[x, y, z, E-m, Px, Py, Pz] 
                                       for ts, _, _, x, y, z, m, E, Px, Py, Pz in input_tokens])
        input_feats = (input_feats_unnorm - feats_mean) / feats_sigma
        
        output_feats_unnorm = np.array([[x, y, z, E-m, Px, Py, Pz] 
                                        for ts, _, _, x, y, z, m, E, Px, Py, Pz in output_tokens])
        target_feats = (output_feats_unnorm - feats_mean) / feats_sigma
        
        particle_types_list.append(input_ids)
        input_features_list.append(input_feats)
        target_features_list.append(target_feats)
        target_interaction_list.append(0)
        has_target_features_list.append(True)
    
    # Process interaction pairs
    for input_tokens, output_tokens in pairs_with_changes:
        if len(input_tokens) == 0:
            continue
        
        input_ids = [encode_id(int(token[1]), int(token[2])) for token in input_tokens]
        input_feats_unnorm = np.array([[x, y, z, E-m, Px, Py, Pz] 
                                       for ts, _, _, x, y, z, m, E, Px, Py, Pz in input_tokens])
        input_feats = (input_feats_unnorm - feats_mean) / feats_sigma
        
        particle_types_list.append(input_ids)
        input_features_list.append(input_feats)
        target_features_list.append(None)
        target_interaction_list.append(1)
        has_target_features_list.append(False)
    
    print(f"✓ Prepared {len(particle_types_list)} steps")
    print(f"  Non-interaction: {sum(1 for x in target_interaction_list if x == 0)}")
    print(f"  Interaction: {sum(target_interaction_list)}")
    
    return {
        'particle_types': particle_types_list,
        'input_features': input_features_list,
        'target_features': target_features_list,
        'target_interaction': target_interaction_list,
        'has_target_features': has_target_features_list
    }


def subset_propagation_data(propagation_data, num_samples=None, random_seed=42):
    """
    Create a random subset of propagation_data for testing.
    
    Parameters:
    -----------
    propagation_data: dict
        Full propagation data dictionary with keys:
        - 'particle_types': List of lists
        - 'input_features': List of arrays
        - 'target_features': List of arrays or None
        - 'target_interaction': List of binary flags
        - 'has_target_features': List of bools
    num_samples: int, optional
        Number of samples to keep. If None, returns all data.
    random_seed: int
        Random seed for reproducibility
        
    Returns:
    --------
    subset_data: dict
        Subset of propagation_data with same structure
    """
    import random
    
    total_samples = len(propagation_data['particle_types'])
    
    if num_samples is None or num_samples >= total_samples:
        return propagation_data
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create random indices
    indices = list(range(total_samples))
    random.shuffle(indices)
    selected_indices = sorted(indices[:num_samples])
    
    # Create subset
    subset_data = {
        'particle_types': [propagation_data['particle_types'][i] for i in selected_indices],
        'input_features': [propagation_data['input_features'][i] for i in selected_indices],
        'target_features': [propagation_data['target_features'][i] for i in selected_indices],
        'target_interaction': [propagation_data['target_interaction'][i] for i in selected_indices],
        'has_target_features': [propagation_data['has_target_features'][i] for i in selected_indices]
    }
    
    print(f"Created subset: {len(subset_data['particle_types'])} samples from {total_samples} total")
    print(f"  Non-interaction steps: {sum(1 for x in subset_data['target_interaction'] if x == 0)}")
    print(f"  Interaction steps: {sum(subset_data['target_interaction'])}")
    
    return subset_data


def subset_interaction_data(interaction_data, num_samples=None, random_seed=42):
    """
    Create a random subset of interaction_data for testing.
    
    Parameters:
    -----------
    interaction_data: dict
        Full interaction data dictionary with keys:
        - 'input_encoded_ids': List of lists
        - 'input_particle_feats': List of lists
        - 'output_encoded_ids': List of lists
        - 'output_particle_feats': List of lists
    num_samples: int, optional
        Number of samples to keep. If None, returns all data.
    random_seed: int
        Random seed for reproducibility
        
    Returns:
    --------
    subset_data: dict
        Subset of interaction_data with same structure
    """
    import random
    
    total_samples = len(interaction_data['input_encoded_ids'])
    
    if num_samples is None or num_samples >= total_samples:
        return interaction_data
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create random indices
    indices = list(range(total_samples))
    random.shuffle(indices)
    selected_indices = sorted(indices[:num_samples])
    
    # Create subset
    subset_data = {
        'input_encoded_ids': [interaction_data['input_encoded_ids'][i] for i in selected_indices],
        'input_particle_feats': [interaction_data['input_particle_feats'][i] for i in selected_indices],
        'output_encoded_ids': [interaction_data['output_encoded_ids'][i] for i in selected_indices],
        'output_particle_feats': [interaction_data['output_particle_feats'][i] for i in selected_indices]
    }
    
    print(f"Created subset: {len(subset_data['input_encoded_ids'])} pairs from {total_samples} total")
    
    return subset_data


def extract_first_last_pairs(raw_sequences):
    """
    Extract pairs of first time step -> last time step from each event sequence.
    Used for FSI (Final State Interaction) training where we predict the final state
    from the initial state.
    
    No truncation - keeps all particles. Padding happens later in collate function.
    
    Parameters:
    -----------
    raw_sequences: List of lists
        Each inner list is a sequence of particle tokens: [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
        
    Returns:
    --------
    pairs: List of tuples
        Each tuple is (input_tokens, output_tokens) where:
        - input_tokens: List of tokens at the FIRST time step (variable length, no truncation)
        - output_tokens: List of tokens at the LAST time step (variable length, no truncation)
    """
    pairs = []
    
    for sequence in raw_sequences:
        if len(sequence) == 0:
            continue
            
        # Group particles by time step
        time_steps = {}
        for token in sequence:
            ts = token[0]
            if ts not in time_steps:
                time_steps[ts] = []
            time_steps[ts].append(token)
        
        # Sort time steps
        sorted_ts = sorted(time_steps.keys())
        
        # Need at least 2 time steps to create a pair
        if len(sorted_ts) < 2:
            continue
        
        # Get first and last time steps
        first_ts = sorted_ts[0]
        last_ts = sorted_ts[-1]
        
        input_tokens = time_steps[first_ts]
        output_tokens = time_steps[last_ts]
        
        # No truncation - keep all particles, padding happens in collate function
        pairs.append((input_tokens, output_tokens))
    
    return pairs


def prepare_fsi_data(pairs, stats_path=None, save_stats_path=None, mask_position_features=False):
    """
    Prepare FSI (Final State Interaction) pairs for transformer training (encoder-decoder architecture).
    
    Prepares pairs where input is the first time step and output is the last time step of each event.
    Returns variable-length sequences (padding happens in collate_fn).
    
    Parameters:
    -----------
    pairs: List of tuples
        Each tuple is (input_tokens, output_tokens) where tokens are:
        [ts, gibuuID, chg, x, y, z, m, E, Px, Py, Pz]
        input_tokens: particles from first time step
        output_tokens: particles from last time step
    stats_path: str, optional
        Path to load pre-computed feature statistics
    save_stats_path: str, optional
        Path to save computed feature statistics
    mask_position_features: bool, optional
        If True, sets position features (x, y, z) to zero in both input and output.
        This is useful for FSI where positions may not be as relevant.
        Default: False
        
    Returns:
    --------
    data_dict: dict
        Dictionary containing:
        - 'input_encoded_ids': List of lists (encoder input: particles + EOS)
        - 'input_particle_feats': List of lists (encoder input features)
        - 'output_encoded_ids': List of lists (decoder input: START + particles + EOS)
        - 'output_particle_feats': List of lists (decoder input features)
    """
    from .constants import EOS_STEP_TOKEN, START_TOKEN
    from .utils import calculate_feature_statistics, load_feature_statistics
    
    print(f"Preparing {len(pairs)} FSI pairs (first step -> last step) for training...")
    if mask_position_features:
        print("  Note: Position features (x, y, z) will be masked (set to zero)")
    
    # Handle feature statistics
    global FEATS_MEAN, FEATS_SIGMA
    from . import constants
    
    if stats_path:
        FEATS_MEAN, FEATS_SIGMA = load_feature_statistics(stats_path)
        constants.FEATS_MEAN = FEATS_MEAN
        constants.FEATS_SIGMA = FEATS_SIGMA
    elif constants.FEATS_MEAN is None or constants.FEATS_SIGMA is None:
        print("Calculating feature statistics...")
        all_tokens = []
        for input_tokens, _ in pairs:
            all_tokens.extend(input_tokens)
        FEATS_MEAN, FEATS_SIGMA = calculate_feature_statistics([all_tokens], save_stats_path)
        constants.FEATS_MEAN = FEATS_MEAN
        constants.FEATS_SIGMA = FEATS_SIGMA
    else:
        FEATS_MEAN = constants.FEATS_MEAN
        FEATS_SIGMA = constants.FEATS_SIGMA
    
    # Process pairs
    input_encoded_ids_list = []
    input_particle_feats_list = []
    output_encoded_ids_list = []
    output_particle_feats_list = []
    
    eos_feats = [0.0] * 7
    start_feats = [0.0] * 7
    
    for input_tokens, output_tokens in pairs:
        # Encoder input: [P1, P2, ..., EOS]
        input_ids = []
        input_feats = []
        for token in input_tokens:
            ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
            encoded_id = encode_id(int(gibuu_id), int(charge))
            features = np.array([x, y, z, E-m, Px, Py, Pz])
            
            # Mask position features if requested
            if mask_position_features:
                features[0] = 0.0  # x
                features[1] = 0.0  # y
                features[2] = 0.0  # z
            
            features = (features - np.array(FEATS_MEAN)) / np.array(FEATS_SIGMA)
            input_ids.append(encoded_id)
            input_feats.append(features.tolist())
        
        input_ids.append(EOS_STEP_TOKEN)
        input_feats.append(eos_feats)
        
        # Decoder input: [START, P1, P2, ..., EOS]
        output_ids = [START_TOKEN]
        output_feats = [start_feats]
        
        for token in output_tokens:
            ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
            encoded_id = encode_id(int(gibuu_id), int(charge))
            features = np.array([x, y, z, E-m, Px, Py, Pz])
            
            # Mask position features if requested
            if mask_position_features:
                features[0] = 0.0  # x
                features[1] = 0.0  # y
                features[2] = 0.0  # z
            
            features = (features - np.array(FEATS_MEAN)) / np.array(FEATS_SIGMA)
            output_ids.append(encoded_id)
            output_feats.append(features.tolist())
        
        output_ids.append(EOS_STEP_TOKEN)
        output_feats.append(eos_feats)
        
        input_encoded_ids_list.append(input_ids)
        input_particle_feats_list.append(input_feats)
        output_encoded_ids_list.append(output_ids)
        output_particle_feats_list.append(output_feats)
    
    print(f"✓ Prepared {len(pairs)} FSI pairs")
    print(f"  Features: [x, y, z, KE, Px, Py, Pz] where KE = E - m")
    print(f"  Normalization: (feature - FEATS_MEAN) / FEATS_SIGMA")
    if mask_position_features:
        print(f"  Position features (x, y, z) masked (set to zero)")
    
    return {
        'input_encoded_ids': input_encoded_ids_list,
        'input_particle_feats': input_particle_feats_list,
        'output_encoded_ids': output_encoded_ids_list,
        'output_particle_feats': output_particle_feats_list
    }


class FSIStreamingDataset(IterableDataset):
    """
    Memory-efficient IterableDataset that streams FSI pairs from one or more NPZ files.
    Loads pairs on-the-fly and prepares them for training (encoding, normalization, etc.).
    Yields all items with a 'split' field indicating train/val assignment.
    Uses deterministic hashing for consistent train/val split and optional shuffling.
    """
    
    def __init__(self, npz_file_paths, split_ratio: float = 0.8, shuffle: bool = True, 
                 mask_position_features: bool = False):
        self.npz_file_paths = list(npz_file_paths)
        self.split_ratio = float(split_ratio)
        self.shuffle = shuffle
        self.mask_position_features = mask_position_features
        
        # Ensure feature statistics are loaded
        from . import constants
        global FEATS_MEAN, FEATS_SIGMA
        if constants.FEATS_MEAN is None or constants.FEATS_SIGMA is None:
            raise ValueError("FEATS_MEAN and FEATS_SIGMA must be set before using FSIStreamingDataset")
        FEATS_MEAN = constants.FEATS_MEAN
        FEATS_SIGMA = constants.FEATS_SIGMA
    
    @staticmethod
    def _get_split(file_path: str, idx: int, split_ratio: float) -> str:
        # Stable hash independent of Python's randomized hash
        import hashlib
        key = f"{file_path}:{idx}".encode("utf-8")
        h = hashlib.md5(key).hexdigest()
        bucket = int(h[:8], 16) % 100  # 0..99
        return "train" if bucket < int(split_ratio * 100) else "val"
    
    def _prepare_pair(self, input_tokens, output_tokens):
        """Prepare a single FSI pair for training."""
        from .constants import EOS_STEP_TOKEN, START_TOKEN
        
        eos_feats = [0.0] * 7
        start_feats = [0.0] * 7
        
        # Encoder input: [P1, P2, ..., EOS]
        input_ids = []
        input_feats = []
        for token in input_tokens:
            ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
            encoded_id = encode_id(int(gibuu_id), int(charge))
            features = np.array([x, y, z, E-m, Px, Py, Pz])
            
            # Mask position features if requested
            if self.mask_position_features:
                features[0] = 0.0  # x
                features[1] = 0.0  # y
                features[2] = 0.0  # z
            
            features = (features - np.array(FEATS_MEAN)) / np.array(FEATS_SIGMA)
            input_ids.append(encoded_id)
            input_feats.append(features.tolist())
        
        input_ids.append(EOS_STEP_TOKEN)
        input_feats.append(eos_feats)
        
        # Decoder input: [START, P1, P2, ..., EOS]
        output_ids = [START_TOKEN]
        output_feats = [start_feats]
        
        for token in output_tokens:
            ts, gibuu_id, charge, x, y, z, m, E, Px, Py, Pz = token
            encoded_id = encode_id(int(gibuu_id), int(charge))
            features = np.array([x, y, z, E-m, Px, Py, Pz])
            
            # Mask position features if requested
            if self.mask_position_features:
                features[0] = 0.0  # x
                features[1] = 0.0  # y
                features[2] = 0.0  # z
            
            features = (features - np.array(FEATS_MEAN)) / np.array(FEATS_SIGMA)
            output_ids.append(encoded_id)
            output_feats.append(features.tolist())
        
        output_ids.append(EOS_STEP_TOKEN)
        output_feats.append(eos_feats)
        
        return {
            'input_encoded_ids': input_ids,
            'input_particle_feats': input_feats,
            'output_encoded_ids': output_ids,
            'output_particle_feats': output_feats
        }
    
    def __iter__(self):
        from time import time
        
        # Check if we have any files to process
        if not self.npz_file_paths:
            print("FSIStreamingDataset: No NPZ files provided, yielding nothing")
            return
        
        # Seed per worker to change order across epochs
        worker_info = torch.utils.data.get_worker_info()
        try:
            base_seed = torch.initial_seed() % (2**32 - 1)
        except:
            base_seed = 0
        rng = np.random.RandomState(base_seed)
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        print(f"FSIStreamingDataset: Worker {worker_id}/{num_workers}, shuffle={self.shuffle}")
        print(f"FSIStreamingDataset: Processing {len(self.npz_file_paths)} files")
        
        for file_idx, fp in enumerate(self.npz_file_paths):
            file_start = time()
            print(f"FSIStreamingDataset: Loading file {file_idx+1}/{len(self.npz_file_paths)}: {fp}")
            
            load_start = time()
            with np.load(fp, allow_pickle=True) as data:
                load_time = time() - load_start
                
                # Reconstruct pairs from flattened format
                input_particles = data['input_particles']
                output_particles = data['output_particles']
                input_lengths = data['input_lengths']
                output_lengths = data['output_lengths']
                
                num_pairs = len(input_lengths)
                print(f"FSIStreamingDataset: File loaded in {load_time:.2f}s, has {num_pairs} pairs")
                
                # Compute offsets for random access
                input_offsets = np.concatenate([[0], np.cumsum(input_lengths[:-1])])
                output_offsets = np.concatenate([[0], np.cumsum(output_lengths[:-1])])
                
                # Create indices
                indices = np.arange(num_pairs)
                if self.shuffle:
                    rng.shuffle(indices)
                
                # Shard indices across workers
                if num_workers > 1:
                    indices = indices[worker_id::num_workers]
                    print(f"FSIStreamingDataset: Worker {worker_id} processing {len(indices)} pairs")
                
                items_yielded = 0
                yield_start = time()
                for i in indices:
                    # Reconstruct pair
                    inp_start = int(input_offsets[i])
                    inp_end = inp_start + int(input_lengths[i])
                    out_start = int(output_offsets[i])
                    out_end = out_start + int(output_lengths[i])
                    
                    input_tokens = input_particles[inp_start:inp_end].tolist()
                    output_tokens = output_particles[out_start:out_end].tolist()
                    
                    # Prepare pair for training
                    item = self._prepare_pair(input_tokens, output_tokens)
                    
                    # Add split field
                    item['split'] = self._get_split(fp, int(i), self.split_ratio)
                    
                    items_yielded += 1
                    yield item
                    
                    if items_yielded % 1000 == 0:
                        yield_time = time() - yield_start
                        rate = items_yielded / yield_time if yield_time > 0 else 0
                        yield_start = time()  # Reset timer
                
                file_time = time() - file_start
                print(f"FSIStreamingDataset: Completed {fp} in {file_time:.2f}s, yielded {items_yielded} pairs")


class FSISplitFilterDataset(IterableDataset):
    """
    Wrapper that filters items from FSIStreamingDataset by split type.
    """
    
    def __init__(self, base_dataset: FSIStreamingDataset, split: str):
        assert split in ("train", "val")
        self.base_dataset = base_dataset
        self.split = split
    
    def __iter__(self):
        from time import time
        
        filter_start = time()
        items_seen = 0
        items_yielded = 0
        for item in self.base_dataset:
            items_seen += 1
            if item['split'] == self.split:
                # Remove split field before yielding
                del item['split']
                items_yielded += 1
                yield item
                if items_yielded % 1000 == 0:
                    filter_time = time() - filter_start
                    rate = items_yielded / filter_time if filter_time > 0 else 0
        
        total_time = time() - filter_start
        print(f"FSISplitFilterDataset({self.split}): Final - yielded {items_yielded} pairs from {items_seen} total in {total_time:.2f}s")
