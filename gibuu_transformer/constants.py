"""
Constants and token definitions for GiBUU Transformer.
"""

# Special tokens
EOS_STEP_TOKEN = 0x400  # 0b10000000000 (1024) - End of time step
EOS_TOKEN = 0  # End of sequence (TB deprecated, now used the same as PAD_TOKEN)
PAD_TOKEN = 0  # Padding token
START_TOKEN = 0x800  # 0b100000000000 (2048) - Start of decoder sequence

# Feature normalization constants
# Computed from training data using GiBUU_dataprep.ipynb
# Features: [x, y, z, KE, Px, Py, Pz] where KE = E - m
FEATS_MEAN = [0.0, 0.0, 6.13, 0.42, 0.0, 0.0, 0.5]
FEATS_SIGMA = [5.89, 5.89, 9.33, 0.975, 0.304, 0.304, 1.123]

# Delta normalization statistics for ParticlePropagationModel
# Used to normalize deltas (differences) between consecutive time steps
# Filtered to exclude transitions where all E/P features are near-zero
FEATS_DELTA_MEAN = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]
FEATS_DELTA_SIGMA = [0.165, 0.165, 0.184, 6.72e-05, 1.25e-2, 1.25e-2, 1.25e-2]

# GiBUU ID to PDG code mapping
GIBUU_CHARGE_TO_PDG = {
    (1, 1): 2212,   # proton
    (1, 0): 2112,   # neutron
    (101, 1): 211,  # pi+
    (101, 0): 111,  # pi0
    (101, -1): -211, # pi-
}

# PDG color mapping for visualization
PDG_COLOR_MAP = {
    12: 'gray',         # electron neutrino
    14: 'gray',         # muon neutrino
    16: 'gray',         # tau neutrino
    11: 'red',          # electron
    -11: 'red',         # positron
    13: 'magenta',      # muon
    -13: 'magenta',     # muon+
    15: 'violet',       # tau
    -15: 'violet',      # tau+
    2212: 'blue',       # proton
    2112: 'cyan',       # neutron
    211: 'orange',      # pi+
    -211: 'darkorange', # pi-
    111: 'gold',        # pi0
    321: 'green',       # K+
    -321: 'darkgreen',  # K-
    311: 'lime',        # K0
    22: 'yellow',       # photon
}
