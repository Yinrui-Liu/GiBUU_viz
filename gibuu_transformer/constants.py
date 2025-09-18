"""
Constants and token definitions for GiBUU Transformer.
"""

# Special tokens
EOS_STEP_TOKEN = 0x400  # 0b10000000000 (1024) - End of time step
EOS_TOKEN = 0  # End of sequence (TB deprecated, now used the same as PAD_TOKEN)
PAD_TOKEN = 0  # Padding token

# Feature normalization constants (from the notebook)
FEATS_MEAN = None #[-7.29314323e-03, -2.40098036e-02, 3.47797358e+00, 1.04073876e+00, -1.41737541e-04, 1.54463698e-04, 2.55886665e-01]
FEATS_SIGMA = None # [4.41566758, 4.4296644, 5.7685071, 0.54804293, 0.25829807, 0.25919817, 0.64238038]

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
