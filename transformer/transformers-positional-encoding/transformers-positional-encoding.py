import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.zeros((seq_length, d_model))

    positions = np.arange(0, seq_length).reshape(-1, 1)

    positive_indices = np.arange(0, d_model, 2)
    negative_indices = np.arange(1, d_model, 2)

    pos_frequency = np.pow(10000, (2*positive_indices)/d_model)
    neg_frequency =  np.pow(10000, (2*negative_indices)/d_model)

    pe[:, 0::2] = np.sin(positions / pos_frequency)
    pe[:, 1::2] =  np.cos(positions / pos_frequency)

    return pe