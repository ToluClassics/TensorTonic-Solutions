import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    positions = np.arange(0, seq_len)[:, np.newaxis]
    
    pos_indices = np.arange(0, d_model, 2)
    neg_indices = np.arange(1, d_model, 2) - 1

    freq = np.power(base, pos_indices/d_model)
    neg_freq = np.power(base, neg_indices/d_model)

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(positions/freq)
    pe[:, 1::2] = np.cos(positions/neg_freq)

    return pe