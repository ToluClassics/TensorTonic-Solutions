import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not max_len:
        max_len = max([len(seq) for seq in seqs])

    seqs = [seq[:max_len] for seq in seqs]

    rows = len(seqs)
    cols = max_len

    dummy_tensor = np.full((rows, cols), pad_value)

    for i in range(len(seqs)):
        seq = seqs[i]
        dummy_tensor[i, :len(seq)] = seq

    return dummy_tensor
        
    
        
        