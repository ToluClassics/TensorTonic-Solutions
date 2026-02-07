import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)

    if x.ndim == 1:
        x = x - np.max(x)
        e_val = np.exp(x)
        return e_val / np.sum(e_val)
    
    if x.ndim == 2:
        max_val = np.max(x, axis=1, keepdims=True)
        diff_max = x - max_val
        e_val = np.exp(diff_max)
        return e_val/np.sum(e_val, axis=1, keepdims=True)