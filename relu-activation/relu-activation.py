import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    if isinstance(x, list):
        x = np.array(x)
    else:
        x = np.array([x]) 
    return np.where(x < 0, 0, x )