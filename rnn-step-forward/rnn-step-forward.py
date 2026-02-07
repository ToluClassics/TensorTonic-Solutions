import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t = np.array(x_t)
    h_prev = np.array(h_prev)
    Wx = np.array(Wx)
    Wh = np.array(Wh)

    l1 = np.matmul(x_t.T, Wx)
    l2 = np.matmul(h_prev.T,Wh)

    inner = l1 + l2 + b
    return np.tanh(inner)



