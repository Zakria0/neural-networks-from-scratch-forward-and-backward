"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x, dtype=float)

    x_work = x.copy()
    for idx in np.ndindex(x.shape):
        x_orig = x_work[idx]
        
        x_work[idx] = x_orig + eps
        f_plus = f(x_work)

        x_work[idx] = x_orig - eps
        f_minus = f(x_work)

        grad[idx] = (f_plus - f_minus) / (2 * eps)

        x_work[idx] = x_orig
    
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    analytic_grad = np.asarray(analytic_grad)
    numeric_grad  = np.asarray(numeric_grad)

    return float(np.max(np.abs((analytic_grad - numeric_grad) / np.maximum(np.maximum(np.abs(analytic_grad), np.abs(numeric_grad)), tol))))

# Step 3 - make_dense (not yet solved)
# TODO: implement

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

