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

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": W, "b": b}

    return {
      'params': params,
      'forward': lambda x: (x @ params['W'] + params['b'], x),
      'backward': lambda dout, cache: (dout @ params['W'].T, {'W': cache.T @ dout, 'b': np.sum(dout, axis=0)})
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    if kind == 'relu':
      return {
        'params': {},
        'forward': lambda x: (np.maximum(x, 0), (x > 0)),
        'backward': lambda dout, cache: (dout * cache, {})
      }

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

