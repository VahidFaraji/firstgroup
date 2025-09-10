#!/usr/bin/env python
# coding: utf-8

# # Adapters_autograd

# In[2]:


"""
The goal here is to expose a simple callback:

    grad_x_fn = build_grad_x_fn_autograd(model, Data)

which you can pass into `fgsm_attack`.

Usage example
-------------
    from attack_fgsm import fgsm_attack
    from adapters_autograd import build_grad_x_fn_autograd

    # model: instance of your NeuralNetwork
    # Data:  the Data class from your project

    grad_x_fn = build_grad_x_fn_autograd(model, Data)

    x_adv = fgsm_attack(x, y, epsilon=0.03, grad_x_fn=grad_x_fn,
                        clip_min=0.0, clip_max=1.0, targeted=False)
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np

Array = np.ndarray


def _patch_data_copy(DataClass) -> None:
    if not hasattr(DataClass, "copy"):
        def _copy(self):
            return self
        # Bind as a method on the class
        setattr(DataClass, "copy", _copy)


def _ensure_2d_col_batch(a: Array) -> Array:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D array (D, B); got shape {a.shape}.")
    return a


def _labels_to_column(y: Array, C: int, idx: int) -> Array:
    y = np.asarray(y)
    # One-hot provided as (C, B)
    if y.ndim == 2 and y.shape[0] == C:
        col = y[:, [idx]]
        return col
    # Class indices provided as (B,)
    if y.ndim == 1 and y.shape[0] >= idx + 1:
        one = np.zeros((C, 1), dtype=y.dtype)
        one[int(y[idx])] = 1
        return one
    raise ValueError(
    )


def build_grad_x_fn_autograd(model, DataClass) -> Callable[[Array, Array], Array]:
    """Create a grad_x_fn(x, y) -> dL/dx callback for the custom autograd model.

    Parameters
    ----------
    model : object
        Your `NeuralNetwork` instance (with `.forward(x)` and `._layers`).
    DataClass : type

    Returns
    -------
    grad_x_fn : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function that computes ∂L/∂x for a batch `x` and labels `y` under the
        square loss. It returns an array with the same shape as `x`.
    """
    _patch_data_copy(DataClass)

    # Infer output dimension C from the last layer's bias shape (C,1)
    try:
        C = model._layers[-1]._bias._data.shape[0]
    except Exception as e:
        raise AttributeError(
        ) from e

    def grad_x_fn(x: Array, y: Array) -> Array:
        """Compute dL/dx for the whole batch by looping over samples. """
        # Ensure (D, B) input
        x_np = _ensure_2d_col_batch(x)
        D, B = x_np.shape

        # Prepare output container (same dtype as x, but ensure float)
        if not np.issubdtype(x_np.dtype, np.floating):
            x_np = x_np.astype(np.float32, copy=False)
        grads = np.zeros_like(x_np, dtype=x_np.dtype)

        # Loop over batch samples and backprop per sample
        for b in range(B):
            # Wrap input sample as Data with shape (D,1)
            x_b = DataClass(x_np[:, [b]], parents=None, grad=False)

            # Build label column (C,1)
            y_col = _labels_to_column(y, C, b)
            y_b = DataClass(y_col.astype(x_np.dtype, copy=False), parents=None, grad=False)

            # Forward
            out_b = model.forward(x_b)  # expected shape (C,1) as Data

            # Square loss: 0.5 * sum((out - y)^2)
            diff = out_b - y_b
            loss_b = 0.5 * (diff ** 2.0)  # no explicit sum reduction needed

            # Backpropagate
            loss_b.backward()

            # Extract gradient w.r.t. input
            grads[:, [b]] = x_b._num_grad.astype(x_np.dtype, copy=False)

            # Optional: clear accumulated grads on model params to avoid buildup
            try:
                for layer in getattr(model, "_layers", []):
                    layer._weights._num_grad.fill(0.0)
                    layer._bias._num_grad.fill(0.0)
            except Exception:
                pass

        return grads

    return grad_x_fn


__all__ = ["build_grad_x_fn_autograd"]


# In[ ]:





# In[ ]:




