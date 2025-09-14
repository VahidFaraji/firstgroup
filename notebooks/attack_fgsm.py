#!/usr/bin/env python
# coding: utf-8

# # FGSM attack

# In[1]:


"""
attack_fgsm.py — Model-agnostic FGSM attack.

Works with any model as long as you provide grad_x_fn(x, labels) -> dL/dx
with the SAME SHAPE as x. Supports untargeted and targeted modes.

- epsilon: float or array broadcastable to x
- clip_min/clip_max: data range (e.g., 0..1)
"""

from __future__ import annotations
from typing import Callable, Optional, Union
import numpy as np

Array = np.ndarray


def _ensure_float_array(x: Array) -> Array:
    """Ensure x is float (preserve dtype if already float)."""
    if not np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32, copy=False)
    return x


def fgsm_attack(
    x: Array,
    y: Optional[Array],
    epsilon: Union[float, Array],
    grad_x_fn: Callable[[Array, Array], Array],
    *,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    targeted: bool = False,
    y_target: Optional[Array] = None,
) -> Array:
    """
    Generate adversarial examples using FGSM (L_inf).

    Parameters
    ----------
    x : np.ndarray
        Input batch (ANY shape). grad_x_fn must return grad with the SAME shape.
    y : np.ndarray or None
        Labels for untargeted FGSM. Ignored when targeted=True.
    epsilon : float or np.ndarray
        Step size (>=0). Can be broadcast to x.
    grad_x_fn : callable
        (x, labels) -> dL/dx with same shape as x.
    clip_min, clip_max : float
        Output clamp range.
    targeted : bool
        If True, move toward y_target (flip the sign).
    y_target : np.ndarray or None
        Required when targeted=True.

    Returns
    -------
    x_adv : np.ndarray  (same shape and float dtype as x)
    """
    x = np.asarray(x)
    x = _ensure_float_array(x)

    if clip_min >= clip_max:
        raise ValueError("clip_min must be < clip_max.")

    if targeted:
        if y_target is None:
            raise ValueError("For targeted=True, provide y_target.")
        labels = y_target
    else:
        if y is None:
            raise ValueError("For targeted=False, provide y.")
        labels = y

    eps = np.asarray(epsilon, dtype=x.dtype)
    if np.any(eps < 0):
        raise ValueError("epsilon must be non-negative.")

    grad = grad_x_fn(x, labels)
    if not isinstance(grad, np.ndarray):
        grad = np.asarray(grad)
    if grad.shape != x.shape:
        raise ValueError(
            f"grad_x_fn must return the SAME shape as x. Got {grad.shape} vs {x.shape}."
        )

    step_dir = np.sign(grad)
    x_adv = x - eps * step_dir if targeted else x + eps * step_dir
    x_adv = np.clip(x_adv, clip_min, clip_max)
    return x_adv.astype(x.dtype, copy=False)

