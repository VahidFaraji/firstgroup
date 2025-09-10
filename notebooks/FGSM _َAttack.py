#!/usr/bin/env python
# coding: utf-8

# # FGSM attack

# In[1]:


"""
attack_fgsm.py — Model-agnostic FGSM attack.
 We need to provide a `grad_x_fn` function that returns dL/dx for a given (x, labels).
"""
from __future__ import annotations

from typing import Callable, Optional, Union
import numpy as np

Array = np.ndarray


def _ensure_float_array(x: Array) -> Array:
    """Ensure `x` is a floating numpy array (preserving dtype if already float)."""
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
    
    """Generate adversarial examples using FGSM (Fast Gradient Sign Method).

    Parameters
    ----------
    x : np.ndarray
        Input batch to attack. Shape must be exactly the same shape returned by `grad_x_fn`. 
    y : np.ndarray or None
        Labels for *untargeted* attack. Ignored when `targeted=True` (use
        `y_target` instead). The format (one-hot or class indices) must match
        what your `grad_x_fn` expects.
    epsilon : float or np.ndarray
        Non-negative step size.
    grad_x_fn : Callable[[np.ndarray, np.ndarray], np.ndarray]
        A function that takes `(x, labels)` and returns `dL/dx` with the same
        shape as `x`.
    clip_min, clip_max : float
        Lower/upper bounds to clip the final adversarial examples. 
    targeted : bool
        If True, *minimizes* the loss towards `y_target` (moves x toward the
        target class), which flips the sign compared to the untargeted case.
    y_target : np.ndarray or None
        Required when `targeted=True`. Labels used by `grad_x_fn` to compute
        the gradient for the target objective.

    Returns
    -------
    x_adv : np.ndarray
        Adversarial examples with the same shape and (floating) dtype as `x`.

    """
    # Validate and normalize inputs
    x = np.asarray(x)
    x = _ensure_float_array(x)

    if clip_min >= clip_max:
        raise ValueError("clip_min must be < clip_max.")

    if targeted:
        if y_target is None:
            raise ValueError("For targeted=True, `y_target` must be provided.")
        labels = y_target
    else:
        if y is None:
            raise ValueError("For targeted=False, `y` must be provided.")
        labels = y

    eps = np.asarray(epsilon, dtype=x.dtype)
    if np.any(eps < 0):
        raise ValueError("`epsilon` must be non-negative.")

    # Compute gradient w.r.t. input
    grad = grad_x_fn(x, labels)
    if not isinstance(grad, np.ndarray):
        grad = np.asarray(grad)
    if grad.shape != x.shape:
        # We enforce exact match to avoid silent broadcasting bugs.
        raise ValueError(
            f"grad_x_fn must return an array with the same shape as x. "
            f"Got grad shape {grad.shape} vs x shape {x.shape}."
        )

    # Take the FGSM step
    step_dir = np.sign(grad)
    # Targeted attack moves *against* the gradient direction (to reduce L toward target)
    if targeted:
        x_adv = x - eps * step_dir
    else:
        x_adv = x + eps * step_dir

    # Clip to valid data range
    x_adv = np.clip(x_adv, clip_min, clip_max)

    # Keep original floating dtype (already ensured)
    return x_adv.astype(x.dtype, copy=False)


__all__ = ["fgsm_attack"]

