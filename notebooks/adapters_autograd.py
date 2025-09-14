#!/usr/bin/env python
# coding: utf-8

# # Adapters_autograd

# In[2]:


"""
adapters_autograd.py — Bridge your team's autograd model to FGSM.

Builds a grad_x_fn that computes dL/dx under SQUARE LOSS using your
existing autograd (Data/Layer/NeuralNetwork), WITHOUT touching team code.

Key features
------------
- Works with inputs shaped (D,B), (B,D), or images with an explicit batch axis
  (B, H, W, ...) or (H, W, ..., B). Returns grads in the SAME SHAPE as x.
- Infers number of classes C from model.forward(x_b) output shape (C,1).
- Supports labels as indices (B,) or one-hot in either (C,B) or (B,C).
- Per-sample backprop to avoid bias-broadcasting pitfalls.

Usage
-----
    grad_x_fn = build_grad_x_fn_autograd(model, Data)
    x_adv = fgsm_attack(x, y, epsilon, grad_x_fn, clip_min=0.0, clip_max=1.0)
"""

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

Array = np.ndarray


# ---------- small utilities ----------

def _patch_data_copy(DataClass) -> None:
    """Ensure Data.copy() exists so model.forward(x.copy()) works when x is Data."""
    if not hasattr(DataClass, "copy"):
        def _copy(self):  # noqa: ANN001
            return self
        setattr(DataClass, "copy", _copy)


def _infer_batch_size_from_labels(y: Array) -> int:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.shape[0]                 # indices: (B,)
    if y.ndim == 2:
        # one-hot: (C,B) یا (B,C) → معمولا B >> C
        return max(y.shape)               # ← FIX: بزرگ‌تر را بچ می‌گیریم
    raise ValueError(f"Unsupported label shape {y.shape}")




def _canonicalize_x(x: Array, y: Array) -> Tuple[Array, Callable[[Array], Array]]:
    """
    Convert x (ANY shape) to a canonical (D, B) 2D array for per-sample backprop,
    using the batch size inferred from y. Returns (x_cb, to_original).

    Rules
    -----
    - 1D: (D,) -> (D,1)
    - 2D:
        * If shape[1] == B and shape[0] != B -> already (D,B)
        * If shape[0] == B and shape[1] != B -> (B,D) -> transpose -> (D,B)
        * If both dims != B (ambiguous)      -> choose smaller dim as D:
              if rows <= cols: assume (D,B)
              else: transpose to (D,B)
        * If both dims == B (square)         -> assume columns are batch: (D,B)
    - >=3D:
        * If exactly one axis equals B       -> move that axis to last and flatten others
        * Else, try common fallbacks:
              - if last axis == B -> use it as batch
              - elif first axis == B -> use it as batch
              - elif any axis == B -> use the first match
              - else: raise
    """
    x = np.asarray(x)
    B = _infer_batch_size_from_labels(y)  # should return true batch size

    # 1D -> (D,1)
    if x.ndim == 1:
        x_cb = x.reshape(-1, 1)
        return x_cb, (lambda g: g.reshape(x.shape))

    # 2D handling
    if x.ndim == 2:
        r, c = x.shape
        if c == B and r != B:
            # Already (D,B)
            return x, (lambda g: g)
        if r == B and c != B:
            # (B,D) -> (D,B)
            return x.T, (lambda g: g.T)

        # Ambiguous cases:
        if r == B and c == B:
            # Square (B,B): assume columns are batch -> already (D,B)
            return x, (lambda g: g)
        # Neither dim equals B: choose smaller dim as features (D)
        if r <= c:
            # Treat as (D,B)
            return x, (lambda g: g)
        else:
            # Treat as (B,D) -> transpose
            return x.T, (lambda g: g.T)

    # >=3D handling
    axes_eq_B = [ax for ax, sz in enumerate(x.shape) if sz == B]
    if len(axes_eq_B) == 1:
        batch_axis = axes_eq_B[0]
    else:
        # Fallbacks if multiple/no matches
        if x.shape[-1] == B:
            batch_axis = x.ndim - 1
        elif x.shape[0] == B:
            batch_axis = 0
        elif len(axes_eq_B) >= 1:
            batch_axis = axes_eq_B[0]
        else:
            raise ValueError(
                f"Cannot infer batch axis for x.shape={x.shape} with B={B}. "
                "Ensure exactly one axis equals the batch size or place batch on the last axis."
            )

    # Move batch axis to the end, flatten others -> (D,B)
    x_moved = np.moveaxis(x, batch_axis, -1)
    D = int(np.prod(x_moved.shape[:-1]))
    x_cb = x_moved.reshape(D, x_moved.shape[-1])  # (D,B)

    def to_original(g_cb: Array) -> Array:
        g_moved = g_cb.reshape(x_moved.shape)
        return np.moveaxis(g_moved, -1, batch_axis)

    return x_cb, to_original



def _labels_col(y: Array, C: int, idx: int) -> Array:
    """
    Return labels for sample idx as (C,1).
    Accepts:
      - indices: (B,)
      - one-hot: (C,B) or (B,C)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        one = np.zeros((C, 1), dtype=np.float32)
        one[int(y[idx])] = 1.0
        return one

    if y.ndim == 2:
        # (C,B)
        if y.shape[0] == C:
            return y[:, [idx]]
        # (B,C)
        if y.shape[1] == C:
            col = y[[idx], :].reshape(C, 1)
            return col
    raise ValueError("Labels must be indices (B,) or one-hot (C,B)/(B,C).")


# ---------- main builder ----------

def build_grad_x_fn_autograd(model, DataClass) -> Callable[[Array, Array], Array]:
    """
    Build grad_x_fn(x, y) that computes dL/dx for a batch under square loss,
    using your team's autograd, without touching team files.

    Requirements:
      - model.forward(Data(x_b)) -> Data with shape (C,1)
      - DataClass supports autograd ops used inside the model
    """
    _patch_data_copy(DataClass)

    def grad_x_fn(x: Array, y: Array) -> Array:
        # Canonicalize x to (D,B) while remembering how to map grads back
        x_cb, to_original = _canonicalize_x(x, y)
        x_cb = x_cb.astype(np.float32, copy=False) if not np.issubdtype(x_cb.dtype, np.floating) else x_cb

        D, B = x_cb.shape
        grads_cb = np.zeros_like(x_cb, dtype=x_cb.dtype)

        for b in range(B):
            # Wrap one column (D,1) as Data
            x_b = DataClass(x_cb[:, [b]], parents=None, grad=False)

            # Forward: output Data with shape (C,1)
            out_b = model.forward(x_b)
            out_arr = getattr(out_b, "_data", None)
            if out_arr is None or out_arr.ndim != 2 or out_arr.shape[1] != 1:
                raise RuntimeError("model.forward must return Data with shape (C,1).")

            C = out_arr.shape[0]
            y_col = _labels_col(y, C, b)
            y_b = DataClass(y_col.astype(x_cb.dtype, copy=False), parents=None, grad=False)

            # Square loss per-sample: 0.5 * ||out - y||^2
            diff = out_b - y_b
            loss_b = 0.5 * (diff ** 2.0)  # reduction handled by Data.backward()

            # Backprop to input
            loss_b.backward()
            grads_cb[:, [b]] = x_b._num_grad.astype(x_cb.dtype, copy=False)

            # Best-effort: clear param grads to avoid accumulation
            try:
                for layer in getattr(model, "_layers", []):
                    layer._weights._num_grad.fill(0.0)
                    layer._bias._num_grad.fill(0.0)
            except Exception:
                pass

        # Map gradients back to the ORIGINAL x shape
        grads = to_original(grads_cb)
        return grads

    return grad_x_fn


# In[ ]:





# In[ ]:




