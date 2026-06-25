"""Compress a long action chunk to a shorter one (server-parity avg-pool).

A policy may emit more action rows than one world-model chunk consumes (e.g. pi05
returns 50, the model takes 25). Compressing the chunk to the model size lets the
world model run a single inference per turn instead of several.

The compression matches the world-model server's own preprocessing:

* gripper dims are nearest-neighbour sampled, preserving open/close timing;
* continuous (arm) dims keep their first/last rows and average-pool the interior.
"""

from __future__ import annotations

import numpy as np

# Gripper columns in the 16-D WM action layout ``[L7_arm, L_grip, R7_arm, R_grip]``.
WM_GRIPPER_DIMS = (7, 15)


def compress_action_chunk(
    actions: np.ndarray,
    target_len: int,
    *,
    gripper_dims: tuple[int, ...] = WM_GRIPPER_DIMS,
) -> np.ndarray:
    """Compress ``(L, C)`` actions to ``(target_len, C)``.

    ``L`` must be ``>= target_len`` — this only down-samples; up-sampling/padding a
    too-short chunk would silently fabricate rows, so it raises instead.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be (L, C); got {actions.shape}")
    L, C = actions.shape
    if L < target_len:
        raise ValueError(
            f"compress_action_chunk needs L >= target_len; got L={L}, target_len={target_len}"
        )
    if L == target_len:
        return actions.copy()
    if target_len == 1:
        return actions[:1].copy()

    grip = [d % C for d in gripper_dims]
    cont = [i for i in range(C) if i not in grip]
    out = np.zeros((target_len, C), dtype=np.float32)

    # gripper: nearest-neighbour down-sample (keep discrete open/close events)
    idx = np.round(np.linspace(0, L - 1, target_len)).astype(int).clip(0, L - 1)
    out[:, grip] = actions[idx][:, grip]

    # continuous: keep endpoints, average-pool the interior
    out[0, cont] = actions[0, cont]
    out[-1, cont] = actions[-1, cont]
    if target_len > 2:
        import torch
        import torch.nn.functional as F

        mid = torch.from_numpy(actions[1:-1, cont]).float().T.unsqueeze(0)  # (1, Cc, L-2)
        pooled = F.adaptive_avg_pool1d(mid, target_len - 2)
        out[1:-1, cont] = pooled.squeeze(0).T.numpy()
    return out
