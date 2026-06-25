"""Forward kinematics for the bundled Genie-01 (G01) dual-arm robot.

``CompiledKinematics`` calls ``_g01_fk.so``. The robot geometry, joint layout, and
the policy-action -> joint mapping are all baked into the compiled library; the
Python side only passes a 16-D action plus the held head/waist joints and gets
back the two end-effector poses. Nothing robot-specific lives in this module.
"""

import ctypes
from pathlib import Path

import numpy as np

_FK_LIB = Path(__file__).resolve().parent / "_g01_fk.so"


class CompiledKinematics:
    """Genie-01 (G01) FK via the bundled compiled library."""

    def __init__(self):
        try:
            lib = ctypes.CDLL(str(_FK_LIB))
        except OSError as e:  # e.g. missing or wrong-arch .so
            raise OSError(
                f"could not load the compiled FK library ({_FK_LIB.name}): {e}. "
                "It is built for a specific platform (linux x86_64)."
            ) from e
        cd = ctypes.POINTER(ctypes.c_double)
        lib.gesim_fk_action.argtypes = [cd] * 4
        lib.gesim_fk_action.restype = None
        self._fk_action = lib.gesim_fk_action

    def fk_action(self, action16, head_waist):
        """16-D policy action + 4 held head/waist joints -> left/right EE poses.

        Returns two ``(7,)`` arrays ``[x, y, z, qx, qy, qz, qw]`` (scalar-last).
        """
        a = np.ascontiguousarray(action16, dtype=np.float64).reshape(-1)
        hw = np.ascontiguousarray(head_waist, dtype=np.float64).reshape(-1)
        if a.shape[0] < 16 or hw.shape[0] != 4:
            raise ValueError(f"need action (>=16,) and head_waist (4,); got {a.shape}, {hw.shape}")
        left = np.zeros(7, dtype=np.float64)
        right = np.zeros(7, dtype=np.float64)
        cd = ctypes.POINTER(ctypes.c_double)
        self._fk_action(
            a.ctypes.data_as(cd),
            hw.ctypes.data_as(cd),
            left.ctypes.data_as(cd),
            right.ctypes.data_as(cd),
        )
        return left, right
