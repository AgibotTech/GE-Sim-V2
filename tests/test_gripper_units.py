import numpy as np
import pytest

from gesim.types import (
    GRIPPER_APERTURE_MAX_MM,
    GRIPPER_APERTURE_MIN_MM,
    wm_action_gripper_to_mm,
    wm_action_row_for_state_compare,
)


def test_wm_action_gripper_to_mm_endpoints():
    assert wm_action_gripper_to_mm(0.0) == pytest.approx(GRIPPER_APERTURE_MIN_MM)
    assert wm_action_gripper_to_mm(1.0) == pytest.approx(GRIPPER_APERTURE_MAX_MM)


def test_wm_action_row_for_state_compare_only_touches_grippers():
    row = np.arange(16, dtype=np.float32)
    out = wm_action_row_for_state_compare(row)
    assert np.array_equal(out[:7], row[:7])
    assert np.array_equal(out[8:15], row[8:15])
    assert out[7] == pytest.approx(wm_action_gripper_to_mm(row[7]))
    assert out[15] == pytest.approx(wm_action_gripper_to_mm(row[15]))
