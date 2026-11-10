import numpy as np
import pytest

from benchmark import compare


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -1.0])
@pytest.mark.parametrize("side", ["reference", "actual"])
def test_compare_rejects_invalid_f0_in_unvoiced_frame(invalid, side):
    reference = np.array([0.0, 100.0])
    actual = reference.copy()
    (reference if side == "reference" else actual)[0] = invalid
    with pytest.raises(ValueError, match=rf"{side} F0.*finite, nonnegative"):
        compare(reference, actual)


def test_compare_silence():
    accuracy = compare(np.zeros(4), np.zeros(4))
    assert accuracy["vuv_disagreement"] == 0
    assert accuracy["joint_voiced_frames"] == 0
    assert accuracy["cents_p99"] is None
    assert accuracy["cents_max"] is None


def test_compare_reports_pitch_and_voicing_errors():
    accuracy = compare(np.array([0.0, 100.0, 200.0]), np.array([100.0, 200.0, 200.0]))
    assert accuracy["vuv_disagreement"] == pytest.approx(1 / 3)
    assert accuracy["joint_voiced_frames"] == 2
    assert accuracy["cents_max"] == pytest.approx(1200)
    assert accuracy["cents_p99"] == pytest.approx(1188)
    assert accuracy["over_600_cents"] == 1
