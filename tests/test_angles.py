import numpy as np
import pytest

from cv.angles import joint_angle


@pytest.mark.parametrize(
    ("a", "c", "expected"),
    [
        ((1, 0), (0, 1), 90.0),  # right angle
        ((1, 0), (-1, 0), 180.0),  # straight limb
        ((1, 0), (2, 0), 0.0),  # fully folded
        ((1, 0), (1, 1), 45.0),
    ],
)
def test_known_angles_2d(a, c, expected):
    assert joint_angle(a, (0, 0), c) == pytest.approx(expected)


def test_3d():
    assert joint_angle((1, 0, 0), (0, 0, 0), (0, 0, 5)) == pytest.approx(90.0)


def test_scale_and_translation_invariant():
    a, b, c = np.array([3.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 4.0])
    assert joint_angle(a * 7 + 2, b * 7 + 2, c * 7 + 2) == pytest.approx(90.0)


def test_vectorized_over_frames():
    a = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
    b = np.zeros((3, 2))
    c = np.array([[0, 1], [-1, 0], [1, 1]], dtype=float)
    np.testing.assert_allclose(joint_angle(a, b, c), [90, 180, 45])


def test_zero_length_segment_is_nan():
    assert np.isnan(joint_angle((0, 0), (0, 0), (1, 0)))


def test_nan_input_is_nan():
    assert np.isnan(joint_angle((np.nan, 0), (0, 0), (1, 0)))
