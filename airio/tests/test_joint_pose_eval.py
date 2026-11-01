import numpy as np

from evaluate_airio_joint_pose import nearest_measurement_indices, skew


def test_nearest_measurement_indices_respects_tolerance():
    imu_times = np.array([0.0, 0.005, 0.010, 0.015])
    measurement_times = np.array([0.0054, 0.0146])

    matched = nearest_measurement_indices(imu_times, measurement_times, 0.001)

    np.testing.assert_array_equal(matched, np.array([-1, 0, -1, 1]))


def test_skew_matrix_matches_cross_product():
    left = np.array([0.3, -0.4, 0.8])
    right = np.array([-0.2, 0.7, 0.5])

    np.testing.assert_allclose(skew(left) @ right, np.cross(left, right))
