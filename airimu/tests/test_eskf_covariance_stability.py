import numpy as np

from evaluation.ESKF_with_anomaly_detection import ESKF


def _make_eskf(*, sigma_bg=1e-6, sigma_ba=1e-12):
    return ESKF(
        init_p=np.zeros(3),
        init_v=np.zeros(3),
        init_q=np.array([1.0, 0.0, 0.0, 0.0]),
        init_bg=np.zeros(3),
        init_ba=np.zeros(3),
        sigma_g=np.zeros(3),
        sigma_a=np.zeros(3),
        sigma_bg=sigma_bg,
        sigma_ba=sigma_ba,
        gps_std=0.1,
        enable_anomaly_detection=False,
    )


def test_covariance_stabilization_projects_to_psd():
    eskf = _make_eskf()
    eskf.P = np.eye(15)
    eskf.P[0, 1] = eskf.P[1, 0] = 2.0
    eskf.P[2, 2] = 1e4

    eskf._stabilize_covariance()

    eigenvalues = np.linalg.eigvalsh(eskf.P)
    assert eigenvalues.min() >= -1e-10
    assert eigenvalues.max() <= eskf.max_cov + 1e-10
    np.testing.assert_allclose(eskf.P, eskf.P.T, atol=1e-12)


def test_bias_random_walk_noise_matches_state_order():
    sigma_bg = 0.02
    sigma_ba = 0.3
    dt = 0.1
    eskf = _make_eskf(sigma_bg=sigma_bg, sigma_ba=sigma_ba)
    eskf.P.fill(0.0)

    eskf.predict(np.zeros(6), dt)

    np.testing.assert_allclose(
        np.diag(eskf.P)[9:12], np.full(3, sigma_bg**2 * dt), rtol=1e-9
    )
    np.testing.assert_allclose(
        np.diag(eskf.P)[12:15], np.full(3, sigma_ba**2 * dt), rtol=1e-9
    )


def test_sparse_pose_update_keeps_state_and_covariance_finite():
    eskf = _make_eskf(sigma_bg=1e-12, sigma_ba=1e-12)
    imu = np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])

    for _ in range(2000):
        eskf.predict(imu, 0.005, sigma_a=np.full(3, 0.2), sigma_g=np.full(3, 5e-4))

    eskf.update_with_pose(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0.1, 0.03
    )

    assert np.isfinite(eskf.p).all()
    assert np.isfinite(eskf.v).all()
    assert np.isfinite(eskf.P).all()
    assert np.linalg.eigvalsh(eskf.P).min() >= -1e-10
