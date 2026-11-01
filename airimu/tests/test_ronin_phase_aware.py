import json

import h5py
import numpy as np
import torch
from pyhocon import ConfigFactory

from datasets.RoNINdataset import RoNINSequence, RoNINWindowDataset
from model.model_ronin_phase_aware import PhaseAwareVelocityModel


def _write_sequence(path, length=600):
    path.mkdir()
    time = np.arange(length, dtype=np.float64) / 200.0
    gyro = np.tile([0.1, 0.2, 0.3], (length, 1))
    acceleration = np.tile([0.0, 0.0, 9.81], (length, 1))
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (length, 1))
    position = np.stack([time, 2.0 * time, np.zeros_like(time)], axis=1)
    with h5py.File(path / 'data.hdf5', 'w') as hdf:
        hdf['synced/time'] = time
        hdf['synced/gyro_uncalib'] = gyro
        hdf['synced/acce'] = acceleration
        hdf['synced/game_rv'] = identity
        hdf['pose/tango_pos'] = position
        hdf['pose/tango_ori'] = identity
    (path / 'info.json').write_text(json.dumps({
        'imu_init_gyro_bias': [0.0, 0.0, 0.0],
        'imu_acce_bias': [0.0, 0.0, 0.0],
        'imu_acce_scale': [1.0, 1.0, 1.0],
        'start_calibration': [1.0, 0.0, 0.0, 0.0],
        'start_frame': 10,
    }), encoding='utf-8')


def test_ronin_sequence_matches_official_feature_and_target_definition(tmp_path):
    sequence_path = tmp_path / 'a001_1'
    _write_sequence(sequence_path)
    sequence = RoNINSequence(sequence_path)

    np.testing.assert_allclose(sequence.features[0, :3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(sequence.features[0, 3:], [0.0, 0.0, 9.81])
    np.testing.assert_allclose(
        sequence.velocity,
        np.tile([1.0, 2.0], (len(sequence.velocity), 1)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.isclose(np.median(np.diff(sequence.time)), 0.005)


def test_ronin_window_dataset_shapes(tmp_path):
    sequence_path = tmp_path / 'a001_1'
    _write_sequence(sequence_path)
    dataset = RoNINWindowDataset(
        tmp_path, sequence_names=['a001_1'], window_size=400, step_size=100
    )
    sample = dataset[0]
    assert sample['features'].shape == (400, 6)
    assert sample['velocity'].shape == (400, 2)
    assert sample['features'].dtype == np.float32


def test_phase_aware_velocity_model_float32_backward():
    conf = ConfigFactory.from_dict({
        'wavelet': 'db4',
        'wavelet_levels': 3,
        'wavelet_hidden': 32,
        'deep_hidden': 32,
        'fusion_dim': 32,
        'lstm_layers': 1,
        'down_ratios': [2, 2],
        'dropout': 0.0,
        'phase_tau': 0.5,
    })
    model = PhaseAwareVelocityModel(conf).float()
    features = torch.randn(2, 400, 6, dtype=torch.float32)
    velocity, aux = model(features, return_aux=True)
    assert velocity.shape == (2, 400, 2)
    assert velocity.dtype == torch.float32
    assert aux['scale_weights'].shape == (2, 4, 400)
    torch.testing.assert_close(
        aux['scale_weights'].sum(dim=1),
        torch.ones(2, 400),
    )
    velocity.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
