import pypose as pp
import torch
from pyhocon import ConfigFactory

from datasets.dataset_utils import motion_collate
from inference_motion import detach_precise_timestamps
from model.code import CodeNetMotionwithRot
from model.model_freq_medium import FreqHybridModelMediumRot, HilbertPhase
from utils import as_so3, move_to, resolve_dtype


def _conf():
    return ConfigFactory.from_dict({
        'propcov': True,
        'dropout': 0.0,
        'rot_mask_ratio': 0.0,
        'rot_mask_value': 0.0,
    })


def test_move_to_casts_floats_but_preserves_indices():
    value = {
        'float': torch.ones(2, dtype=torch.float64),
        'index': torch.arange(2),
        'mask': torch.ones(2, dtype=torch.bool),
    }
    actual = move_to(value, 'cpu', dtype=resolve_dtype('float32'))
    assert actual['float'].dtype == torch.float32
    assert actual['index'].dtype == torch.int64
    assert actual['mask'].dtype == torch.bool


def test_absolute_timestamps_stay_float64_during_fp32_inference():
    timestamps = 1_403_636_580.0 + torch.arange(4, dtype=torch.float64) * 0.005
    network_data, precise = detach_precise_timestamps(
        {'ts': timestamps[None], 'acc': torch.ones(1, 3, 3, dtype=torch.float64)}
    )
    moved = move_to(network_data, 'cpu', dtype=torch.float32)

    assert moved['acc'].dtype == torch.float32
    assert precise.dtype == torch.float64
    torch.testing.assert_close(
        precise[0, 1:] - precise[0, :-1],
        torch.full((3,), 0.005, dtype=torch.float64),
        rtol=0.0,
        atol=2e-7,
    )
    assert torch.unique(precise).numel() == 4
    assert torch.unique(precise.float()).numel() == 1


def test_motion_collate_preserves_rotations():
    sample = {
        'timestamp': torch.arange(201, dtype=torch.float64),
        'dt': torch.ones(200, 1, dtype=torch.float64),
        'acc': torch.ones(200, 3, dtype=torch.float64),
        'gyro': torch.ones(200, 3, dtype=torch.float64),
        'rot': pp.identity_SO3(200).tensor(),
        'gt_pos': torch.ones(201, 3, dtype=torch.float64),
        'gt_rot': pp.identity_SO3(201).tensor(),
        'gt_vel': torch.ones(201, 3, dtype=torch.float64),
        'init_pos': torch.ones(1, 3, dtype=torch.float64),
        'init_rot': pp.identity_SO3(1).tensor(),
        'init_vel': torch.ones(1, 3, dtype=torch.float64),
    }
    data, init_state, label = motion_collate([sample, sample])
    assert hasattr(data['rot'], 'Inv')
    assert hasattr(init_state['rot'], 'Inv')
    assert hasattr(label['gt_rot'], 'Inv')

    moved = move_to(label, 'cpu', dtype=torch.float32)
    rotation_log = as_so3(moved['gt_rot'][:, :-1, :]).Log().tensor()
    assert rotation_log.dtype == torch.float32
    assert rotation_log.shape == (2, 200, 3)


def test_airio_baseline_float32_forward():
    model = CodeNetMotionwithRot(_conf()).float().eval()
    data = {'acc': torch.randn(2, 200, 3), 'gyro': torch.randn(2, 200, 3)}
    rotation_log = torch.randn(2, 200, 3) * 0.1
    output = model(data, rotation_log)
    assert output['net_vel'].dtype == torch.float32
    assert output['net_vel'].shape == (2, 23, 3)
    assert torch.isfinite(output['cov']).all()


def test_wavelet_phase_float32_forward():
    model = FreqHybridModelMediumRot(_conf()).float().eval()
    data = {'acc': torch.randn(2, 200, 3), 'gyro': torch.randn(2, 200, 3)}
    rotation_log = torch.randn(2, 200, 3) * 0.1
    output = model(data, rotation_log)
    assert output['net_vel'].dtype == torch.float32
    assert output['net_vel'].shape == (2, 200, 3)
    assert torch.isfinite(output['cov']).all()


def _reference_hilbert_phase(x):
    length = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)
    multiplier = -1j * torch.sign(torch.fft.fftfreq(length, device=x.device))
    if length % 2 == 0:
        multiplier[length // 2] = 0
    transformed = torch.fft.ifft(spectrum * multiplier, dim=-1).real
    magnitude = torch.sqrt(x.square() + transformed.square() + 1e-9)
    phase = torch.atan2(transformed, x)
    return magnitude, torch.sin(phase), torch.cos(phase)


def test_wavelet_hilbert_phase_matches_full_fft_reference():
    phase = HilbertPhase()
    for dtype, atol in ((torch.float32, 2e-5), (torch.float64, 5e-13)):
        for length in (127, 128, 257, 258):
            torch.manual_seed(length)
            signal = torch.randn(2, 6, length, dtype=dtype)
            actual = phase(signal)
            expected = _reference_hilbert_phase(signal)
            for actual_value, expected_value in zip(actual, expected):
                torch.testing.assert_close(actual_value, expected_value, rtol=atol, atol=atol)
