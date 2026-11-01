import torch
import pypose as pp
from pyhocon import ConfigFactory

from datasets.dataset_utils import custom_collate
from model.model_freq_improved import HilbertPhase, ImprovedFreqHybridModel
from model.model_phase_aware import PhaseAwareFreqHybridModel, PhaseAwareWaveletBranchSeq
from model.losses import get_RMSE, loss_
from model.loss_func import loss_fc_list
from utils import cat_state, move_to, resolve_dtype


def _reference_hilbert_phase(x):
    length = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)
    multiplier = -1j * torch.sign(torch.fft.fftfreq(length, device=x.device))
    if length % 2 == 0:
        multiplier[length // 2] = 0
    transformed = torch.fft.ifft(spectrum * multiplier, dim=-1).real
    magnitude = torch.sqrt(x.square() + transformed.square())
    phase = torch.atan2(transformed, x)
    return magnitude, torch.sin(phase), torch.cos(phase)


def _model_conf():
    return ConfigFactory.from_dict({
        'network': 'freqnet_improved',
        'gtrot': True,
        'propcov': True,
        'sampling': 25,
        'dropout': 0.0,
        'precision': 'float32',
    })


def test_half_spectrum_hilbert_matches_full_fft_reference():
    phase = HilbertPhase()
    for dtype, atol in ((torch.float32, 2e-5), (torch.float64, 5e-13)):
        for length in (127, 128, 257, 258):
            torch.manual_seed(length)
            signal = torch.randn(2, 6, length, dtype=dtype)
            actual = phase(signal)
            expected = _reference_hilbert_phase(signal)
            for actual_value, expected_value in zip(actual, expected):
                torch.testing.assert_close(actual_value, expected_value, rtol=atol, atol=atol)


def test_move_to_casts_only_floating_tensors():
    nested = {
        'float': torch.ones(2, dtype=torch.float64),
        'mask': torch.ones(2, dtype=torch.bool),
        'index': torch.arange(2, dtype=torch.int64),
    }
    result = move_to(nested, 'cpu', dtype=resolve_dtype('float32'))
    assert result['float'].dtype == torch.float32
    assert result['mask'].dtype == torch.bool
    assert result['index'].dtype == torch.int64


def test_cat_state_handles_a_short_final_batch():
    states = {
        'pos': [torch.zeros(3, 5, 3), torch.ones(1, 5, 3)],
        'rot': [pp.identity_SO3(3, 5), pp.identity_SO3(1, 5)],
        'cov': [torch.zeros(3, 9, 9), torch.ones(1, 9, 9)],
    }
    cat_state(states)
    assert states['pos'].shape == (1, 20, 3)
    assert states['rot'].shape == (1, 20, 4)
    assert states['cov'].shape == (4, 9, 9)


def test_float32_model_integrates_and_backpropagates():
    model = ImprovedFreqHybridModel(_model_conf()).float().train()
    batch, steps, pad = 1, 50, 9
    data = {
        'acc': torch.randn(batch, steps + pad, 3),
        'gyro': torch.randn(batch, steps + pad, 3) * 0.01,
        'dt': torch.full((batch, steps, 1), 0.005),
        'rot': pp.identity_SO3(batch, steps, dtype=torch.float32),
    }
    init_state = {
        'pos': torch.zeros(batch, 1, 3),
        'vel': torch.zeros(batch, 1, 3),
        'rot': pp.identity_SO3(batch, 1, dtype=torch.float32),
    }

    output = model(data, init_state)
    loss = output['pos'].square().mean() + output['correction_acc'].square().mean()
    loss = loss + output['cov'].mean()
    loss.backward()

    assert output['pos'].dtype == torch.float32
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_integrator_restores_plain_quaternion_batches():
    model = ImprovedFreqHybridModel(_model_conf()).float().eval()
    batch, steps, pad = 2, 50, 9
    data = {
        'acc': torch.randn(batch, steps + pad, 3),
        'gyro': torch.randn(batch, steps + pad, 3) * 0.01,
        'dt': torch.full((batch, steps, 1), 0.005),
        # Simulate a shuffled collate/device transfer that lost LieTensor type.
        'rot': pp.identity_SO3(batch, steps, dtype=torch.float32).tensor(),
    }
    init_state = {
        'pos': torch.zeros(batch, 1, 3),
        'vel': torch.zeros(batch, 1, 3),
        'rot': pp.identity_SO3(batch, 1, dtype=torch.float32).tensor(),
    }

    output = model(data, init_state)

    assert hasattr(output['rot'], 'Inv')
    assert output['pos'].shape == (batch, steps, 3)


def test_collate_explicitly_preserves_all_so3_fields():
    sample = {
        'dt': torch.ones(4, 1),
        'acc': torch.ones(4, 3),
        'gyro': torch.ones(4, 3),
        'rot': pp.identity_SO3(4).tensor(),
        'gt_pos': torch.ones(4, 3),
        'gt_rot': pp.identity_SO3(4).tensor(),
        'gt_vel': torch.ones(4, 3),
        'init_pos': torch.ones(1, 3),
        'init_rot': pp.identity_SO3(1).tensor(),
        'init_vel': torch.ones(1, 3),
    }

    data, init_state, label = custom_collate([sample, sample])

    assert hasattr(data['rot'], 'Inv')
    assert hasattr(init_state['rot'], 'Inv')
    assert hasattr(label['gt_rot'], 'Inv')


def test_rotation_loss_restores_plain_quaternion_storage():
    pred = pp.identity_SO3(2, 50).tensor()
    target = pp.identity_SO3(2, 50).tensor()

    loss, distance = loss_(
        loss_fc_list['Huber_loss005'], pred, target, sampling=25, dtype='rot'
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(distance, torch.zeros_like(distance))


def test_eval_errors_remain_per_sample_for_uneven_batches():
    batch, steps = 3, 4
    prediction = {
        'pos': torch.zeros(batch, steps, 3),
        'vel': torch.zeros(batch, steps, 3),
        'rot': pp.identity_SO3(batch, steps),
    }
    label = {
        'gt_pos': torch.zeros(batch, steps, 3),
        'gt_vel': torch.zeros(batch, steps, 3),
        'gt_rot': pp.identity_SO3(batch, steps),
    }
    label['gt_pos'][:, -1, 0] = torch.tensor([1.0, 2.0, 3.0])

    errors = get_RMSE(prediction, label)

    assert errors['pos'].shape == (batch,)
    torch.testing.assert_close(errors['pos'], torch.tensor([1.0, 2.0, 3.0]))


def test_phase_aware_branch_routes_scales_and_suppresses_zero_signal_phase():
    branch = PhaseAwareWaveletBranchSeq(
        levels=3, hidden_dim=32, dropout=0.0, phase_tau=0.5
    ).eval()
    signal = torch.zeros(2, 129, 6)
    with torch.inference_mode():
        feature, diagnostics = branch(signal, return_aux=True)

    assert feature.shape == (2, 129, 32)
    assert diagnostics['scale_weights'].shape == (2, 4, 129)
    assert diagnostics['phase_reliability'].shape == (2, 3, 129)
    torch.testing.assert_close(
        diagnostics['scale_weights'].sum(dim=1), torch.ones(2, 129)
    )
    assert torch.count_nonzero(diagnostics['phase_reliability']) == 0


def test_phase_aware_branch_supports_parameter_matched_no_phase_ablation():
    model = PhaseAwareWaveletBranchSeq(
        in_channels=6, levels=3, hidden_dim=32, dropout=0.0,
        phase_tau=0.5, use_phase=False,
    )
    signal = torch.randn(2, 257, 6)
    output, diagnostics = model(signal, return_aux=True)
    assert output.shape == (2, 257, 32)
    assert diagnostics['scale_weights'].shape == (2, 4, 257)
    output.square().mean().backward()
    assert all(parameter.grad is None for encoder in model.phase_encoders for parameter in encoder.parameters())


def test_phase_aware_float32_model_integrates_and_backpropagates():
    conf = _model_conf()
    conf['network'] = 'freqnet_phase_aware'
    conf['phase_tau'] = 0.5
    model = PhaseAwareFreqHybridModel(conf).float().train()
    batch, steps, pad = 1, 50, 9
    data = {
        'acc': torch.randn(batch, steps + pad, 3),
        'gyro': torch.randn(batch, steps + pad, 3) * 0.01,
        'dt': torch.full((batch, steps, 1), 0.005),
        'rot': pp.identity_SO3(batch, steps, dtype=torch.float32),
    }
    init_state = {
        'pos': torch.zeros(batch, 1, 3),
        'vel': torch.zeros(batch, 1, 3),
        'rot': pp.identity_SO3(batch, 1, dtype=torch.float32),
    }

    output = model(data, init_state)
    loss = output['pos'].square().mean() + output['correction_acc'].square().mean()
    loss = loss + output['cov'].mean()
    loss.backward()

    assert output['pos'].dtype == torch.float32
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
