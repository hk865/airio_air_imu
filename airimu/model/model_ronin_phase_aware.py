import torch
import torch.nn as nn

from model.model_freq_improved import DeepBranchSeq
from model.model_phase_aware import PhaseAwareWaveletBranchSeq


class PhaseAwareVelocityModel(nn.Module):
    """Phase-aware horizontal-velocity network for RoNIN-style data.

    Unlike the UAV model, this module does not predict IMU corrections and does
    not run strapdown integration. RoNIN's target is body velocity while the IMU
    phone can move independently in a subject's hand, so a direct velocity head
    is the task-consistent adaptation.
    """

    def __init__(self, conf):
        super().__init__()
        input_dim = int(conf.get('input_dim', 6))
        output_dim = int(conf.get('output_dim', 2))
        wavelet_hidden = int(conf.get('wavelet_hidden', 128))
        deep_hidden = int(conf.get('deep_hidden', 128))
        fusion_dim = int(conf.get('fusion_dim', 128))
        dropout = float(conf.get('dropout', 0.2))

        self.wavelet_branch = PhaseAwareWaveletBranchSeq(
            in_channels=input_dim,
            wavelet=conf.get('wavelet', 'db4'),
            levels=int(conf.get('wavelet_levels', 4)),
            hidden_dim=wavelet_hidden,
            dropout=dropout,
            phase_tau=float(conf.get('phase_tau', 0.5)),
            eps=float(conf.get('phase_eps', 1e-6)),
        )
        self.deep_branch = DeepBranchSeq(
            in_channels=input_dim,
            hidden_size=deep_hidden,
            num_layers=int(conf.get('lstm_layers', 2)),
            out_dim=deep_hidden,
            down_ratios=tuple(conf.get('down_ratios', (2, 2))),
            dropout=dropout,
        )

        self.project_wavelet = nn.Linear(wavelet_hidden, fusion_dim)
        self.project_deep = nn.Linear(deep_hidden, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.velocity_head = nn.Linear(fusion_dim, output_dim)

    def forward(self, features, return_aux=False):
        """Predict per-sample horizontal velocity from ``(B, T, 6)`` IMU."""
        if features.ndim != 3 or features.shape[-1] != 6:
            raise ValueError(
                f'features must have shape (batch, time, 6), got {tuple(features.shape)}'
            )

        if return_aux:
            wavelet, wavelet_aux = self.wavelet_branch(features, return_aux=True)
        else:
            wavelet = self.wavelet_branch(features)
            wavelet_aux = None
        deep = self.deep_branch(features)

        wavelet = self.project_wavelet(wavelet)
        deep = self.project_deep(deep)
        gate = self.gate(torch.cat([wavelet, deep], dim=-1))
        fused = self.dropout(gate * wavelet + (1.0 - gate) * deep)
        velocity = self.velocity_head(fused)

        if not return_aux:
            return velocity
        return velocity, {
            **wavelet_aux,
            'branch_gate': gate,
        }
