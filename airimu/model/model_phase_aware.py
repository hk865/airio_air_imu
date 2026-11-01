import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward

from model.model_freq_improved import HilbertPhase, ImprovedFreqHybridModel


class PhaseAwareWaveletBranchSeq(nn.Module):
    """Amplitude-confidence-modulated, multi-scale phase representation.

    Detail bands are represented by their analytic magnitude and phase. Phase
    is attenuated where the corresponding magnitude is small, because phase is
    ill-defined close to zero amplitude. The low-pass band is retained for
    slowly varying bias/motion information. A learned scale router replaces the
    unweighted sum used by the original frequency branch.
    """

    def __init__(self, in_channels=6, wavelet='db4', levels=4, hidden_dim=128,
                 dropout=0.2, phase_tau=0.5, eps=1e-6, use_phase=True):
        super().__init__()
        if levels < 1:
            raise ValueError('levels must be at least 1')
        if phase_tau <= 0:
            raise ValueError('phase_tau must be positive')

        self.levels = levels
        self.phase_tau = phase_tau
        self.eps = eps
        self.use_phase = use_phase
        self.dwt = DWT1DForward(J=levels, wave=wavelet, mode='symmetric')
        self.hilbert = HilbertPhase()

        self.amplitude_encoders = nn.ModuleList([
            self._make_encoder(in_channels, hidden_dim, final_activation=True)
            for _ in range(levels)
        ])
        self.phase_encoders = nn.ModuleList([
            self._make_encoder(in_channels * 2, hidden_dim, final_activation=False)
            for _ in range(levels)
        ])
        self.detail_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.lowpass_encoder = self._make_encoder(
            in_channels, hidden_dim, final_activation=True
        )

        router_hidden = max(hidden_dim // 4, 16)
        self.scale_router = nn.Sequential(
            nn.Conv1d(hidden_dim, router_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(router_hidden, 1, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_encoder(in_channels, hidden_dim, final_activation):
        layers = [
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
        ]
        if final_activation:
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def _normalize_magnitude(self, magnitude):
        scale = magnitude.mean(dim=-1, keepdim=True).detach().clamp_min(self.eps)
        normalized = torch.log1p(magnitude / scale)
        reliability = magnitude / (magnitude + self.phase_tau * scale)
        return normalized, reliability

    def forward(self, x, return_aux=False):
        # x: (batch, time, channels)
        signal = x.permute(0, 2, 1)
        lowpass, details = self.dwt(signal)
        target_length = signal.shape[-1]

        scale_features = []
        reliability_maps = []
        for level, detail in enumerate(details):
            magnitude, sin_phase, cos_phase = self.hilbert(detail)
            normalized_magnitude, reliability = self._normalize_magnitude(magnitude)

            amplitude_feature = self.amplitude_encoders[level](normalized_magnitude)
            if self.use_phase:
                phase_input = torch.cat(
                    [reliability * sin_phase, reliability * cos_phase], dim=1
                )
                phase_feature = self.phase_encoders[level](phase_input)
            else:
                # Keep the tensor shape and full parameter budget unchanged so
                # this is a controlled phase ablation, not a smaller model.
                phase_feature = torch.zeros_like(amplitude_feature)
            feature = self.detail_fusion(
                torch.cat([amplitude_feature, phase_feature], dim=1)
            )
            scale_features.append(
                F.interpolate(feature, size=target_length, mode='linear', align_corners=False)
            )
            reliability_maps.append(
                F.interpolate(
                    reliability.mean(dim=1, keepdim=True),
                    size=target_length,
                    mode='linear',
                    align_corners=False,
                ).squeeze(1)
            )

        # Preserve slowly varying motion/bias information instead of discarding yl.
        lowpass_scale = lowpass.abs().mean(dim=-1, keepdim=True).detach().clamp_min(self.eps)
        normalized_lowpass = torch.sign(lowpass) * torch.log1p(lowpass.abs() / lowpass_scale)
        lowpass_feature = self.lowpass_encoder(normalized_lowpass)
        scale_features.append(
            F.interpolate(
                lowpass_feature, size=target_length, mode='linear', align_corners=False
            )
        )

        stacked = torch.stack(scale_features, dim=1)  # (B, J+1, H, T)
        logits = torch.stack(
            [self.scale_router(feature) for feature in scale_features], dim=1
        )
        scale_weights = torch.softmax(logits, dim=1)
        combined = (stacked * scale_weights).sum(dim=1)
        combined = self.dropout(combined).permute(0, 2, 1)

        if not return_aux:
            return combined
        return combined, {
            'scale_weights': scale_weights.squeeze(2),
            'phase_reliability': torch.stack(reliability_maps, dim=1),
        }


class PhaseAwareFreqHybridModel(ImprovedFreqHybridModel):
    """FreqNet with true amplitude/phase decomposition and scale routing."""

    def __init__(self, conf):
        super().__init__(conf)
        self.wavelet_branch = PhaseAwareWaveletBranchSeq(
            in_channels=6,
            wavelet=conf.get('wavelet', 'db4'),
            levels=int(conf.get('wavelet_levels', 4)),
            hidden_dim=int(conf.get('wavelet_hidden', 128)),
            dropout=conf.get('dropout', 0.2),
            phase_tau=float(conf.get('phase_tau', 0.5)),
            eps=float(conf.get('phase_eps', 1e-6)),
            use_phase=bool(conf.get('use_phase', True)),
        )
