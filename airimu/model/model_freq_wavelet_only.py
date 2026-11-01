import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DForward
from model.net import ModelBase


class HilbertTransform(nn.Module):
    """
    使用 Hilbert 变换提取幅度和相位（sin/cos 编码）
    输入: (B, C, T)
    输出: magnitude (B, C, T), sin_phase (B, C, T), cos_phase (B, C, T)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        # 构造希尔伯特变换的频率响应
        h = torch.zeros(T // 2 + 1, device=x.device)
        h[0] = 1.0
        if T % 2 == 0:
            h[1:T//2] = 2.0
            h[T//2] = 1.0
        else:
            h[1:(T+1)//2] = 2.0
        X_hilbert = X * h.unsqueeze(0).unsqueeze(0)
        x_hilbert = torch.fft.irfft(X_hilbert, n=T, dim=-1)
        analytic_signal = x + 1j * x_hilbert
        magnitude = torch.abs(analytic_signal)
        phase = torch.angle(analytic_signal)
        # 使用 sin/cos 编码相位，避免周期跳变
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        return magnitude, sin_phase, cos_phase


class SimpleWaveletBranch(nn.Module):
    """
    改进版小波分支：
    - 使用所有 J+1 个频段（yl + yh[0..J-1]）
    - 相位信息采用 sin/cos 编码
    - 轻量级频段注意力融合（样本级动态加权）
    """
    def __init__(self, in_channels=6, wavelet='db4', J=4,
                 hidden_dim=128, use_phase=True, dropout=0.2):
        super().__init__()
        self.J = J
        self.use_phase = use_phase
        self.hidden_dim = hidden_dim

        # 小波分解（固定小波，可后续改为可学习）
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')

        # 可选的相位编码器（输入通道变为 2，因为 sin 和 cos）
        if use_phase:
            self.hilbert = HilbertTransform()
            self.phase_encoder = nn.Sequential(
                nn.Conv1d(in_channels * 2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        # 各频段共享的卷积处理器
        self.freq_processor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 频段注意力模块
        self.num_bands = J + 1  # yl + yh[0..J-1]
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # 时序建模（关键！添加LSTM增强时序能力）
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x, return_weights=False):
        """
        输入: x (B, T, C)
        输出: (B, T, hidden_dim)
        可选: return_weights=True 返回频段权重
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, T)

        # 小波分解
        yl, yh = self.dwt(x)

        # 收集所有频段的特征
        all_feats = []
        raw_feats = []

        # 处理近似分量 yl
        feat = self.freq_processor(yl)
        raw_feats.append(feat)
        feat_up = F.interpolate(feat, size=T, mode='linear', align_corners=False)
        all_feats.append(feat_up)

        # 处理所有细节分量
        for j in range(self.J):
            feat = self.freq_processor(yh[j])
            raw_feats.append(feat)
            feat_up = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            all_feats.append(feat_up)

        # 频段注意力融合
        band_representations = [feat.mean(dim=-1) for feat in raw_feats]
        band_stack = torch.stack(band_representations, dim=1)
        weights_logits = self.attn_mlp(band_stack)
        weights = torch.softmax(weights_logits.squeeze(-1), dim=1)

        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
        stacked_feats = torch.stack(all_feats, dim=1)
        fused = (weights_expanded * stacked_feats).sum(dim=1)

        # 添加相位特征
        if self.use_phase:
            magnitude, sin_phase, cos_phase = self.hilbert(x)
            phase_dual = torch.cat([sin_phase, cos_phase], dim=1)
            phase_feat = self.phase_encoder(phase_dual)
            fused = fused + phase_feat

        fused = self.dropout(fused)

        # 时序建模（关键改进！）
        fused = fused.permute(0, 2, 1)  # (B, T, hidden_dim)
        lstm_out, _ = self.temporal_lstm(fused)  # (B, T, hidden_dim)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, hidden_dim, T)

        output = self.fusion(lstm_out)  # (B, hidden_dim, T)

        if return_weights:
            return output.permute(0, 2, 1), weights
        return output.permute(0, 2, 1)


class WaveletOnlyModel(ModelBase):
    """
    纯小波模型：去掉 deep 分支，只用频域分支完成全部任务
    核心改进：
    1. 完整的频段覆盖（yl + 所有 yh）
    2. 样本级频段注意力
    3. sin/cos 相位编码
    4. 添加 LSTM 时序建模
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        hidden_dim = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.3)

        self.pad_len = 9

        # 单分支：改进的小波分支
        self.wavelet_branch = SimpleWaveletBranch(
            in_channels=input_dim,
            wavelet='db4',
            J=4,  # 使用 J=4 覆盖更低频率
            hidden_dim=hidden_dim,
            use_phase=True,
            dropout=dropout
        )

        # 输出头
        self.correction_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

        nn.init.constant_(self.logvar_head.weight, 0.01)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def inference(self, data, return_freq_weights=False):
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)

        # 单分支处理
        if return_freq_weights:
            features, freq_weights = self.wavelet_branch(feature, return_weights=True)
        else:
            features = self.wavelet_branch(feature)
            freq_weights = None

        # 输出
        correction = self.correction_head(features)
        logvar = self.logvar_head(features)

        # 去除 padding
        correction = correction[:, self.pad_len:, :]

        cov_state = {'acc_cov': None, 'gyro_cov': None}
        if self.conf.propcov:
            cov_state['acc_cov'] = torch.exp(logvar[:, self.pad_len:, :3] - 5.0)
            cov_state['gyro_cov'] = torch.exp(logvar[:, self.pad_len:, 3:] - 5.0)

        result = {
            **cov_state,
            'correction_acc': correction[..., :3],
            'correction_gyro': correction[..., 3:]
        }

        if freq_weights is not None:
            result['freq_weights'] = freq_weights

        return result

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        data['corrected_acc'] = data['acc'][:, self.pad_len:, :] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'][:, self.pad_len:, :] + inference_state['correction_gyro']

        out_state = self.integrate(init_state=init_state, data=data, cov_state=inference_state)

        return {
            **out_state,
            'correction_acc': inference_state['correction_acc'],
            'correction_gyro': inference_state['correction_gyro'],
            'corrected_acc': data['corrected_acc'],
            'corrected_gyro': data['corrected_gyro']
        }
