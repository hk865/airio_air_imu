import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_wavelets import DWT1DForward,DTCWTForward,DWT1DInverse
from model.net import ModelBase
from model.net import ModelBase


class HilbertPhase(nn.Module):
    """
    正确的离散希尔伯特变换实现
    输入: (B, C, T) 实信号 / 支持任意前导维度 (..., T)
    输出: 幅度, sin(瞬时相位), cos(瞬时相位)
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps  # 防止除零的极小值

    def forward(self, x: torch.Tensor):
        # 1. 输入校验：必须是实信号
        if torch.is_complex(x):
            raise ValueError("希尔伯特变换仅支持实信号输入")

        # 2. 支持任意前导维度，仅对最后一维变换
        T = x.shape[-1]
        device = x.device

        # 3. 傅里叶变换
        X = torch.fft.fft(x, dim=-1)

        # 4. 构造标准离散希尔伯特频域乘子（修复奈奎斯特bug）
        f = torch.fft.fftfreq(T, device=device)
        H = -1j * torch.sign(f)

        # 奈奎斯特频率置0（偶数长度信号）
        if T % 2 == 0:
            H[T // 2] = 0

        # 5. 频域相乘 + 逆变换
        X_h = X * H
        x_h = torch.fft.ifft(X_h, dim=-1).real  # 希尔伯特变换结果

        # 6. 解析信号幅度（加eps防止除零）
        mag = torch.sqrt(x ** 2 + x_h ** 2 + self.eps)

        phase = torch.atan2(x_h, x)

        # 相位三角函数
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        return mag, sin_phase, cos_phase


class WaveletBranchSeq(nn.Module):
    def __init__(self, in_channels=6, wavelet='db4', J=4, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.J = J
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')
        self.idwt = DWT1DInverse(wave=wavelet, mode='symmetric')
        self.hilbert = HilbertPhase()

        # 每个尺度（近似 + J个细节）独立编码器
        self.scale_convs = nn.ModuleList()
        for _ in range(J + 1):
            self.scale_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            ))

        self.phase_convs = nn.ModuleList()
        for _ in range(J + 1):
            self.phase_convs.append(nn.Sequential(
                nn.Conv1d(in_channels * 2, 64, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            ))

        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)          # (B, C, T)
        B, C, T = x.shape
        #print(f"WaveletBranch input length: {T}")   # 调试用，可删除

        yl, yh = self.dwt(x)            # 小波分解
        zero_details = [torch.zeros_like(d) for d in yh]

        scale_feats = []
        for idx in range(self.J + 1):
            if idx == 0:
                coeffs_for_recon = (yl, zero_details)
            else:
                detail_idx = idx - 1
                cur_details = [torch.zeros_like(d) if i != detail_idx else yh[i]
                               for i, d in enumerate(yh)]
                coeffs_for_recon = (torch.zeros_like(yl), cur_details)

            recon_signal = self.idwt(coeffs_for_recon)   # (B, C, T_recon)
            # 强制对齐到原始长度（截断或插值）
            if recon_signal.size(-1) != T:
                recon_signal = F.interpolate(recon_signal, size=T, mode='linear', align_corners=False)
            else:
                recon_signal = recon_signal[:, :, :T]   # 安全截断

            # 幅度特征
            mag_feat = self.scale_convs[idx](recon_signal)
            # 相位特征
            _, sin_p, cos_p = self.hilbert(recon_signal)
            phase_input = torch.cat([sin_p, cos_p], dim=1)
            phase_feat = self.phase_convs[idx](phase_input)
            # 融合
            combined = torch.cat([mag_feat, phase_feat], dim=1)
            fused = self.fusion(combined)
            scale_feats.append(fused)

        combined = torch.stack(scale_feats, dim=0).sum(dim=0)
        combined = self.dropout(combined)
        return combined.permute(0, 2, 1)   # (B, T, hidden_dim)

class DeepBranchSeq(nn.Module):
    def __init__(self, in_channels=6, hidden_size=128, num_layers=2, out_dim=128,
                 down_ratios=(2, 2), dropout=0.2):
        super().__init__()
        self.down_ratios = down_ratios
        self.total_stride = 1
        for r in down_ratios:
            self.total_stride *= r

        layers = []
        ch = in_channels
        for stride in down_ratios:
            layers.append(nn.Conv1d(ch, ch * 2, kernel_size=5, stride=stride, padding=2))
            layers.append(nn.BatchNorm1d(ch * 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            ch = ch * 2
        self.down_conv = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=ch, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        B, T, C = x.shape
        #print(f"DeepBranch input length: {T}")   # 调试用
        x = x.permute(0, 2, 1)
        x_down = self.down_conv(x)
        x_down = x_down.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_down)
        lstm_out = lstm_out.permute(0, 2, 1)
        up_out = F.interpolate(lstm_out, size=T, mode='linear', align_corners=False)
        up_out = up_out.permute(0, 2, 1)
        up_out = self.fc_dropout(up_out)
        out = self.fc(up_out)
        # 确保输出长度严格等于 T（可能由于插值数值误差产生 ±1，做一次保护）
        if out.size(1) != T:
            out = out[:, :T, :]
        return out


class ImprovedFreqHybridModelWithUpdate(ModelBase):
    """
    最小化改进版本：
    - 基于model_freq的可靠结构
    - 只在小波分支添加相位特征
    - 保持其他所有组件不变
    """

    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        wavelet_out = 128
        deep_hidden = 128
        d_model = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.3)

        # 保持model_freq的padding设置
        self.pad_len = 9

        # 改进的小波分支（添加相位）
        self.wavelet_branch = WaveletBranchSeq(
            in_channels=input_dim,
            J=4, hidden_dim=wavelet_out, dropout=dropout
        )

        # 保持model_freq的深度分支不变
        self.deep_branch = DeepBranchSeq(
            in_channels=input_dim, hidden_size=deep_hidden,
            num_layers=2, out_dim=deep_hidden,
            down_ratios=(2, 2), dropout=dropout
        )

        # 保持model_freq的投影和融合方式
        self.proj_w = nn.Linear(wavelet_out, d_model)
        self.proj_d = nn.Linear(deep_hidden, d_model)
        self.proj_drop = nn.Dropout(dropout)

        # 保持model_freq的门控融合
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # 保持model_freq的输出头
        self.correction_head = nn.Linear(d_model, output_dim)
        self.logvar_head = nn.Linear(d_model, output_dim)

        nn.init.constant_(self.logvar_head.weight, 0.01)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def inference(self, data):
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)

        w_feat = self.wavelet_branch(feature)
        d_feat = self.deep_branch(feature)

        # 动态对齐长度（防止任何分支输出长度不一致）
        min_len = min(w_feat.size(1), d_feat.size(1))
        if w_feat.size(1) != min_len:
            w_feat = w_feat[:, :min_len, :]
        if d_feat.size(1) != min_len:
            d_feat = d_feat[:, :min_len, :]

        w_feat = self.proj_w(w_feat)
        d_feat = self.proj_d(d_feat)
        w_feat = self.proj_drop(w_feat)
        d_feat = self.proj_drop(d_feat)

        combined = torch.cat([w_feat, d_feat], dim=-1)
        gate = self.sigmoid(self.gate_linear(combined))
        fused = gate * w_feat + (1 - gate) * d_feat
        fused = self.dropout(fused)

        correction = self.correction_head(fused)
        logvar = self.logvar_head(fused)

        # 去除前面的 padding
        correction = correction[:, self.pad_len:, :]
        logvar = logvar[:, self.pad_len:, :]

        cov_state = {'acc_cov': None, 'gyro_cov': None}
        if self.conf.propcov:
            cov_state['acc_cov'] = torch.exp(logvar[:, :, :3] - 5.0)
            cov_state['gyro_cov'] = torch.exp(logvar[:, :, 3:] - 5.0)

        return {**cov_state, 'correction_acc': correction[..., :3], 'correction_gyro': correction[..., 3:]}

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

