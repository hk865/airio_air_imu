import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DForward


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
        x_h = torch.fft.ifft(X_h, dim=-1).imag  # 希尔伯特变换结果

        # 6. 解析信号幅度（加eps防止除零）
        mag = torch.sqrt(x ** 2 + x_h ** 2 + self.eps)

        phase = torch.atan2(x_h, x)

        # 相位三角函数
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        return mag, sin_phase, cos_phase


class WaveletBranchSeqNoInterp(nn.Module):
    """
    小波分支 - 无插值版本，输出下采样后的长度
    """
    def __init__(self, in_channels=6, wavelet='db4', J=3, hidden_dim=48, dropout=0.3):
        super().__init__()
        self.J = J
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')
        self.hilbert = HilbertPhase()
        self.total_stride = 2**J  # 小波分解的总下采样倍数

        # 中等幅度编码器
        self.level_convs = nn.ModuleList()
        for _ in range(J):
            self.level_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            ))

        # 中等相位编码器
        self.phase_convs = nn.ModuleList()
        for _ in range(J):
            self.phase_convs.append(nn.Sequential(
                nn.Conv1d(in_channels * 2, 32, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            ))

        # 幅度+相位融合
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, T)
        _, yh = self.dwt(x)

        level_feats = []
        # 使用最低频的小波系数长度作为输出长度
        T_out = yh[-1].shape[-1]
        
        for j in range(self.J):
            # 幅度特征
            mag_feat = self.level_convs[j](yh[j])

            # 相位特征
            _, sin_p, cos_p = self.hilbert(yh[j])
            phase_input = torch.cat([sin_p, cos_p], dim=1)
            phase_feat = self.phase_convs[j](phase_input)

            # 下采样到最低频长度（而不是上采样回原始长度）
            if mag_feat.shape[-1] != T_out:
                mag_down = F.interpolate(mag_feat, size=T_out, mode='linear', align_corners=False)
                phase_down = F.interpolate(phase_feat, size=T_out, mode='linear', align_corners=False)
            else:
                mag_down = mag_feat
                phase_down = phase_feat

            # 拼接幅度和相位
            combined = torch.cat([mag_down, phase_down], dim=1)
            fused = self.fusion(combined)
            level_feats.append(fused)

        combined = torch.stack(level_feats, dim=0).sum(dim=0)
        combined = self.dropout(combined)
        return combined.permute(0, 2, 1)  # (B, T_out, hidden_dim)


class DeepBranchSeqNoInterp(nn.Module):
    """
    深度分支 - 无插值版本，输出下采样后的长度
    """
    def __init__(self, in_channels=6, hidden_size=48, num_layers=1, out_dim=48,
                 down_ratios=(3, 3), dropout=0.3):
        super().__init__()
        self.down_ratios = down_ratios
        self.total_stride = 1
        for r in down_ratios:
            self.total_stride *= r

        # 下采样卷积块
        layers = []
        ch = in_channels
        for stride in down_ratios:
            layers.append(nn.Conv1d(ch, ch * 2, kernel_size=5, stride=stride, padding=2))
            layers.append(nn.BatchNorm1d(ch * 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            ch = ch * 2
        self.down_conv = nn.Sequential(*layers)

        # LSTM
        self.lstm = nn.LSTM(input_size=ch, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0)

        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x_down = self.down_conv(x)  # (B, C_down, T_down)
        x_down = x_down.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_down)  # (B, T_down, hidden_size)
        # 移除上采样操作，直接输出下采样后的长度
        lstm_out = self.fc_dropout(lstm_out)
        out = self.fc(lstm_out)  # (B, T_down, out_dim)
        return out


class FreqHybridModelMediumNoInterp(nn.Module):
    """
    中等参数量版本 - 无插值版本
    输出长度 = 输入长度 // 9 (因为 down_ratios=(3,3) 和 J=3 小波分解)
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        wavelet_out = 48
        deep_hidden = 48
        d_model = 64
        output_dim = 3
        dropout = getattr(conf, 'dropout', 0.3)

        # 小波分支 - 无插值
        self.wavelet_branch = WaveletBranchSeqNoInterp(
            in_channels=input_dim, wavelet='db4',
            J=3, hidden_dim=wavelet_out, dropout=dropout
        )

        # 深度分支 - 无插值，使用 (3,3) 下采样与CodeNet一致
        self.deep_branch = DeepBranchSeqNoInterp(
            in_channels=input_dim, hidden_size=deep_hidden,
            num_layers=1, out_dim=deep_hidden,
            down_ratios=(3, 3), dropout=dropout
        )

        # 投影和融合
        self.proj_w = nn.Linear(wavelet_out, d_model)
        self.proj_d = nn.Linear(deep_hidden, d_model)
        self.proj_drop = nn.Dropout(dropout)

        # 门控融合
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # 时序建模 GRU
        self.gru1 = nn.GRU(input_size=d_model, hidden_size=64,
                           num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64,
                           num_layers=1, batch_first=True, bidirectional=True)

        # 速度解码器
        self.veldecoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

        # 协方差解码器
        self.velcov_decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

    def encoder(self, x):
        """编码器：提取特征"""
        # 双分支特征提取 - 输出长度已经一致（都是下采样后的长度）
        w_feat = self.wavelet_branch(x)
        d_feat = self.deep_branch(x)

        # 确保两个分支输出长度一致
        min_len = min(w_feat.shape[1], d_feat.shape[1])
        w_feat = w_feat[:, :min_len, :]
        d_feat = d_feat[:, :min_len, :]

        # 投影到相同维度
        w_feat = self.proj_w(w_feat)
        d_feat = self.proj_d(d_feat)
        w_feat = self.proj_drop(w_feat)
        d_feat = self.proj_drop(d_feat)

        # 门控融合
        combined = torch.cat([w_feat, d_feat], dim=-1)
        gate = self.sigmoid(self.gate_linear(combined))
        fused = gate * w_feat + (1 - gate) * d_feat
        fused = self.dropout(fused)

        # 时序建模
        x, _ = self.gru1(fused)
        x, _ = self.gru2(x)
        return x

    def cov_decoder(self, x):
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def decoder(self, x):
        vel = self.veldecoder(x)
        return vel

    def get_label(self, gt_label):
        """
        对齐标签长度与模型输出长度
        模型输出长度 = 输入长度 // 9
        """
        # 计算输出长度
        B, T, C = gt_label.shape
        s_idx = (7 - 3) + 3 * (7 - 1 - 3) + 1  # 对应 CodeNet 的计算方式
        output_len = (T - 1 - 1) // 9 + 1
        
        select_label = gt_label[:, s_idx::9, :]
        
        # 补齐长度差异
        diff = output_len - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:, -1:, :].repeat(1, diff, 1)), dim=1)
        elif diff < 0:
            select_label = select_label[:, :output_len, :]
            
        return select_label

    def forward(self, data, rot=None):
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature)
        net_vel = self.decoder(feature)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        return {"cov": cov, "net_vel": net_vel}


class FreqHybridModelMediumRotNoInterp(FreqHybridModelMediumNoInterp):
    """
    中等参数量带旋转编码版本 - 无插值版本
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        rot_dim = 3
        wavelet_out = 48
        deep_hidden = 48
        d_model = 64
        output_dim = 3
        dropout = getattr(conf, 'dropout', 0.3)

        # Mask参数
        self.rot_mask_ratio = getattr(conf, 'rot_mask_ratio', 0.0)
        self.rot_mask_value = getattr(conf, 'rot_mask_value', 0.0)

        # IMU特征编码器 - 无插值
        self.wavelet_branch = WaveletBranchSeqNoInterp(
            in_channels=input_dim, wavelet='db4',
            J=3, hidden_dim=wavelet_out, dropout=dropout
        )
        self.deep_branch = DeepBranchSeqNoInterp(
            in_channels=input_dim, hidden_size=deep_hidden,
            num_layers=1, out_dim=deep_hidden,
            down_ratios=(3, 3), dropout=dropout
        )

        # 旋转特征编码器：仅深度分支 - 无插值
        self.rot_encoder = DeepBranchSeqNoInterp(
            in_channels=rot_dim, hidden_size=deep_hidden,
            num_layers=1, out_dim=deep_hidden,
            down_ratios=(3, 3), dropout=dropout
        )

        # 投影层
        self.proj_w = nn.Linear(wavelet_out, d_model)
        self.proj_d = nn.Linear(deep_hidden, d_model)
        self.proj_rot = nn.Linear(deep_hidden, d_model)
        self.proj_drop = nn.Dropout(dropout)

        # 融合层
        self.fusion_fcn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        self.fusion_bn = nn.BatchNorm1d(d_model)

        # 时序建模 GRU
        self.gru1 = nn.GRU(input_size=d_model, hidden_size=64,
                           num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64,
                           num_layers=1, batch_first=True, bidirectional=True)

        # 解码器
        self.veldecoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

        self.velcov_decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )

    def apply_rot_mask(self, rot, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.rot_mask_ratio

        if mask_ratio <= 0 or not self.training:
            return rot, None

        B, T, C = rot.shape
        mask = torch.rand(B, T, device=rot.device) > mask_ratio
        mask = mask.unsqueeze(-1).expand(-1, -1, C)
        masked_rot = torch.where(mask, rot, torch.tensor(self.rot_mask_value, device=rot.device))
        return masked_rot, mask

    def encoder(self, feature, rot, mask_ratio=None):
        """编码器"""
        # IMU特征编码（双分支）- 无插值
        w_feat = self.wavelet_branch(feature)
        d_feat = self.deep_branch(feature)
        
        # 确保两个分支输出长度一致
        min_len = min(w_feat.shape[1], d_feat.shape[1])
        w_feat = w_feat[:, :min_len, :]
        d_feat = d_feat[:, :min_len, :]
        
        w_feat = self.proj_w(w_feat)
        d_feat = self.proj_d(d_feat)

        # IMU门控融合
        combined_imu = torch.cat([w_feat, d_feat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(combined_imu))
        imu_fused = gate * w_feat + (1 - gate) * d_feat
        imu_fused = self.proj_drop(imu_fused)

        # 旋转特征编码（仅深度分支，带mask）- 无插值
        if self.training and mask_ratio is not None:
            rot, _ = self.apply_rot_mask(rot, mask_ratio)
        elif self.training and self.rot_mask_ratio > 0:
            rot, _ = self.apply_rot_mask(rot)

        rot_feat = self.rot_encoder(rot)
        rot_feat = self.proj_rot(rot_feat)
        
        # 确保旋转特征长度与IMU特征一致
        min_len = min(imu_fused.shape[1], rot_feat.shape[1])
        imu_fused = imu_fused[:, :min_len, :]
        rot_feat = rot_feat[:, :min_len, :]

        # 拼接IMU和旋转特征
        combined = torch.cat([imu_fused, rot_feat], dim=-1)

        # 融合降维
        x = self.fusion_fcn(combined)
        x = self.fusion_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.gelu(x)

        # 时序建模
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def forward(self, data, rot=None, mask_ratio=None):
        assert rot is not None, "FreqHybridModelMediumRotNoInterp requires rot input"

        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature, rot, mask_ratio)
        net_vel = self.decoder(feature)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        return {"cov": cov, "net_vel": net_vel}
