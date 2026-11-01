
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DForward
import pypose as pp


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


class WaveletBranchSeq(nn.Module):
    """轻量级小波分支，J=4，控制参数量"""
    def __init__(self, in_channels=6, wavelet='db4', J=4, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.J = J
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')
        self.hilbert = HilbertPhase()

        # 极轻量级幅度编码器
        self.level_convs = nn.ModuleList()
        for _ in range(J):
            self.level_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, 24, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(24, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            ))

        # 极轻量级相位编码器
        self.phase_convs = nn.ModuleList()
        for _ in range(J):
            self.phase_convs.append(nn.Sequential(
                nn.Conv1d(in_channels * 2, 24, kernel_size=3, padding=1),
                nn.Tanh(),
                nn.Conv1d(24, hidden_dim, kernel_size=3, padding=1),
            ))

        # 幅度+相位融合
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, yh = self.dwt(x)
        B, C, T = x.shape

        level_feats = []
        for j in range(self.J):
            mag_feat = self.level_convs[j](yh[j])
            _, sin_p, cos_p = self.hilbert(yh[j])
            phase_input = torch.cat([sin_p, cos_p], dim=1)
            phase_feat = self.phase_convs[j](phase_input)
            mag_up = F.interpolate(mag_feat, size=T, mode='linear', align_corners=False)
            phase_up = F.interpolate(phase_feat, size=T, mode='linear', align_corners=False)
            combined = torch.cat([mag_up, phase_up], dim=1)
            fused = self.fusion(combined)
            level_feats.append(fused)

        combined = torch.stack(level_feats, dim=0).sum(dim=0)
        combined = self.dropout(combined)
        return combined.permute(0, 2, 1)


class DeepBranchLSTM(nn.Module):
    """深度分支（下采样卷积 + LSTM时序处理）"""
    def __init__(self, in_channels=6, hidden_size=64, num_layers=2, out_dim=64, dropout=0.2):
        super().__init__()
        
        # 下采样卷积块：T -> T//4
        self.down_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # LSTM时序处理
        self.lstm = nn.LSTM(input_size=48, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # 上采样回原始长度
        self.fc = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # 下采样
        x = x.permute(0, 2, 1)  # (B, C, T)
        x_down = self.down_conv(x)  # (B, 48, T//4)
        
        # LSTM处理
        x_down = x_down.permute(0, 2, 1)  # (B, T//4, 48)
        lstm_out, _ = self.lstm(x_down)  # (B, T//4, hidden_size)
        
        # 上采样回T
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, hidden_size, T//4)
        up_out = F.interpolate(lstm_out, size=T, mode='linear', align_corners=False)  # (B, hidden_size, T)
        up_out = up_out.permute(0, 2, 1)  # (B, T, hidden_size)
        
        up_out = self.dropout(up_out)
        out = self.fc(up_out)  # (B, T, out_dim)
        return out


class FreqLSTMModel(nn.Module):
    """
    小波分支 + 深度LSTM分支（处理IMU时序）
    - 输出 net_vel 和 cov (形状 B, T//9, 3)
    - 提供 get_label 方法用于标签对齐
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        wavelet_out = 32
        deep_out = 32
        d_model = 64  # 小波32 + 深度32 = 64
        output_dim = 3
        dropout = conf.get('dropout', 0.3)

        # 网络结构参数（用于 get_label 计算）
        self.k_list = [7, 7]
        self.p_list = [3, 3]
        self.s_list = [3, 3]

        # 小波分支 J=4
        self.wavelet_branch = WaveletBranchSeq(
            in_channels=input_dim, wavelet='db4',
            J=4, hidden_dim=wavelet_out, dropout=dropout
        )

        # 深度LSTM分支（带时序处理）
        self.deep_branch = DeepBranchLSTM(
            in_channels=input_dim, hidden_size=48, 
            num_layers=2, out_dim=deep_out, dropout=dropout
        )

        # 投影层
        self.proj_drop = nn.Dropout(dropout)

        # 下采样卷积：将 T 下采样到 T//9
        self.down_conv1 = nn.Sequential(
            nn.Conv1d(d_model, 48, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 特征处理卷积（无GRU）：先增维后缩维
        self.feature_conv = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出头
        self.veldecoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

        self.velcov_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

    def encoder(self, x):
        """编码器：小波特征 + 深度LSTM特征，融合后下采样"""
        # 双分支特征提取
        w_feat = self.wavelet_branch(x)  # (B, T, 32)
        d_feat = self.deep_branch(x)     # (B, T, 32)
        
        # 拼接融合
        feat = torch.cat([w_feat, d_feat], dim=-1)  # (B, T, 64)
        feat = self.proj_drop(feat)

        # 下采样：T -> T//9
        feat = feat.permute(0, 2, 1)
        feat = self.down_conv1(feat)
        feat = self.down_conv2(feat)

        # 卷积特征处理（先增维 64->96 后缩维 96->64）
        feat = self.feature_conv(feat)
        feat = feat.permute(0, 2, 1)

        return feat

    def decoder(self, x):
        """速度解码器"""
        vel = self.veldecoder(x)
        return vel

    def cov_decoder(self, x):
        """协方差解码器"""
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def get_label(self, gt_label):
        """标签对齐"""
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:, -1:, :].repeat(1, diff, 1)), dim=1)
        return select_label

    def forward(self, data, rot=None):
        """前向传播"""
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature)
        net_vel = self.decoder(feature)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        return {"cov": cov, "net_vel": net_vel}


class FreqLSTMModelwithRot(nn.Module):
    """
    小波分支 + 深度LSTM分支 + 旋转分支（轻量卷积）
    输出形状 (B, T//9, 3) 与 CodeNetMotionwithRot 相同
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        rot_dim = 3
        wavelet_out = 32
        deep_out = 32
        rot_out = 32
        d_model = 96  # 小波32 + 深度32 + 旋转32 = 96
        output_dim = 3
        dropout = conf.get('dropout', 0.3)

        # 网络结构参数
        self.k_list = [7, 7]
        self.p_list = [3, 3]
        self.s_list = [3, 3]

        # 小波分支 J=4（IMU）
        self.wavelet_branch = WaveletBranchSeq(
            in_channels=input_dim, wavelet='db4',
            J=4, hidden_dim=wavelet_out, dropout=dropout
        )

        # 深度LSTM分支（IMU时序处理）
        self.deep_branch = DeepBranchLSTM(
            in_channels=input_dim, hidden_size=48,
            num_layers=2, out_dim=deep_out, dropout=dropout
        )

        # 旋转分支（轻量卷积，无时序处理）
        self.rot_branch = WaveletBranchSeq(
            in_channels=rot_dim, wavelet='db4',
            J=4, hidden_dim=rot_out, dropout=dropout
        )

        # 投影层
        self.proj_drop = nn.Dropout(dropout)

        # 下采样卷积：将 T 下采样到 T//9
        self.down_conv1 = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 特征处理卷积：先增维后缩维
        self.feature_conv = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出头
        self.veldecoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

        self.velcov_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )

    def encoder(self, feature, rot):
        """编码器：小波 + 深度LSTM + 旋转，融合后下采样"""
        # 三分支特征提取
        w_feat = self.wavelet_branch(feature)  # (B, T, 32)
        d_feat = self.deep_branch(feature)     # (B, T, 32)
        r_feat = self.rot_branch(rot)          # (B, T, 32)
        
        # 拼接融合
        feat = torch.cat([w_feat, d_feat, r_feat], dim=-1)  # (B, T, 96)
        feat = self.proj_drop(feat)

        # 下采样：T -> T//9
        feat = feat.permute(0, 2, 1)
        feat = self.down_conv1(feat)
        feat = self.down_conv2(feat)

        # 卷积特征处理（先增维 64->96 后缩维 96->64）
        feat = self.feature_conv(feat)
        feat = feat.permute(0, 2, 1)

        return feat

    def decoder(self, x):
        """速度解码器"""
        vel = self.veldecoder(x)
        return vel

    def cov_decoder(self, x):
        """协方差解码器"""
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def get_label(self, gt_label):
        """标签对齐"""
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:, -1:, :].repeat(1, diff, 1)), dim=1)
        return select_label

    def forward(self, data, rot=None):
        """
        前向传播
        Args:
            data: dict 包含 'acc' 和 'gyro'
            rot: 旋转输入 (B, T, 3)
        Returns:
            dict: {'net_vel': velocity, 'cov': covariance}
        """
        assert rot is not None, "FreqLSTMModelwithRot requires rot input"
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature, rot)
        net_vel = self.decoder(feature)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        return {"cov": cov, "net_vel": net_vel}
