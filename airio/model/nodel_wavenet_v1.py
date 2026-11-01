import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DForward


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
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)
        return magnitude, sin_phase, cos_phase


class SimpleWaveletBranch(nn.Module):
    """
    改进版小波分支（最终）：
    - 频段专用卷积
    - 子带独立 Hilbert 变换，幅度+相位融合
    - 注意力权重基于融合特征（而非纯幅度）
    - 两层 GRU（128→256→128）
    """
    def __init__(self, in_channels=6, wavelet='db4', J=4,
                 hidden_dim=128, use_phase=True, dropout=0.2):
        super().__init__()
        self.J = J
        self.use_phase = use_phase
        self.hidden_dim = hidden_dim

        # 小波分解
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')

        # 频段专用卷积
        self.num_bands = J + 1
        self.freq_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            ) for _ in range(self.num_bands)
        ])

        # 相位编码（子带共享）
        if use_phase:
            self.hilbert = HilbertTransform()
            self.phase_encoder = nn.Sequential(
                nn.Conv1d(in_channels * 2, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        # 频段注意力模块（输入仍为 hidden_dim，因为融合特征也是该维度）
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 时序建模
        self.gru1 = nn.GRU(hidden_dim, hidden_dim*2, 1, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim*2, hidden_dim, 1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim*2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        """
        输入: x (B, T, C)
        输出: (B, T, hidden_dim)
        可选: return_weights=True 返回频段权重（基于融合特征）
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # (B, C, T)

        # 小波分解
        yl, yh = self.dwt(x)    # yl: (B, C, T/2^J), yh: list of J tensors

        # 存储每个子带的融合特征（上采样后）
        fused_feats = []

        def process_subband(coeff, idx):
            # 幅度特征
            feat = self.freq_processors[idx](coeff)               # (B, hidden_dim, len_coeff)
            # 上采样到原长 T
            up_feat = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            # 如果使用相位，添加相位特征
            if self.use_phase:
                # 对子带系数做 Hilbert 变换（获得幅度、sin/cos相位）
                mag, sin_phase, cos_phase = self.hilbert(coeff)   # each (B, C, len_coeff)
                phase_dual = torch.cat([sin_phase, cos_phase], dim=1)   # (B, 2*C, len_coeff)
                phase_feat = self.phase_encoder(phase_dual)             # (B, hidden_dim, len_coeff)
                phase_up = F.interpolate(phase_feat, size=T, mode='linear', align_corners=False)
                # 融合：简单加法（可改为门控，但加法已足够）
                up_feat = up_feat + phase_up
            return up_feat

        # 处理近似分量 yl
        fused_feats.append(process_subband(yl, 0))

        # 处理所有细节分量
        for j in range(self.J):
            fused_feats.append(process_subband(yh[j], j+1))

        # ---- 频段注意力（基于融合特征）----
        # 对每个子带，全局平均池化 -> (B, hidden_dim)
        band_representations = [feat.mean(dim=-1) for feat in fused_feats]
        band_stack = torch.stack(band_representations, dim=1)   # (B, num_bands, hidden_dim)
        weights_logits = self.attn_mlp(band_stack)              # (B, num_bands, 1)
        weights = torch.softmax(weights_logits.squeeze(-1), dim=1)  # (B, num_bands)

        # 加权融合
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B, num_bands, 1, 1)
        stacked_feats = torch.stack(fused_feats, dim=1)         # (B, num_bands, hidden_dim, T)
        fused = (weights_expanded * stacked_feats).sum(dim=1)   # (B, hidden_dim, T)

        # 时序建模
        fused = fused.permute(0, 2, 1)                # (B, T, hidden_dim)
        out, _ = self.gru1(fused)
        out = self.norm1(out)
        out, _ = self.gru2(out)
        out = self.norm2(out)
        out = self.dropout(out)

        if return_weights:
            return out, weights
        return out


class WaveletOnlyModel(nn.Module):
    """
    纯小波模型（改进版）：
    - 频段专用卷积
    - 相位门控融合
    - GRU 时序建模
    - 与 CodeNetMotion 接口完全兼容
    """
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        hidden_dim = 128
        output_dim = 3  # 速度输出维度为3
        dropout = getattr(conf, 'dropout', 0.3)
        
        # 小波分支（输出逐点特征）
        self.wavelet_branch = SimpleWaveletBranch(
            in_channels=input_dim,
            wavelet='db4',
            J=4,
            hidden_dim=hidden_dim,
            use_phase=True,
            dropout=dropout
        )

        # 速度解码器（与 CodeNetMotion 一致）
        self.veldecoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), 
            nn.GELU(), 
            nn.Linear(128, output_dim)
        )
        
        # 协方差解码器（与 CodeNetMotion 一致）
        self.velcov_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), 
            nn.GELU(), 
            nn.Linear(128, output_dim)
        )

    def encoder(self, x):
        """
        编码器：提取特征
        输入: x (B, T, 6) - acc和gyro的拼接
        输出: (B, T, hidden_dim)
        """
        return self.wavelet_branch(x)

    def cov_decoder(self, x):
        """
        协方差解码器
        输入: x (B, T, hidden_dim)
        输出: cov (B, T, 3)
        """
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def decoder(self, x):
        """
        速度解码器
        输入: x (B, T, hidden_dim)
        输出: vel (B, T, 3)
        """
        vel = self.veldecoder(x)
        return vel

    def get_label(self, gt_label):
        """
        对齐标签长度与模型输出长度
        模型输出比输入少一个时间步（因为GRU或数据处理）
        """
        # 去掉最后一个时间步，与模型输出对齐
        return gt_label[:, :-1, :]

    def forward(self, data, rot=None):
        """
        前向传播
        输入: 
            data: dict 包含 "acc" 和 "gyro"
            rot: 可选的旋转输入（本模型不使用）
        输出: dict 包含 "cov" 和 "net_vel"
        """
        # 拼接 acc 和 gyro
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)  # (B, T, 6)
        
        # 编码
        feature = self.encoder(feature)  # (B, T, hidden_dim)
        
        # 解码速度
        net_vel = self.decoder(feature)  # (B, T, 3)

        # 协方差传播
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        return {"cov": cov, "net_vel": net_vel}


class WaveletOnlyModelwithRot(WaveletOnlyModel):
    """
    带旋转编码的小波模型：
    - 双分支小波编码器（IMU特征 + 旋转特征）
    - 时间步级别融合（在GRU之前拼接）
    - 与 CodeNetMotionwithRot 接口完全兼容
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        rot_dim = 3  # 旋转输入维度
        hidden_dim = 128
        output_dim = 3
        dropout = getattr(conf, 'dropout', 0.3)
        
        # IMU特征编码器（已继承自父类，这里重新定义为feature_encoder）
        self.feature_encoder = SimpleWaveletBranch(
            in_channels=input_dim,
            wavelet='db4',
            J=4,
            hidden_dim=hidden_dim,
            use_phase=True,
            dropout=dropout
        )
        
        # 旋转特征编码器
        self.ori_encoder = SimpleWaveletBranch(
            in_channels=rot_dim,
            wavelet='db4',
            J=4,
            hidden_dim=hidden_dim,
            use_phase=True,
            dropout=dropout
        )
        
        # 融合层：将两个128维特征拼接后降维到64维
        self.fusion_fcn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        
        # 时序建模GRU（与CodeNetMotionwithRot一致）
        self.gru1 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, 
                           num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=hidden_dim*2, hidden_size=hidden_dim, 
                           num_layers=1, batch_first=True, bidirectional=True)
        
        # 解码器（输入维度256，因为双向GRU输出hidden_dim*2）
        self.veldecoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        self.velcov_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def encoder(self, feature, rot):
        """
        编码器：分别编码IMU特征和旋转特征，然后融合
        输入: 
            feature: (B, T, 6) - acc和gyro的拼接
            rot: (B, T, 3) - 旋转特征（旋转矢量）
        输出: (B, T, hidden_dim*2) - 融合后的特征
        """
        # 分别编码
        x1 = self.feature_encoder(feature)  # (B, T, hidden_dim)
        x2 = self.ori_encoder(rot)          # (B, T, hidden_dim)
        
        # 时间步级别拼接
        x = torch.cat([x1, x2], dim=-1)     # (B, T, hidden_dim*2)
        
        # 融合降维
        x = self.fusion_fcn(x)              # (B, T, hidden_dim)
        x = self.fusion_bn(x.transpose(-1, -2)).transpose(-1, -2)  # BatchNorm
        x = F.gelu(x)
        
        # 时序建模
        x, _ = self.gru1(x)                 # (B, T, hidden_dim*2)
        x, _ = self.gru2(x)                 # (B, T, hidden_dim*2)
        return x

    def forward(self, data, rot=None):
        """
        前向传播
        输入: 
            data: dict 包含 "acc" 和 "gyro"
            rot: (B, T, 3) 旋转输入（旋转矢量）
        输出: dict 包含 "cov" 和 "net_vel"
        """
        assert rot is not None, "WaveletOnlyModelwithRot requires rot input"
        
        # 拼接 acc 和 gyro
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)  # (B, T, 6)
        
        # 编码（包含旋转）
        feature = self.encoder(feature, rot)  # (B, T, hidden_dim*2)
        
        # 解码速度
        net_vel = self.decoder(feature)  # (B, T, 3)

        # 协方差传播
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        return {"cov": cov, "net_vel": net_vel}
