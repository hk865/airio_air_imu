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


class OverlapAggregator(nn.Module):
    """
    模仿 CodeNet 的 _update 机制：
    - 输入特征序列 (B, T, D)
    - 每隔 interval 步取一个块，预测该块对应的校正量
    - 通过滑动窗口累加生成输出序列 (B, T_out, output_dim)
    """
    def __init__(self, hidden_dim, output_dim, interval=9):
        super().__init__()
        self.interval = interval
        self.inter_head = interval // 2
        self.inter_tail = interval - self.inter_head
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, feat):
        """
        feat: (B, T, D)
        返回: (B, T_out, output_dim), 其中 T_out = T - interval
        """
        B, T, D = feat.shape
        # 实际可用的帧数（去除最前面 interval 长度的填充）
        T_out = T - self.interval
        if T_out <= 0:
            # 安全处理，一般不会发生
            return torch.zeros(B, 0, self.linear.out_features, device=feat.device)

        # 计算特征块的数量（每个块对应输出中的一个时间窗口）
        num_blocks = (T_out + self.interval - 1) // self.interval
        # 从特征中每隔 interval 取一个块（相当于下采样）
        feat_blocks = feat[:, ::self.interval, :]   # (B, num_blocks, D)
        # 确保与 num_blocks 对齐（可能因切片略有差异，但通常一致）
        if feat_blocks.size(1) < num_blocks:
            # 补齐（极少数情况，由于T不是interval整数倍）
            pad = num_blocks - feat_blocks.size(1)
            feat_blocks = F.pad(feat_blocks, (0,0,0,pad))

        # 预测每个块的校正量
        block_corr = self.linear(feat_blocks)       # (B, num_blocks, output_dim)

        # 初始化输出
        out = torch.zeros(B, T_out, block_corr.shape[-1], device=feat.device)

        # 滑动窗口累加
        for i in range(num_blocks):
            s = max(0, i * self.interval - self.inter_head)
            e = min(T_out, i * self.interval + self.inter_tail)
            if s >= e:
                continue
            out[:, s:e, :] += block_corr[:, i:i+1, :]
        return out


class WaveletOnlyModel(ModelBase):
    """
    纯小波模型（改进版）：
    - 频段专用卷积
    - 相位门控融合
    - GRU 时序建模
    - 重叠累加输出头（模仿 CodeNet 的平滑机制）
    - 与基类接口完全兼容
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        hidden_dim = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.3)
        interval = 9   # 与 CodeNet 保持一致

        self.interval = interval
        # 不再需要 pad_len，因为 OverlapAggregator 会自动处理边界
        self.pad_len = 0

        # 小波分支（输出逐点特征）
        self.wavelet_branch = SimpleWaveletBranch(
            in_channels=input_dim,
            wavelet='db4',
            J=4,
            hidden_dim=hidden_dim,
            use_phase=True,
            dropout=dropout
        )

        # 重叠累加输出头（输出校正量）
        self.aggregator = OverlapAggregator(hidden_dim, output_dim, interval=interval)

        # 可选：协方差输出头（同样使用重叠累加）
        self.cov_aggregator = OverlapAggregator(hidden_dim, output_dim, interval=interval)

        # 初始化输出头的偏置（使初始校正量接近0）
        nn.init.zeros_(self.aggregator.linear.weight)
        nn.init.zeros_(self.aggregator.linear.bias)
        nn.init.zeros_(self.cov_aggregator.linear.weight)
        nn.init.constant_(self.cov_aggregator.linear.bias, -5.0)  # 协方差初始化为较小值

    def inference(self, data, return_freq_weights=False):
        """
        返回与 CodeNet 完全一致的字典结构
        """
        # 输入特征：原始长度（包含额外的 interval 填充）
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)   # (B, T, 6)
        # 小波分支提取特征
        if return_freq_weights:
            features, freq_weights = self.wavelet_branch(feature, return_weights=True)
        else:
            features = self.wavelet_branch(feature)
            freq_weights = None

        # 通过重叠累加生成校正量（长度自动变为 T - interval）
        correction = self.aggregator(features)                     # (B, T_out, 6)
        correction_acc = correction[..., :3]
        correction_gyro = correction[..., 3:]

        # 协方差传播（可选）
        cov_state = {'acc_cov': None, 'gyro_cov': None}
        if self.conf.propcov:
            logvar = self.cov_aggregator(features)                 # (B, T_out, 6)
            acc_cov = torch.exp(logvar[..., :3] - 5.0).double()
            gyro_cov = torch.exp(logvar[..., 3:] - 5.0).double()
            cov_state['acc_cov'] = acc_cov
            cov_state['gyro_cov'] = gyro_cov

        result = {
            'cov_state': cov_state,
            'correction_acc': correction_acc,
            'correction_gyro': correction_gyro
        }
        if freq_weights is not None:
            result['freq_weights'] = freq_weights
        return result

    def forward(self, data, init_state):
        """
        保持与基类完全相同的接口
        """
        inference_state = self.inference(data)

        # 注意：此时 correction_acc 的长度已经自动减去 interval，与原始数据对齐
        # 不需要再手动切掉 pad_len，因为 inference 返回的就是对齐后的长度
        data['corrected_acc'] = data['acc'][:, self.interval:, :] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'][:, self.interval:, :] + inference_state['correction_gyro']

        # 调用积分模块（基类中的 integrate 方法）
        out_state = self.integrate(
            init_state=init_state,
            data=data,
            cov_state=inference_state['cov_state']
        )

        return {
            **out_state,
            'correction_acc': inference_state['correction_acc'],
            'correction_gyro': inference_state['correction_gyro'],
            'corrected_acc': data['corrected_acc'],
            'corrected_gyro': data['corrected_gyro']
        }