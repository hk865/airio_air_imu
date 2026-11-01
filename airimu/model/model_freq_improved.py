import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward
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
        # 3. 实信号只需要非负频率。相比完整 fft/ifft，这个实现少存储
        #    约一半复数频谱，同时保持标准离散 Hilbert 变换的定义。
        X = torch.fft.rfft(x, dim=-1)

        # 4. DC 和偶数长度下的 Nyquist 分量为 0，其余正频率乘 -i。
        H = torch.zeros_like(X)
        positive_end = -1 if T % 2 == 0 else None
        H[..., 1:positive_end] = -1j

        # 5. 频域相乘 + 逆变换
        x_h = torch.fft.irfft(X * H, n=T, dim=-1)

        # 6. 解析信号幅度。保持真实零幅值，供相位置信度门控使用；
        #    eps 应在需要做除法的位置加入，而不是人为抬高幅值。
        mag = torch.sqrt(x ** 2 + x_h ** 2)

        phase = torch.atan2(x_h, x)

        # 相位三角函数
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        return mag, sin_phase, cos_phase


class WaveletBranchSeq(nn.Module):
    """
    在model_freq基础上添加相位特征
    保持原有结构，只增加相位编码分支
    """
    def __init__(self, in_channels=6, wavelet='db4', J=4, hidden_dim=128,
                 dropout=0.2, level_aggregation='sum'):
        super().__init__()
        self.J = J
        if level_aggregation not in ('sum', 'mean'):
            raise ValueError("level_aggregation must be 'sum' or 'mean'")
        self.level_aggregation = level_aggregation
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')
        self.hilbert = HilbertPhase()
        
        # 原有的幅度编码器（保持不变）
        self.level_convs = nn.ModuleList()
        for _ in range(J):
            self.level_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            ))
        
        # 新增的相位编码器（轻量级）
        self.phase_convs = nn.ModuleList()
        for _ in range(J):
            self.phase_convs.append(nn.Sequential(
                nn.Conv1d(in_channels * 2, 64, kernel_size=3, padding=1),  # sin+cos
                nn.Tanh(),  # 保留负值
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            ))
        
        # 幅度+相位融合
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        _, yh = self.dwt(x)
        B, C, T = x.shape

        level_feats = []
        for j in range(self.J):
            # 幅度特征
            mag_feat = self.level_convs[j](yh[j])
            
            # 相位特征
            _, sin_p, cos_p = self.hilbert(yh[j])
            phase_input = torch.cat([sin_p, cos_p], dim=1)
            phase_feat = self.phase_convs[j](phase_input)
            
            # 上采样到原始长度
            mag_up = F.interpolate(mag_feat, size=T, mode='linear', align_corners=False)
            phase_up = F.interpolate(phase_feat, size=T, mode='linear', align_corners=False)
            
            # 拼接幅度和相位
            combined = torch.cat([mag_up, phase_up], dim=1)
            fused = self.fusion(combined)
            level_feats.append(fused)

        combined = torch.stack(level_feats, dim=0)
        combined = combined.sum(dim=0) if self.level_aggregation == 'sum' else combined.mean(dim=0)
        combined = self.dropout(combined)
        return combined.permute(0, 2, 1)  # (B, T, hidden_dim)


class DeepBranchSeq(nn.Module):
    """
    保持model_freq的原始结构不变
    """
    def __init__(self, in_channels=6, hidden_size=128, num_layers=2, out_dim=128,
                 down_ratios=(2, 2), dropout=0.2):
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
                            dropout=dropout if num_layers > 1 else 0)

        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x_down = self.down_conv(x)
        x_down = x_down.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_down)
        lstm_out = lstm_out.permute(0, 2, 1)
        up_out = F.interpolate(lstm_out, size=T, mode='linear', align_corners=False)
        up_out = up_out.permute(0, 2, 1)
        up_out = self.fc_dropout(up_out)
        out = self.fc(up_out)
        return out


class ImprovedFreqHybridModel(ModelBase):
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
        wavelet_out = int(conf.get('wavelet_hidden', 128))
        deep_hidden = int(conf.get('deep_hidden', 128))
        d_model = int(conf.get('fusion_dim', 128))
        output_dim = 6
        dropout = conf.get('dropout', 0.3)
        wavelet = conf.get('wavelet', 'db4')
        wavelet_levels = int(conf.get('wavelet_levels', 4))
        level_aggregation = conf.get('level_aggregation', 'sum')
        lstm_layers = int(conf.get('lstm_layers', 2))
        down_ratios = tuple(conf.get('down_ratios', (2, 2)))

        # 保持model_freq的padding设置
        self.pad_len = 9
        self.cov_logvar_offset = conf.get('cov_logvar_offset', -5.0)
        self.cov_log_min = conf.get('cov_log_min', -18.0)
        self.cov_log_max = conf.get('cov_log_max', 10.0)

        # 改进的小波分支（添加相位）
        self.wavelet_branch = WaveletBranchSeq(
            in_channels=input_dim, wavelet=wavelet,
            J=wavelet_levels, hidden_dim=wavelet_out, dropout=dropout,
            level_aggregation=level_aggregation,
        )
        
        # 保持model_freq的深度分支不变
        self.deep_branch = DeepBranchSeq(
            in_channels=input_dim, hidden_size=deep_hidden,
            num_layers=lstm_layers, out_dim=deep_hidden,
            down_ratios=down_ratios, dropout=dropout
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
        '''
        保持model_freq的inference逻辑不变
        '''
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)

        w_feat = self.wavelet_branch(feature)
        d_feat = self.deep_branch(feature)

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

        # covariance propagation
        cov_state = {'acc_cov': None, 'gyro_cov': None}
        if self.conf.propcov:
            log_variance = (logvar[:, self.pad_len:] + self.cov_logvar_offset).clamp(
                min=self.cov_log_min, max=self.cov_log_max
            )
            cov_state['acc_cov'] = torch.exp(log_variance[..., :3])
            cov_state['gyro_cov'] = torch.exp(log_variance[..., 3:])

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
