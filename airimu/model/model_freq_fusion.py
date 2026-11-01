import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DForward
from model.net import ModelBase


class EarlyFusionFreqModel(ModelBase):
    """
    早期融合模型：频域特征与时域CNN特征先融合，再时序建模
    
    核心改进：
    1. 小波特征和CNN特征在底层融合
    2. 融合后的联合特征进行LSTM时序建模
    3. 避免频域分支缺少时序关联的问题
    """
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        hidden_dim = 128
        d_model = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.3)
        
        self.pad_len = 9

        # ========== 频域编码器 ==========
        self.dwt = DWT1DForward(J=3, wave='db4', mode='symmetric')
        
        # 小波特征编码 (每层独立处理)
        self.wavelet_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU()
            ) for _ in range(3)  # J=3层
        ])
        
        # 频域特征融合: yl + cD1 + cD2 + cD3 = 4层, 每层 hidden_dim//2
        self.freq_fusion = nn.Conv1d(hidden_dim // 2 * 4, hidden_dim, kernel_size=1)

        # ========== 时域编码器 ==========
        # 与深度分支类似的CNN
        layers = []
        ch = input_dim
        for stride in [2, 2]:
            layers.extend([
                nn.Conv1d(ch, ch * 2, kernel_size=5, stride=stride, padding=2),
                nn.BatchNorm1d(ch * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            ch = ch * 2
        self.temporal_cnn = nn.Sequential(*layers)
        
        # 时域特征投影到相同维度
        self.temporal_proj = nn.Conv1d(ch, hidden_dim, kernel_size=1)

        # ========== 早期融合 + 时序建模 ==========
        # 融合层：频域特征 + 时域特征
        self.early_fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 联合时序建模 (关键！)
        self.joint_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 上采样将在forward中动态使用interpolate

        # ========== 输出头 ==========
        self.correction_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

        nn.init.constant_(self.logvar_head.weight, 0.01)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def encode_frequency(self, x):
        """
        频域编码
        x: (B, C, T)
        输出: (B, hidden_dim, T)
        """
        B, C, T = x.shape
        
        # 小波分解
        yl, yh = self.dwt(x)  # yl: (B, C, T/8), yh: [cD1, cD2, cD3]
        
        # 处理各层
        level_feats = []
        
        # yl (低频趋势) - 使用第0个编码器
        feat_yl = self.wavelet_encoders[0](yl)
        feat_yl = F.interpolate(feat_yl, size=T, mode='linear', align_corners=False)
        level_feats.append(feat_yl)
        
        # yh各层 (高频细节) - 使用前3个高频分量
        for j in range(min(3, len(yh))):
            cD = yh[j]
            feat = self.wavelet_encoders[j](cD)
            feat_up = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            level_feats.append(feat_up)
        
        # 融合各层: yl + cD1 + cD2 + cD3 = 4层
        # 但每个输出是 hidden_dim//2 = 64
        # 总共 64 * 4 = 256
        # 需要修改 freq_fusion 的输入通道
        combined = torch.cat(level_feats, dim=1)  # (B, hidden_dim//2 * 4, T)
        freq_feat = self.freq_fusion(combined)     # (B, hidden_dim, T)
        
        return freq_feat

    def encode_temporal(self, x, target_T):
        """
        时域编码
        x: (B, C, T)
        target_T: 目标输出长度
        输出: (B, hidden_dim, target_T)
        """
        B, C, T = x.shape
        
        # CNN下采样
        temp_feat = self.temporal_cnn(x)  # (B, ch*4, T/4)
        
        # 投影
        temp_feat = self.temporal_proj(temp_feat)  # (B, hidden_dim, T/4)
        
        # 上采样到目标长度
        temp_feat = F.interpolate(temp_feat, size=target_T, mode='linear', align_corners=False)
        
        return temp_feat

    def inference(self, data):
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        B, T, C = feature.shape
        
        # 转置为 (B, C, T)
        x = feature.permute(0, 2, 1)
        
        # ========== 双分支编码 ==========
        freq_feat = self.encode_frequency(x)            # (B, hidden_dim, T)
        temp_feat = self.encode_temporal(x, target_T=T)  # (B, hidden_dim, T)
        
        # ========== 早期融合 ==========
        combined = torch.cat([freq_feat, temp_feat], dim=1)  # (B, hidden_dim*2, T)
        fused = self.early_fusion(combined)                   # (B, hidden_dim, T)
        
        # ========== 联合时序建模 (关键！) ==========
        fused = fused.permute(0, 2, 1)           # (B, T, hidden_dim)
        lstm_out, _ = self.joint_lstm(fused)     # (B, T, hidden_dim)
        
        # ========== 输出 ==========
        correction = self.correction_head(lstm_out)  # (B, T, output_dim)
        logvar = self.logvar_head(lstm_out)
        
        # 去除padding
        correction = correction[:, self.pad_len:, :]
        
        cov_state = {'acc_cov': None, 'gyro_cov': None}
        if self.conf.propcov:
            cov_state['acc_cov'] = torch.exp(logvar[:, self.pad_len:, :3] - 5.0)
            cov_state['gyro_cov'] = torch.exp(logvar[:, self.pad_len:, 3:] - 5.0)

        return {
            **cov_state,
            'correction_acc': correction[..., :3],
            'correction_gyro': correction[..., 3:]
        }

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
