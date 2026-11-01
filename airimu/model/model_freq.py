import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_wavelets import DWT1DForward
from model.net import ModelBase

class WaveletBranchSeq(nn.Module):
    def __init__(self, in_channels=6, wavelet='db4', J=4, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.J = J
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')
        self.level_convs = nn.ModuleList()
        for _ in range(J):
            self.level_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU()
            ))
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
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
            feat = self.level_convs[j](yh[j])
            feat_up = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            level_feats.append(feat_up)

        combined = torch.stack(level_feats, dim=0).sum(dim=0)  # (B, hidden_dim, T)
        combined = self.dropout(combined)
        combined = self.fusion(combined)                      # (B, hidden_dim, T)
        return combined.permute(0, 2, 1)                      # (B, T, hidden_dim)

class DeepBranchSeq(nn.Module):
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
        # x: (B, T, C)
        B, T, C = x.shape
        x = x.permute(0, 2, 1)                # (B, C, T)
        x_down = self.down_conv(x)            # (B, C_down, T_down)
        x_down = x_down.permute(0, 2, 1)      # (B, T_down, C_down)
        lstm_out, _ = self.lstm(x_down)       # (B, T_down, hidden_size)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, hidden_size, T_down)
        # 使用 interpolate 确保输出尺寸与输入一致
        up_out = F.interpolate(lstm_out, size=T, mode='linear', align_corners=False)  # (B, hidden_size, T)
        up_out = up_out.permute(0, 2, 1)      # (B, T, hidden_size)
        up_out = self.fc_dropout(up_out)
        out = self.fc(up_out)                # (B, T, out_dim)
        return out

class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat1, feat2):
        # feat1, feat2: (B, T, d_model)
        combined = torch.cat([feat1, feat2], dim=-1)          # (B, T, 2*d_model)
        gate = self.sigmoid(self.gate_linear(combined))       # (B, T, d_model)
        fused = gate * feat1 + (1 - gate) * feat2
        return fused

class FreqHybridModel(ModelBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf

        input_dim = 6
        wavelet_out = 128
        deep_hidden = 128
        d_model = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.3)

        # padding 长度，用于去除前面填充的 gravity padding
        self.pad_len = 9

        self.wavelet_branch = WaveletBranchSeq(in_channels=input_dim, wavelet='db4',
                                               J=4, hidden_dim=wavelet_out,
                                               dropout=dropout)
        self.deep_branch = DeepBranchSeq(in_channels=input_dim, hidden_size=deep_hidden,
                                         num_layers=2, out_dim=deep_hidden,
                                         down_ratios=(2, 2), dropout=dropout)

        # 投影层
        self.proj_w = nn.Linear(wavelet_out, d_model)
        self.proj_d = nn.Linear(deep_hidden, d_model)
        self.proj_drop = nn.Dropout(dropout)

        # 门控融合
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        # 输出头
        self.correction_head = nn.Linear(d_model, output_dim)
        self.logvar_head = nn.Linear(d_model, output_dim)

        nn.init.constant_(self.logvar_head.weight, 0.01)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def inference(self, data):
        '''
        Pure inference, generate the network output.
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
            cov_state['acc_cov'] = torch.exp(logvar[:, self.pad_len:, :3] - 5.0)
            cov_state['gyro_cov'] = torch.exp(logvar[:, self.pad_len:, 3:] - 5.0)

        return {**cov_state, 'correction_acc': correction[..., :3], 'correction_gyro': correction[..., 3:]}

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        # 去除输入数据前面的 padding
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

