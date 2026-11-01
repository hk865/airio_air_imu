"""
简化版频率-相位特征融合模型
用于验证频域/相位特征的有效性

设计原则：
1. 保持原始 CodeNetMotionwithRot 结构完全不变
2. 只添加 FrequencyPhaseEncoder 分支
3. 使用简单门控融合，便于观察新特征贡献
4. 参数量接近原始模型，便于公平对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward
import pypose as pp
class ModelBase(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        if "gravity" in conf.keys():
            self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True, gravity=0.0)
            print("conf.ngravity", conf.ngravity, self.integrator.gravity)
        else:
            self.integrator = pp.module.IMUPreintegrator(prop_cov=conf.propcov, reset=True)
        print("network constructed: ", self.conf.network, "gtrot: ", self.conf.gtrot)

    def _select(self, data, start, end):

        select = {}
        for k in data.keys():
            if data[k] is None:
                select[k] = None
            else:
                select[k] = data[k][:, start:end]
        return select

    def integrate(self, init_state, data, cov_state):
        B, F = data["corrected_acc"].shape[:2]
        inte_pos, inte_vel, inte_rot, inte_cov = [], [], [], []
        gt_rot = None
        if self.conf.gtrot:
            gt_rot = data['rot']
        if "posonly" in self.conf.keys():
            data['corrected_gyro'] = data['gyro']

        if self.conf.sampling:
            inte_state = None
            for iter in range(0, F, self.conf.sampling):
                if (F - iter) < self.conf.sampling: continue
                start, end = iter, iter + self.conf.sampling
                selected_data = self._select(data, start, end)
                selected_cov_state = self._select(cov_state, start, end)

                # take the init sate from last frame as the init state of the next frame
                if inte_state is not None:
                    init_state = {
                        "pos": inte_state["pos"][:, -1:, :],
                        "vel": inte_state["vel"][:, -1:, :],
                        "rot": inte_state["rot"][:, -1:, :],
                    }
                    if self.conf.propcov:
                        init_state["Rij"] = inte_state["Rij"]
                        init_state["cov"] = inte_state["cov"]

                if self.conf.gtrot:
                    gt_rot = selected_data['rot']

                ## starting point and ending point
                inte_state = self.integrator(init_state=init_state, dt=selected_data['dt'],
                                             gyro=selected_data['corrected_gyro'],
                                             acc=selected_data['corrected_acc'], rot=gt_rot,
                                             acc_cov=selected_cov_state['acc_cov'],
                                             gyro_cov=selected_cov_state['gyro_cov'])

                inte_pos.append(inte_state['pos'])
                inte_rot.append(inte_state['rot'])
                inte_vel.append(inte_state['vel'])
                inte_cov.append(inte_state['cov'])

            out_state = {
                'pos': torch.cat(inte_pos, dim=1),
                'vel': torch.cat(inte_vel, dim=1),
                'rot': torch.cat(inte_rot, dim=1),
            }
            if self.conf.propcov:
                out_state['cov'] = torch.stack(inte_cov, dim=1)
        else:

            out_state = self.integrator(init_state=init_state, dt=data['dt'], gyro=data['corrected_gyro'],
                                        acc=data['corrected_acc'], rot=gt_rot, acc_cov=cov_state['acc_cov'],
                                        gyro_cov=cov_state['gyro_cov'])

        return {**out_state, **cov_state}

    def inference(self, data):
        '''
        Pure inference, generate the network output.
        '''
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature)
        correction = self.decoder(feature)

        # Correction update
        data['corrected_acc'] = correction[..., :3] + data["acc"]
        data['corrected_gyro'] = correction[..., 3:] + data["gyro"]

        # covariance propagation
        cov_state = {'acc_cov': None, 'gyro_cov': None, }
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = cov[..., :3];
            cov_state['gyro_cov'] = cov[..., 3:]

        return {**cov_state, 'correction_acc': correction[..., :3], 'correction_gyro': correction[..., 3:]}

    ## For reference
    def forward(self, data, init_state):
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature)
        correction = self.decoder(feature)

        # Correction update
        data['corrected_acc'] = correction[..., :3] + data["acc"]
        data['corrected_gyro'] = correction[..., 3:] + data["gyro"]

        # covariance propagation
        cov_state = {'acc_cov': None, 'gyro_cov': None, }
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
            cov_state['acc_cov'] = cov[..., :3];
            cov_state['gyro_cov'] = cov[..., 3:]

        out_state = self.integrate(init_state=init_state, data=data, cov_state=cov_state)
        return {**out_state, 'correction_acc': correction[..., :3], 'correction_gyro': correction[..., 3:]}

class HilbertPhase(nn.Module):
    """Hilbert 变换，输出 sin/cos 相位对"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)
        h = torch.zeros(T // 2 + 1, device=x.device)
        h[0] = 1.0
        if T % 2 == 0:
            h[1:T // 2] = 2.0
            h[T // 2] = 1.0
        else:
            h[1:(T + 1) // 2] = 2.0
        X_h = X * h.unsqueeze(0).unsqueeze(0)
        x_h = torch.fft.irfft(X_h, n=T, dim=-1)
        analytic = x + 1j * x_h
        mag = torch.abs(analytic)
        phase = torch.angle(analytic)
        return mag, torch.sin(phase), torch.cos(phase)


class WaveletBranchSeq(nn.Module):
    """
    在model_freq基础上添加相位特征
    保持原有结构，只增加相位编码分支
    """

    def __init__(self, in_channels=6, wavelet='db4', J=4, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.J = J
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

        combined = torch.stack(level_feats, dim=0).sum(dim=0)
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
        wavelet_out = 128
        deep_hidden = 128
        d_model = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.3)

        # 保持model_freq的padding设置
        self.pad_len = 9

        # 改进的小波分支（添加相位）
        self.wavelet_branch = WaveletBranchSeq(
            in_channels=input_dim, wavelet='db4',
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
            cov_state['acc_cov'] = torch.exp(logvar[:, self.pad_len:, :3] - 5.0)
            cov_state['gyro_cov'] = torch.exp(logvar[:, self.pad_len:, 3:] - 5.0)

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


class ImprovedFreqHybridModelwithRot(nn.Module):
    """
    类似 CodeNetMotionwithRot 的接口:
    - 输入: data (acc, gyro), rot (旋转的Log表示)
    - 输出: net_vel (体速度), cov (不确定性), pos (位置，可选)
    - 保留积分器逻辑用于计算位置
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        rot_dim = 3
        wavelet_out = 128
        deep_hidden = 128
        d_model = 128
        output_dim = 3
        dropout = getattr(conf, 'dropout', 0.3)

        self.k_list = [7, 7]
        self.p_list = [3, 3]
        self.s_list = [3, 3]
        self.pad_len = 9

        self.rot_mask_ratio = getattr(conf, 'rot_mask_ratio', 0.0)
        self.rot_mask_value = getattr(conf, 'rot_mask_value', 0.0)

        self.wavelet_branch = WaveletBranchSeq(
            in_channels=input_dim, wavelet='db4',
            J=4, hidden_dim=wavelet_out, dropout=dropout
        )

        self.deep_branch = DeepBranchSeq(
            in_channels=input_dim, hidden_size=deep_hidden,
            num_layers=2, out_dim=deep_hidden,
            down_ratios=(2, 2), dropout=dropout
        )

        self.rot_encoder = DeepBranchSeq(
            in_channels=rot_dim, hidden_size=deep_hidden,
            num_layers=2, out_dim=deep_hidden,
            down_ratios=(2, 2), dropout=dropout
        )

        self.proj_w = nn.Linear(wavelet_out, d_model)
        self.proj_d = nn.Linear(deep_hidden, d_model)
        self.proj_rot = nn.Linear(deep_hidden, d_model)
        self.proj_drop = nn.Dropout(dropout)

        self.gate_linear = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(dropout)

        self.fusion_fcn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        self.fusion_bn = nn.BatchNorm1d(d_model)

        self.gru1 = nn.GRU(input_size=d_model, hidden_size=64,
                           num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64,
                           num_layers=1, batch_first=True, bidirectional=True)

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

        if hasattr(conf, 'gravity') and conf.gravity is not None:
            self.integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True, gravity=0.0)
        else:
            self.integrator = pp.module.IMUPreintegrator(prop_cov=False, reset=True)

        self.dt = getattr(conf, 'dt', 0.005)

    def apply_rot_mask(self, rot, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.rot_mask_ratio

        if mask_ratio <= 0 or not self.training:
            return rot, None

        B, T, C = rot.shape
        mask = torch.rand(B, T, device=rot.device) > mask_ratio
        mask = mask.unsqueeze(-1).expand(-1, -1, C)
        masked_rot = torch.where(mask, rot, torch.tensor(self.rot_mask_value, device=rot.device, dtype=rot.dtype))
        return masked_rot, mask

    def encoder(self, feature, rot):
        w_feat = self.wavelet_branch(feature)
        d_feat = self.deep_branch(feature)
        w_feat = self.proj_w(w_feat)
        d_feat = self.proj_d(d_feat)

        combined_imu = torch.cat([w_feat, d_feat], dim=-1)
        gate = self.sigmoid(self.gate_linear(combined_imu))
        imu_fused = gate * w_feat + (1 - gate) * d_feat
        imu_fused = self.proj_drop(imu_fused)

        if self.training and self.rot_mask_ratio > 0:
            rot, _ = self.apply_rot_mask(rot)

        rot_feat = self.rot_encoder(rot)
        rot_feat = self.proj_rot(rot_feat)

        combined = torch.cat([imu_fused, rot_feat], dim=-1)

        x = self.fusion_fcn(combined)
        x = self.fusion_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.gelu(x)

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def cov_decoder(self, x):
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def decoder(self, x):
        vel = self.veldecoder(x)
        return vel

    def get_label(self, gt_label):
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 - self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0] * self.s_list[1], :]
        L_out = (gt_label.shape[1] - 1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label, gt_label[:, -1:, :].repeat(1, diff, 1)), dim=1)
        return select_label

    def integrate_velocity(self, vel_body, rot, dt=None):
        """
        将体速度积分到位置
        vel_body: (B, T, 3) 体坐标系下的速度
        rot: (B, T, 3) 旋转的Log表示 (SO3的对数映射)
        返回: (B, T, 3) 位置
        """
        if dt is None:
            dt = self.dt

        B, T, _ = vel_body.shape
        device = vel_body.device
        dtype = vel_body.dtype

        rot_so3 = pp.SO3(rot).Exp()

        vel_world = torch.einsum('btij,btj->bti', rot_so3.tensor(), vel_body)

        pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
        for t in range(1, T):
            pos[:, t, :] = pos[:, t-1, :] + vel_world[:, t-1, :] * dt

        return pos

    def forward(self, data, rot=None, init_state=None):
        assert rot is not None, "ImprovedFreqHybridModelwithRot requires rot input"

        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature, rot)
        net_vel = self.decoder(feature)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        pos = None
        if init_state is not None and getattr(self.conf, 'posloss', False):
            pos = self.integrate_velocity(net_vel, rot)

        return {
            "cov": cov,
            "net_vel": net_vel,
            "pos": pos
        }
