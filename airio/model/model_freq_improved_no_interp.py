import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward
import pypose as pp


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


class WaveletBranchSeqNoInterp(nn.Module):
    """
    小波分支 - 无插值版本，输出下采样后的长度
    """

    def __init__(self, in_channels=6, wavelet='db4', J=4, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.J = J
        self.dwt = DWT1DForward(J=J, wave=wavelet, mode='symmetric')
        self.hilbert = HilbertPhase()
        self.total_stride = 2**J  # 小波分解的总下采样倍数

        # 幅度编码器（LeakyReLU）
        self.level_convs = nn.ModuleList()
        for _ in range(J):
            self.level_convs.append(nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1)
            ))

        # 相位编码器（轻量级）
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
            nn.LeakyReLU(0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        _, yh = self.dwt(x)
        
        # 使用最低频的小波系数长度作为输出长度
        T_out = yh[-1].shape[-1]

        level_feats = []
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

    def __init__(self, in_channels=6, hidden_size=128, num_layers=2, out_dim=128,
                 down_ratios=(3, 3), dropout=0.5):
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
            layers.append(nn.LeakyReLU(0.1, inplace=True))
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
        # 移除上采样操作，直接输出下采样后的长度
        lstm_out = self.fc_dropout(lstm_out)
        out = self.fc(lstm_out)
        return out


class ImprovedFreqHybridModelNoInterp(nn.Module):
    """
    ImprovedFreqHybridModel 无插值版本
    - 输出长度 = 输入长度 // 16 (J=4小波分解) 或 // 9 (使用down_ratios=(3,3))
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        wavelet_out = 128
        deep_hidden = 128
        d_model = 128
        output_dim = 6
        dropout = conf.get('dropout', 0.5)

        # 无插值版本，使用 (3,3) 下采样与CodeNet一致
        self.wavelet_branch = WaveletBranchSeqNoInterp(
            in_channels=input_dim, wavelet='db4',
            J=4, hidden_dim=wavelet_out, dropout=dropout
        )

        # 深度分支 - 无插值，使用 (3,3) 下采样
        self.deep_branch = DeepBranchSeqNoInterp(
            in_channels=input_dim, hidden_size=deep_hidden,
            num_layers=2, out_dim=deep_hidden,
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

        # 输出头
        self.correction_head = nn.Linear(d_model, output_dim)
        self.logvar_head = nn.Linear(d_model, output_dim)

        nn.init.constant_(self.logvar_head.weight, 0.01)
        nn.init.constant_(self.logvar_head.bias, -5.0)

    def inference(self, data):
        '''
        无插值版本的inference
        '''
        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)

        w_feat = self.wavelet_branch(feature)
        d_feat = self.deep_branch(feature)

        # 确保两个分支输出长度一致
        min_len = min(w_feat.shape[1], d_feat.shape[1])
        w_feat = w_feat[:, :min_len, :]
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

        # covariance propagation
        cov_state = {'acc_cov': None, 'gyro_cov': None}
        if self.conf.propcov:
            cov_state['acc_cov'] = torch.exp(logvar[..., :3] - 5.0)
            cov_state['gyro_cov'] = torch.exp(logvar[..., 3:] - 5.0)

        return {**cov_state, 'correction_acc': correction[..., :3], 'correction_gyro': correction[..., 3:]}

    def forward(self, data, init_state):
        inference_state = self.inference(data)

        # 注意：这里需要处理长度对齐问题
        # 由于输出长度缩短，需要对齐acc和gyro
        min_len = inference_state['correction_acc'].shape[1]
        data['corrected_acc'] = data['acc'][:, :min_len, :] + inference_state['correction_acc']
        data['corrected_gyro'] = data['gyro'][:, :min_len, :] + inference_state['correction_gyro']

        # 截断其他数据以匹配长度
        data['dt'] = data['dt'][:, :min_len, :]
        if 'rot' in data:
            data['rot'] = data['rot'][:, :min_len, :]

        out_state = self.integrate(init_state=init_state, data=data, cov_state=inference_state)

        return {
            **out_state,
            'correction_acc': inference_state['correction_acc'],
            'correction_gyro': inference_state['correction_gyro'],
            'corrected_acc': data['corrected_acc'],
            'corrected_gyro': data['corrected_gyro']
        }

    def integrate(self, init_state, data, cov_state):
        """简化版积分器"""
        B, F = data["corrected_acc"].shape[:2]
        
        # 初始化状态
        if 'pos' not in init_state:
            init_state = {
                'pos': torch.zeros(B, 1, 3, device=data['corrected_acc'].device),
                'vel': torch.zeros(B, 1, 3, device=data['corrected_acc'].device),
                'rot': torch.zeros(B, 1, 3, device=data['corrected_acc'].device),
            }
        
        inte_pos, inte_vel, inte_rot = [init_state['pos']], [init_state['vel']], [init_state['rot']]
        
        # 简单的欧拉积分
        for t in range(F):
            dt = data['dt'][:, t:t+1, :]
            acc = data['corrected_acc'][:, t:t+1, :]
            gyro = data['corrected_gyro'][:, t:t+1, :]
            
            # 速度积分
            new_vel = inte_vel[-1] + acc * dt
            inte_vel.append(new_vel)
            
            # 位置积分
            new_pos = inte_pos[-1] + inte_vel[-1] * dt
            inte_pos.append(new_pos)
            
            # 旋转积分（简化）
            new_rot = inte_rot[-1] + gyro * dt
            inte_rot.append(new_rot)
        
        out_state = {
            'pos': torch.cat(inte_pos[1:], dim=1),
            'vel': torch.cat(inte_vel[1:], dim=1),
            'rot': torch.cat(inte_rot[1:], dim=1),
        }
        
        if self.conf.propcov:
            out_state['cov'] = torch.stack([cov_state['acc_cov'], cov_state['gyro_cov']], dim=-1)
            
        return {**out_state, **cov_state}


class ImprovedFreqHybridModelwithRotNoInterp(nn.Module):
    """
    ImprovedFreqHybridModelwithRot 无插值版本
    - 输入: data (acc, gyro), rot (旋转的Log表示)
    - 输出: net_vel (体速度), cov (不确定性)
    - 输出长度 = 输入长度 // 9
    """

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        input_dim = 6
        rot_dim = 3
        wavelet_out = 64
        deep_hidden = 64
        d_model = 80
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

        # 门控融合
        self.gate_linear = nn.Linear(2 * d_model, d_model)

        # 融合层
        self.fusion_fcn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        self.fusion_bn = nn.BatchNorm1d(d_model)

        # 时序建模 GRU
        self.gru1 = nn.GRU(input_size=d_model, hidden_size=80,
                           num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=160, hidden_size=80,
                           num_layers=1, batch_first=True, bidirectional=True)

        # 解码器
        self.veldecoder = nn.Sequential(
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Linear(80, output_dim)
        )

        self.velcov_decoder = nn.Sequential(
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Linear(80, output_dim)
        )

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

    def encoder(self, feature, rot, mask_ratio=None):
        """编码器 - 无插值版本"""
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

    def integrate_velocity(self, vel_body, rot, dt=None, init_pos=None, window_size=None):
        """
        将体速度积分到位置
        vel_body: (B, T, 3) 体坐标系下的速度
        rot: (B, T, 3) 旋转矢量 (so3 李代数)
        init_pos: (B, 3) 或 (B, num_windows, 3) 初始位置
        window_size: int 或 None，如果提供则使用滑动窗口积分
        返回: (B, T, 3) 位置
        """
        if dt is None:
            dt = self.dt

        B, T, _ = vel_body.shape
        device = vel_body.device
        dtype = vel_body.dtype

        # rot 是 so3 (旋转矢量)，需要用 pp.so3
        # .Exp() 得到 SO3，.matrix() 得到旋转矩阵 (B, T, 3, 3)
        rot_matrix = pp.so3(rot).Exp().matrix()

        vel_world = torch.einsum('btij,btj->bti', rot_matrix, vel_body)

        pos = torch.zeros(B, T, 3, device=device, dtype=dtype)

        # 检查是否使用滑动窗口
        if window_size is not None and init_pos is not None and init_pos.dim() == 3:
            # init_pos: (B, num_windows, 3)
            num_windows = init_pos.shape[1]
            # 每个窗口独立积分
            for w in range(num_windows):
                start_idx = w * window_size
                end_idx = min((w + 1) * window_size, T)
                if start_idx >= T:
                    break

                # 设置窗口初始位置
                pos[:, start_idx, :] = init_pos[:, w, :]

                # 在窗口内积分
                for t in range(start_idx + 1, end_idx):
                    pos[:, t, :] = pos[:, t-1, :] + vel_world[:, t-1, :] * dt
        else:
            # 原始方式：整个序列一个初始位置
            if init_pos is not None:
                if init_pos.dim() == 3:
                    # 如果 init_pos 是 (B, 1, 3)，取第一个
                    init_pos = init_pos[:, 0, :]
                pos[:, 0, :] = init_pos
            else:
                pos[:, 0, :] = 0.0

            for t in range(1, T):
                pos[:, t, :] = pos[:, t-1, :] + vel_world[:, t-1, :] * dt

        return pos

    def forward(self, data, rot=None, init_state=None):
        assert rot is not None, "ImprovedFreqHybridModelwithRotNoInterp requires rot input"

        feature = torch.cat([data["acc"], data["gyro"]], dim=-1)
        feature = self.encoder(feature, rot)
        net_vel = self.decoder(feature)

        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)

        pos = None
        if init_state is not None and getattr(self.conf, 'posloss', False):
            init_pos = init_state.get('pos', None)
            window_size = getattr(self.conf, 'pos_window_size', None)
            pos = self.integrate_velocity(net_vel, rot, init_pos=init_pos, window_size=window_size)

        return {
            "cov": cov,
            "net_vel": net_vel,
            "pos": pos
        }
