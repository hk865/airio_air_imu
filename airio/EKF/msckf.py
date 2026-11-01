"""
MSCKF (Multi-State Constraint Kalman Filter) 实现
参考TLIO的scekf.py设计

关键特点:
1. 维护滑动窗口的历史状态 (si_Rs, si_ps)
2. 状态增广: 每次观测更新时将当前状态克隆到窗口
3. 多状态约束: 观测可以约束窗口中的多个状态
4. 边缘化: 窗口满时移除最老的状态
"""

import torch
import numpy as np
import pypose as pp


def hat(v):
    """向量叉乘矩阵"""
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def mat_exp(v):
    """矩阵指数 (so3 -> SO3)"""
    v = v.flatten()
    angle = np.linalg.norm(v)
    if angle < 1e-6:
        return np.eye(3)
    axis = v / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def mat_log(R):
    """矩阵对数 (SO3 -> so3)"""
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    
    if angle < 1e-6:
        return np.zeros(3)
    
    K = (R - R.T) / (2 * np.sin(angle))
    return angle * np.array([K[2, 1], K[0, 2], K[1, 0]])


class MSCKFState:
    """MSCKF状态类"""
    
    def __init__(self):
        # 当前状态
        self.s_R = None  # 3x3 旋转矩阵
        self.s_v = None  # 3x1 速度
        self.s_p = None  # 3x1 位置
        self.s_bg = None  # 3x1 gyro bias
        self.s_ba = None  # 3x1 acc bias
        self.s_timestamp_us = -1
        
        # 滑动窗口历史状态
        self.N = 0  # 窗口大小
        self.si_Rs = []  # 历史旋转 [N x 3 x 3]
        self.si_ps = []  # 历史位置 [N x 3 x 1]
        self.si_timestamps_us = []  # 历史时间戳 [N]
        
    def initialize(self, t_us, R, v, p, bg, ba):
        """初始化状态"""
        # 统一转换为numpy数组
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x)
        
        self.s_R = to_numpy(R).copy()
        self.s_v = to_numpy(v).copy()
        self.s_p = to_numpy(p).copy()
        self.s_bg = to_numpy(bg).copy()
        self.s_ba = to_numpy(ba).copy()
        self.s_timestamp_us = t_us
        self.N = 0
        self.si_Rs = []
        self.si_ps = []
        self.si_timestamps_us = []
        
    def augment(self, t_us):
        """状态增广: 将当前状态克隆到窗口"""
        # 调试：检查s_p的形状
        # if self.s_p is not None:
        #     print(f"Augmenting state: p shape={self.s_p.shape}, R shape={self.s_R.shape}")
        #     print(f"  p value: {self.s_p.flatten()[:3]}")
        self.si_Rs.append(self.s_R.copy())
        self.si_ps.append(self.s_p.copy())
        self.si_timestamps_us.append(t_us)
        self.N += 1
        
    def marginalize(self, cut_idx):
        """边缘化: 移除窗口中前cut_idx个状态"""
        if cut_idx <= 0:
            return
        self.si_Rs = self.si_Rs[cut_idx:]
        self.si_ps = self.si_ps[cut_idx:]
        self.si_timestamps_us = self.si_timestamps_us[cut_idx:]
        self.N -= cut_idx
        
    def apply_correction(self, dX):
        """应用状态修正"""
        # dX = [dtheta_past, dp_past, dtheta_curr, dv, dp, dbg, dba]
        # dX_past: N x 6, dX_curr: 15
        dX_past = dX[:self.N * 6]
        dX_curr = dX[self.N * 6:]
        
        # 修正历史状态
        if self.N > 0:
            for i in range(self.N):
                dtheta = dX_past[6*i:6*i+3]
                dp = dX_past[6*i+3:6*i+6]
                
                dR = mat_exp(dtheta)
                self.si_Rs[i] = dR @ self.si_Rs[i]
                self.si_ps[i] = self.si_ps[i] + dp.reshape(3, 1)
        
        # 修正当前状态
        dtheta = dX_curr[0:3]
        dv = dX_curr[3:6]
        dp = dX_curr[6:9]
        dbg = dX_curr[9:12]
        dba = dX_curr[12:15]
        
        dR = mat_exp(dtheta)
        self.s_R = dR @ self.s_R
        self.s_v = self.s_v + dv.reshape(3, 1)
        self.s_p = self.s_p + dp.reshape(3, 1)
        self.s_bg = self.s_bg + dbg.reshape(3, 1)
        self.s_ba = self.s_ba + dba.reshape(3, 1)


class MSCKF:
    """MSCKF滤波器"""
    
    def __init__(self, 
                 sigma_na=0.01,      # acc噪声 (m/s^2)
                 sigma_ng=0.001,     # gyro噪声 (rad/s)
                 ita_ba=1e-4,        # acc bias随机游走
                 ita_bg=1e-6,        # gyro bias随机游走
                 init_rot_sigma=0.01,  # 初始旋转标准差 (rad)
                 init_vel_sigma=0.1,   # 初始速度标准差 (m/s)
                 init_pos_sigma=0.1,   # 初始位置标准差 (m)
                 init_bg_sigma=0.01,   # 初始gyro bias标准差 (rad/s)
                 init_ba_sigma=0.1,    # 初始acc bias标准差 (m/s^2)
                 max_window_size=10,   # 最大窗口大小
                 g_norm=9.81007):
        
        self.g = np.array([0, 0, -g_norm]).reshape(3, 1)
        
        # 噪声参数
        self.sigma_na = sigma_na
        self.sigma_ng = sigma_ng
        self.ita_ba = ita_ba
        self.ita_bg = ita_bg
        
        # 初始协方差参数
        self.init_rot_sigma = init_rot_sigma
        self.init_vel_sigma = init_vel_sigma
        self.init_pos_sigma = init_pos_sigma
        self.init_bg_sigma = init_bg_sigma
        self.init_ba_sigma = init_ba_sigma
        
        # 窗口大小
        self.max_window_size = max_window_size
        
        # 状态
        self.state = MSCKFState()
        
        # 协方差矩阵
        self.Sigma = None  # 完整协方差 (6N + 15) x (6N + 15)
        self.Sigma15 = None  # 当前状态协方差 15 x 15
        
        # 噪声矩阵
        self.W = None  # 观测噪声 6 x 6
        self.Q = None  # 过程噪声 6 x 6
        
        self.initialized = False
        
    def initialize(self, t_us, R, v, p, bg, ba):
        """初始化滤波器"""
        self.state.initialize(t_us, R, v, p, bg, ba)
        self._prepare_noise_matrices()
        self._reset_covariance()
        self.initialized = True
        
    def _prepare_noise_matrices(self):
        """准备噪声矩阵"""
        var_a = self.sigma_na ** 2
        var_g = self.sigma_ng ** 2
        var_ba = self.ita_ba ** 2
        var_bg = self.ita_bg ** 2
        
        self.W = np.diag([var_g, var_g, var_g, var_a, var_a, var_a])
        self.Q = np.diag([var_bg, var_bg, var_bg, var_ba, var_ba, var_ba])
        
    def _reset_covariance(self):
        """重置协方差矩阵"""
        N = self.state.N
        dim = 6 * N + 15
        
        var_rot = self.init_rot_sigma ** 2
        var_vel = self.init_vel_sigma ** 2
        var_pos = self.init_pos_sigma ** 2
        var_bg = self.init_bg_sigma ** 2
        var_ba = self.init_ba_sigma ** 2
        
        self.Sigma = np.eye(dim) * 1e-6  # 小值避免奇异
        
        # 历史状态协方差
        for i in range(N):
            self.Sigma[6*i:6*i+3, 6*i:6*i+3] = np.eye(3) * var_rot
            self.Sigma[6*i+3:6*i+6, 6*i+3:6*i+6] = np.eye(3) * var_pos
        
        # 当前状态协方差
        start = 6 * N
        self.Sigma[start:start+3, start:start+3] = np.eye(3) * var_rot
        self.Sigma[start+3:start+6, start+3:start+6] = np.eye(3) * var_vel
        self.Sigma[start+6:start+9, start+6:start+9] = np.eye(3) * var_pos
        self.Sigma[start+9:start+12, start+9:start+12] = np.eye(3) * var_bg
        self.Sigma[start+12:start+15, start+12:start+15] = np.eye(3) * var_ba
        
        self.Sigma15 = self.Sigma[-15:, -15:]
        
    def propagate(self, acc, gyr, t_us, do_augment=False):
        """
        状态传播
        acc: 3x1 加速度 (body frame)
        gyr: 3x1 陀螺仪 (body frame)
        t_us: 当前时间戳 (us)
        do_augment: 是否进行状态增广
        """
        if not self.initialized:
            return
        
        R_k = self.state.s_R
        v_k = self.state.s_v
        p_k = self.state.s_p
        bg_k = self.state.s_bg
        ba_k = self.state.s_ba
        
        # 确保形状正确
        # print(f"Initial shapes - R_k: {R_k.shape}, v_k: {v_k.shape}, p_k: {p_k.shape}")
        if v_k.shape != (3, 1):
            v_k = v_k.reshape(3, 1)
        if p_k.shape != (3, 1):
            p_k = p_k.reshape(3, 1)
        if bg_k.shape != (3, 1):
            bg_k = bg_k.reshape(3, 1)
        if ba_k.shape != (3, 1):
            ba_k = ba_k.reshape(3, 1)
        # print(f"Corrected shapes - v_k: {v_k.shape}, p_k: {p_k.shape}")
        
        dt = (t_us - self.state.s_timestamp_us) * 1e-6  # 转换为秒
        if dt <= 0:
            return
        
        # IMU积分
        w = gyr - bg_k
        a = acc - ba_k
        
        # 旋转更新
        dtheta = w * dt
        dR = mat_exp(dtheta.flatten())

        R_kp1 = R_k @ dR
        
        # 速度更新
        # 确保形状正确
        # print(f"R_k shape: {R_k.shape}, a shape: {a.shape}, g shape: {self.g.shape}")
        a_world = R_k @ a + self.g  # 原来的正确方向
        # print(f"a_world shape: {a_world.shape}, v_k shape: {v_k.shape}")
        v_kp1 = v_k + a_world * dt
        # print(f"v_kp1 shape: {v_kp1.shape}, p_k shape: {p_k.shape}")
        
        # 位置更新
        p_kp1 = p_k + v_k * dt + 0.5 * a_world * dt**2
        # print(f"p_kp1 shape: {p_kp1.shape}")
        
        # 计算状态转移矩阵 A (15 x 15)
        A = np.eye(15)
        A[3:6, 0:3] = -hat((R_k @ a).flatten()) * dt
        A[6:9, 0:3] = -hat((R_k @ a).flatten()) * 0.5 * dt**2
        A[6:9, 3:6] = np.eye(3) * dt
        A[0:3, 9:12] = -R_kp1 @ hat(np.eye(3)) * dt  # 简化的雅可比
        A[3:6, 12:15] = -R_k * dt
        A[6:9, 12:15] = -0.5 * R_k * dt**2
        
        # 计算噪声矩阵 B (15 x 6)
        B = np.zeros((15, 6))
        B[0:3, 0:3] = -R_kp1 * dt
        B[3:6, 3:6] = -R_k * dt
        B[6:9, 3:6] = -0.5 * R_k * dt**2
        
        N = self.state.N
        
        # 先增广状态（如果需要）
        if do_augment and N < self.max_window_size:
            self.state.augment(t_us)
            N = self.state.N  # 更新N
        
        # 构建状态转移矩阵（使用更新后的N）
        A_aug = np.eye(15 + 6*N)
        A_aug[-15:, -15:] = A
        
        B_aug = np.zeros((15 + 6*N, 6))
        B_aug[-15:, :] = B
        
        # 如果是新增广的状态，需要扩展协方差矩阵
        if do_augment:
            # 扩展Sigma以包含新状态
            old_dim = 15 + 6*(N-1)
            new_dim = 15 + 6*N
            Sigma_new = np.eye(new_dim) * 1e-6
            Sigma_new[:old_dim, :old_dim] = self.Sigma
            # 新状态的协方差从当前状态复制
            if N > 1:
                Sigma_new[6*(N-1):6*N, 6*(N-1):6*N] = self.Sigma[-15:-9, -15:-9]  # 复制旋转和位置协方差
            else:
                # 第一次增广，复制当前状态的旋转和位置协方差
                Sigma_new[:6, :6] = self.Sigma[:6, :6]
            self.Sigma = Sigma_new
        
        # 传播协方差
        Sigma_k = self.Sigma
        Sigma_kp1 = A_aug @ Sigma_k @ A_aug.T + B_aug @ self.W @ B_aug.T * dt
        
        # 更新状态和协方差
        self.state.s_R = R_kp1
        self.state.s_v = v_kp1
        self.state.s_p = p_kp1
        self.state.s_timestamp_us = t_us
        
        self.Sigma = Sigma_kp1
        self.Sigma15 = self.Sigma[-15:, -15:]
        
    def update(self, meas, meas_cov, t_begin_us, t_end_us, obs_type='position'):
        """
        观测更新
        meas: 3x1 观测值 (position或displacement)
        meas_cov: 3x3 观测协方差
        t_begin_us: 起始时间戳
        t_end_us: 结束时间戳
        obs_type: 'position' (绝对位置) 或 'displacement' (位移)
        """
        if not self.initialized:
            return False
        
        # 查找对应的历史状态索引
        try:
            end_idx = self.state.si_timestamps_us.index(t_end_us)
        except ValueError:
            print(f"Timestamp not found in past states: {t_end_us}")
            return False
        
        # 构建观测模型
        R_end = self.state.si_Rs[end_idx]
        p_end = self.state.si_ps[end_idx]
        
        # 确保 p_end 是正确的形状
        p_end = np.array(p_end).squeeze()
        if p_end.shape != (3,):
            print(f"Unexpected p_end shape: {p_end.shape}, value: {p_end}")
            # 检查si_ps中的所有元素
            for i, ps in enumerate(self.state.si_ps):
                print(f"  si_ps[{i}] shape: {np.array(ps).shape}")
            return False
        
        if obs_type == 'displacement':
            # 位移观测: p_end - p_begin = meas
            try:
                begin_idx = self.state.si_timestamps_us.index(t_begin_us)
            except ValueError:
                print(f"Timestamp not found in past states: {t_begin_us}")
                return False
            p_begin = self.state.si_ps[begin_idx]
            p_begin = np.array(p_begin).squeeze()
            if p_begin.shape != (3,):
                print(f"Unexpected p_begin shape: {p_begin.shape}")
                return False
            pred = (p_end - p_begin).flatten()
            meas_flat = meas.flatten()
            
            # 计算残差
            residual = meas_flat - pred
            
            # 构建观测矩阵 H
            # H = [0, ..., -I, ..., I, ..., 0] (对应begin_idx和end_idx)
            N = self.state.N
            dim = 6 * N + 15
            H = np.zeros((3, dim))
            
            # 对begin状态的雅可比
            H[:, 6*begin_idx+3:6*begin_idx+6] = -np.eye(3)
            # 对end状态的雅可比
            H[:, 6*end_idx+3:6*end_idx+6] = np.eye(3)
        else:
            # 位置观测: p_end = meas
            pred = p_end.flatten()
            meas_flat = meas.flatten()
            
            # 确保形状匹配
            if pred.shape != meas_flat.shape:
                print(f"Shape mismatch: pred {pred.shape}, meas {meas_flat.shape}")
                return False
            
            # 计算残差
            residual = meas_flat - pred
            
            # 构建观测矩阵 H
            # H = [0, ..., I, ..., 0] (对应end_idx)
            N = self.state.N
            dim = 6 * N + 15
            H = np.zeros((3, dim))
            
            # 对end状态的雅可比
            H[:, 6*end_idx+3:6*end_idx+6] = np.eye(3)
        
        # 计算卡尔曼增益
        S = H @ self.Sigma @ H.T + meas_cov
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Singular matrix in update")
            return False
        
        K = self.Sigma @ H.T @ S_inv
        
        # 状态修正
        dX = K @ residual
        self.state.apply_correction(dX)
        
        # 协方差更新 (Joseph form)
        I_KH = np.eye(dim) - K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + K @ meas_cov @ K.T
        self.Sigma15 = self.Sigma[-15:, -15:]
        
        return True
        
    def marginalize_old_states(self, keep_count=None):
        """边缘化旧状态"""
        if keep_count is None:
            keep_count = self.max_window_size // 2
        
        if self.state.N > keep_count:
            cut_idx = self.state.N - keep_count
            
            # 边缘化协方差
            # Sigma = [Sigma_11, Sigma_12; Sigma_21, Sigma_22]
            # 边缘化后: Sigma = Sigma_22 - Sigma_21 @ Sigma_11^-1 @ Sigma_12
            dim_old = 6 * cut_idx
            dim_new = 6 * (self.state.N - cut_idx) + 15
            
            Sigma_11 = self.Sigma[:dim_old, :dim_old]
            Sigma_12 = self.Sigma[:dim_old, dim_old:]
            Sigma_21 = self.Sigma[dim_old:, :dim_old]
            Sigma_22 = self.Sigma[dim_old:, dim_old:]
            
            try:
                Sigma_11_inv = np.linalg.inv(Sigma_11)
                self.Sigma = Sigma_22 - Sigma_21 @ Sigma_11_inv @ Sigma_12
            except np.linalg.LinAlgError:
                # 如果奇异，直接丢弃
                self.Sigma = Sigma_22
            
            # 边缘化状态
            self.state.marginalize(cut_idx)
            
    def get_state(self):
        """获取当前状态"""
        return {
            'R': self.state.s_R.copy(),
            'v': self.state.s_v.copy(),
            'p': self.state.s_p.copy(),
            'bg': self.state.s_bg.copy(),
            'ba': self.state.s_ba.copy(),
            'timestamp': self.state.s_timestamp_us
        }
        
    def get_past_state(self, t_us):
        """获取历史状态"""
        try:
            idx = self.state.si_timestamps_us.index(t_us)
            return self.state.si_Rs[idx], self.state.si_ps[idx]
        except ValueError:
            return None, None
