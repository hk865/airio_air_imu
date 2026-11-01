import numpy as np
from scipy.linalg import block_diag, expm

# ---------- 辅助数学函数 ----------
def skew(v):
    """向量 -> 反对称矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quat_multiply(q, p):
    """四元数乘法 q ⊗ p (Hamilton: w,x,y,z)"""
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.array([
        qw * pw - qx * px - qy * py - qz * pz,
        qw * px + qx * pw + qy * pz - qz * py,
        qw * py - qx * pz + qy * pw + qz * px,
        qw * pz + qx * py - qy * px + qz * pw
    ])


def quat_inverse(q):
    """四元数共轭（单位四元数下即为逆）"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    """用单位四元数 q 旋转向量 v"""
    q_vec = q[1:]
    q_w = q[0]
    return v + 2 * np.cross(q_vec, np.cross(q_vec, v) + q_w * v)


def quat_from_axis_angle(axis, angle):
    """轴角 -> 四元数"""
    half = angle * 0.5
    return np.array([np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)])


def exp_quat(delta_theta):
    """从旋转向量 δθ 构造对应的四元数增量"""
    angle = np.linalg.norm(delta_theta)
    if angle < 1e-12:
        return np.array([1., 0., 0., 0.])
    axis = delta_theta / angle
    return quat_from_axis_angle(axis, angle)


def quat_to_rot(q):
    """四元数 -> 旋转矩阵 (Hamilton: w,x,y,z)"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])


# ---------- 稳定的ESKF类 ----------
class ESKF_Stable:
    """
    误差状态卡尔曼滤波器 (ESKF) - 数值稳定版本
    
    使用平方根滤波和更稳定的数值方法
    """
    def __init__(self, init_p, init_v, init_q, init_bg, init_ba,
                 sigma_g, sigma_a, sigma_bg, sigma_ba, gps_std, 
                 g=np.array([0, 0, -9.81]),
                 max_cov=1e2):
        # 名义状态
        self.p = init_p.copy()
        self.v = init_v.copy()
        self.q = init_q.copy() / np.linalg.norm(init_q)  # 确保单位化 [w, x, y, z]
        self.bg = init_bg.copy()
        self.ba = init_ba.copy()

        # 误差状态协方差矩阵 (15x15)
        # 使用更合理的初始值
        self.P = np.eye(15) * 1e-6
        self.P[0:3, 0:3] = np.eye(3) * 0.01   # 位置初始不确定性 0.1m
        self.P[3:6, 3:6] = np.eye(3) * 0.01   # 速度初始不确定性 0.1m/s
        self.P[6:9, 6:9] = np.eye(3) * 0.001  # 姿态初始不确定性
        self.P[9:12, 9:12] = np.eye(3) * 1e-6  # 陀螺仪偏置
        self.P[12:15, 12:15] = np.eye(3) * 1e-6  # 加速度计偏置

        # 重力向量
        self.g = g.copy()

        # 噪声参数
        self.sigma_g = sigma_g
        self.sigma_a = sigma_a
        self.sigma_bg = sigma_bg
        self.sigma_ba = sigma_ba

        # GPS观测噪声
        self.R = (gps_std ** 2) * np.eye(3)

        # 常数矩阵
        self.I15 = np.eye(15)
        self.I3 = np.eye(3)
        
        # 协方差上限
        self.max_cov = max_cov
        
        # 重置计数
        self.reset_count = 0

    def predict(self, imu_meas, dt, sigma_a=None, sigma_g=None):
        """
        IMU预测步骤
        
        Args:
            imu_meas: IMU测量值 [a_m, w_m]
            dt: 时间步长
            sigma_a: 当前时刻加速度计噪声密度（可选，使用初始化时的值）
            sigma_g: 当前时刻陀螺仪噪声密度（可选，使用初始化时的值）
        """
        a_m = imu_meas[:3]
        w_m = imu_meas[3:6]
        
        # 使用传入的噪声参数或默认值
        if sigma_a is None:
            sigma_a = self.sigma_a
        if sigma_g is None:
            sigma_g = self.sigma_g

        # ----- 1. 名义状态传播 -----
        R_nom = quat_to_rot(self.q)
        
        # 加速度（世界系）
        a_world = R_nom @ (a_m - self.ba) + self.g
        
        # 位置和速度更新
        self.p = self.p + self.v * dt + 0.5 * a_world * dt ** 2
        self.v = self.v + a_world * dt
        
        # 姿态更新
        w_corrected = w_m - self.bg
        angle = np.linalg.norm(w_corrected) * dt
        if angle > 1e-12:
            axis = w_corrected / np.linalg.norm(w_corrected)
            dq = quat_from_axis_angle(axis, angle)
        else:
            dq = np.array([1., 0., 0., 0.])
        
        self.q = quat_multiply(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)  # 保持单位化

        # ----- 2. 误差状态协方差传播 -----
        # 构建状态转移矩阵 F
        F = np.eye(15)
        F[0:3, 3:6] = self.I3 * dt
        
        # 姿态相关项
        a_hat = a_m - self.ba
        F[3:6, 6:9] = -R_nom @ skew(a_hat) * dt
        F[3:6, 12:15] = -R_nom * dt
        
        w_hat = w_m - self.bg
        F[6:9, 6:9] = self.I3 - skew(w_hat) * dt
        F[6:9, 9:12] = -self.I3 * dt

        # 构建噪声矩阵 G
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R_nom
        G[6:9, 3:6] = -self.I3
        G[9:12, 6:9] = self.I3
        G[12:15, 9:12] = self.I3

        # 离散噪声协方差（使用逐时刻、逐轴的噪声参数）
        # sigma_a和sigma_g可以是标量或形状为(3,)的数组
        if np.ndim(sigma_a) == 0:
            Q_a = (sigma_a ** 2) * self.I3 * dt
        else:
            Q_a = np.diag(sigma_a ** 2) * dt
            
        if np.ndim(sigma_g) == 0:
            Q_g = (sigma_g ** 2) * self.I3 * dt
        else:
            Q_g = np.diag(sigma_g ** 2) * dt
            
        Q = block_diag(
            Q_a,
            Q_g,
            (self.sigma_ba ** 2) * self.I3 * dt,
            (self.sigma_bg ** 2) * self.I3 * dt
        )

        # 协方差传播
        self.P = F @ self.P @ F.T + G @ Q @ G.T
        
        # 数值稳定性处理
        self._stabilize_covariance()

        return self.p.copy(), self.q.copy()

    def update(self, gps_meas):
        """
        GPS更新步骤
        """
        # 观测矩阵
        H = np.zeros((3, 15))
        H[0:3, 0:3] = self.I3

        # 新息
        y = gps_meas - self.p

        # 新息协方差
        S = H @ self.P @ H.T + self.R
        
        # 卡尔曼增益（使用更稳定的求解方法）
        try:
            # 使用Cholesky分解求解
            L = np.linalg.cholesky(S + 1e-9 * self.I3)
            K = self.P @ H.T @ np.linalg.inv(L).T @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用直接求逆
            K = self.P @ H.T @ np.linalg.inv(S + 1e-6 * self.I3)

        # 误差状态更新
        delta_x = K @ y

        # 将误差注入名义状态
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        
        # 姿态更新（使用小角度近似）
        delta_theta = delta_x[6:9]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > 1e-12:
            axis = delta_theta / delta_theta_norm
            dq = quat_from_axis_angle(axis, delta_theta_norm)
            self.q = quat_multiply(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)
        
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # 协方差更新（Joseph形式）
        I_KH = self.I15 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # 数值稳定性处理
        self._stabilize_covariance()

        return self.p.copy()

    def update_with_velocity(self, gps_vel, vel_std=0.1):
        """
        GPS速度观测更新 - 用于约束姿态（特别是航向角）
        
        Args:
            gps_vel: GPS测量的速度（世界系）
            vel_std: 速度观测标准差
        """
        # 观测矩阵 H (3x15) ：观测速度误差 δv
        H = np.zeros((3, 15))
        H[0:3, 3:6] = self.I3

        # 新息（观测与名义速度之差）
        y = gps_vel - self.v

        # 速度观测噪声协方差
        R_vel = (vel_std ** 2) * np.eye(3)

        # 新息协方差
        S = H @ self.P @ H.T + R_vel

        # 卡尔曼增益
        try:
            L = np.linalg.cholesky(S + 1e-9 * self.I3)
            K = self.P @ H.T @ np.linalg.inv(L).T @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-6 * self.I3)

        # 误差状态更新
        delta_x = K @ y

        # 将误差注入名义状态
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]

        # 姿态更新
        delta_theta = delta_x[6:9]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > 1e-12:
            axis = delta_theta / delta_theta_norm
            dq = quat_from_axis_angle(axis, delta_theta_norm)
            self.q = quat_multiply(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)

        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # 协方差更新
        I_KH = self.I15 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_vel @ K.T

        # 数值稳定性处理
        self._stabilize_covariance()

        return self.v.copy()

    def update_with_attitude(self, q_meas, att_std=0.01):
        """
        姿态观测更新 - 直接使用姿态测量（如双天线GPS或真值）
        
        Args:
            q_meas: 姿态四元数测量值 [w, x, y, z]
            att_std: 姿态观测标准差（弧度）
        """
        # 计算姿态误差（在切空间）
        # 误差四元数：q_err = q_meas_inv ⊗ q_nominal
        q_meas_inv = quat_inverse(q_meas)
        q_err = quat_multiply(q_meas_inv, self.q)
        
        # 转换为旋转向量（小角度近似）
        # q_err = [cos(θ/2), sin(θ/2) * axis]
        # 对于小角度：δθ ≈ 2 * q_err_vec
        if q_err[0] > 1.0:
            q_err = q_err / np.linalg.norm(q_err)
        theta = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
        if theta > 1e-12:
            axis = q_err[1:] / np.sin(theta / 2)
            delta_theta_meas = theta * axis
        else:
            delta_theta_meas = np.zeros(3)
        
        # 观测矩阵 H (3x15) ：只观测姿态误差
        H = np.zeros((3, 15))
        H[0:3, 6:9] = self.I3

        # 新息（姿态误差）
        y = -delta_theta_meas  # 负号是因为我们估计的是名义到测量的误差

        # 姿态观测噪声协方差
        R_att = (att_std ** 2) * np.eye(3)

        # 新息协方差
        S = H @ self.P @ H.T + R_att

        # 卡尔曼增益
        try:
            L = np.linalg.cholesky(S + 1e-9 * self.I3)
            K = self.P @ H.T @ np.linalg.inv(L).T @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.inv(S + 1e-6 * self.I3)

        # 误差状态更新
        delta_x = K @ y

        # 将误差注入名义状态
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]

        # 姿态更新
        delta_theta = delta_x[6:9]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > 1e-12:
            axis = delta_theta / delta_theta_norm
            dq = quat_from_axis_angle(axis, delta_theta_norm)
            self.q = quat_multiply(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)

        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # 协方差更新
        I_KH = self.I15 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_att @ K.T

        # 数值稳定性处理
        self._stabilize_covariance()

        return self.q.copy()

    def _stabilize_covariance(self):
        """
        协方差矩阵数值稳定性处理
        """
        # 检查NaN或Inf（首先检查，防止传播）
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self.reset_count += 1
            if self.reset_count <= 5:  # 只打印前5次
                print(f"Warning: ESKF covariance contains NaN or Inf (reset #{self.reset_count}), resetting")
            # 重置为合理的初始值
            self.P = np.eye(15) * 1e-4
            self.P[0:3, 0:3] = np.eye(3) * 0.01
            self.P[3:6, 3:6] = np.eye(3) * 0.01
            return
        
        # 保证对称性
        self.P = 0.5 * (self.P + self.P.T)
        
        # 限制协方差值（更严格的限制）
        self.P = np.clip(self.P, -self.max_cov, self.max_cov)
        
        # 确保对角线元素为正且合理
        for i in range(15):
            if self.P[i, i] < 1e-12:
                self.P[i, i] = 1e-12
            elif self.P[i, i] > self.max_cov:
                self.P[i, i] = self.max_cov
        
        # 检查非对角线元素（确保不超过对角线元素的乘积）
        for i in range(15):
            for j in range(i+1, 15):
                max_off_diag = np.sqrt(self.P[i, i] * self.P[j, j])
                if abs(self.P[i, j]) > max_off_diag:
                    self.P[i, j] = np.sign(self.P[i, j]) * max_off_diag * 0.9
                    self.P[j, i] = self.P[i, j]


# 保持向后兼容
ESKF = ESKF_Stable
