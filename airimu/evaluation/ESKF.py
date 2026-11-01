import numpy as np
from scipy.linalg import block_diag
from pypose import *

# ---------- 辅助数学函数 ----------
def skew(v):
    """向量 -> 反对称矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quat_multiply(q, p):
    """四元数乘法 q ⊗ p"""
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
    """从旋转向量 δθ 构造对应的四元数增量（小角度近似）"""
    angle = np.linalg.norm(delta_theta)
    if angle < 1e-12:
        return np.array([1., 0., 0., 0.])
    axis = delta_theta / angle
    return quat_from_axis_angle(axis, angle)


def quat_to_rot(q):
    """四元数 -> 旋转矩阵"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])


# ---------- ESKF 类 ----------
class ESKF:
    def __init__(self, init_p, init_v, init_q, init_bg, init_ba,
                 sigma_g, sigma_a, sigma_bg, sigma_ba, gps_std, g=np.array([0, 0, -9.81])):
        # 名义状态
        self.p = init_p.copy()
        self.v = init_v.copy()
        self.q = init_q.copy()  # [w, x, y, z]
        self.bg = init_bg.copy()
        self.ba = init_ba.copy()

        # 误差状态协方差矩阵 (15x15)，初始不确定性
        # 使用更小的初始协方差
        self.P = np.eye(15) * 1e-6
        # 位置、速度、姿态有较小的不确定性
        self.P[0:3, 0:3] = np.eye(3) * 1e-4  # 位置
        self.P[3:6, 3:6] = np.eye(3) * 1e-2  # 速度
        self.P[6:9, 6:9] = np.eye(3) * 1e-4  # 姿态
        self.P[9:12, 9:12] = np.eye(3) * 1e-6  # 陀螺仪偏置
        self.P[12:15, 12:15] = np.eye(3) * 1e-6  # 加速度计偏置

        # 重力向量
        self.g = g.copy()

        # 噪声参数（连续域噪声密度）
        self.sigma_g = sigma_g  # 陀螺仪噪声密度 rad/s/√Hz
        self.sigma_a = sigma_a  # 加速度计噪声密度 m/s²/√Hz
        self.sigma_bg = sigma_bg  # 陀螺仪偏置随机游走 rad/s²/√Hz
        self.sigma_ba = sigma_ba  # 加速度计偏置随机游走 m/s³/√Hz

        # GPS 观测噪声协方差
        self.R = (gps_std ** 2) * np.eye(3)

        # 预分配矩阵避免重复创建
        self.I15 = np.eye(15)
        self.I3 = np.eye(3)
        self.Z3 = np.zeros((3, 3))
        
        # 协方差上限，防止数值溢出
        self.max_cov = 1e4

    def predict(self, imu_meas, dt):
        """
        IMU 测量值推进名义状态并传播误差协方差
        imu_meas: [a_m, ω_m] 加速度计和陀螺仪测量值 (body 系)
        dt: 时间步长
        """
        a_m = imu_meas[:3]
        w_m = imu_meas[3:6]

        # ----- 1. 名义状态传播（非线性）-----
        R_nom = quat_to_rot(self.q)

        # 加速度转换到世界系，并去除偏置
        a_world = R_nom @ (a_m - self.ba) + self.g

        # 位置、速度更新
        self.p = self.p + self.v * dt + 0.5 * a_world * dt ** 2
        self.v = self.v + a_world * dt

        # 姿态更新（角速度积分）
        w_corrected = w_m - self.bg
        dq = exp_quat(w_corrected * dt)
        self.q = quat_multiply(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)  # 保持单位化

        # 偏置不变（随机游走在误差动态中处理）

        # ----- 2. 构建连续时间误差状态矩阵 F_c (15x15) -----
        F_c = np.zeros((15, 15))
        # δp_dot = δv
        F_c[0:3, 3:6] = self.I3
        # δv_dot = -R [a_hat]× δθ - R δb_a
        a_hat = a_m - self.ba
        F_c[3:6, 6:9] = -R_nom @ skew(a_hat)
        F_c[3:6, 12:15] = -R_nom
        # δθ_dot = -[ω_hat]× δθ - δb_g
        w_hat = w_m - self.bg
        F_c[6:9, 6:9] = -skew(w_hat)
        F_c[6:9, 9:12] = -self.I3

        # ----- 3. 离散化状态转移矩阵 Φ = I + F_c * dt -----
        Phi = self.I15 + F_c * dt

        # ----- 4. 噪声矩阵 G_c (15x12) 和离散化 Q_d -----
        G_c = np.zeros((15, 12))
        G_c[3:6, 0:3] = -R_nom  # 加速度计白噪声
        G_c[6:9, 3:6] = -self.I3  # 陀螺仪白噪声
        G_c[9:12, 6:9] = self.I3  # 陀螺仪偏置噪声
        G_c[12:15, 9:12] = self.I3  # 加速度计偏置噪声

        # 连续噪声协方差矩阵 Qc (12x12)
        Qc = block_diag(
            (self.sigma_g ** 2) * self.I3,
            (self.sigma_a ** 2) * self.I3,
            (self.sigma_bg ** 2) * self.I3,
            (self.sigma_ba ** 2) * self.I3
        )

        # 离散化：Q_d ≈ G_c * Qc * G_c^T * dt
        Q_d = G_c @ Qc @ G_c.T * dt

        # ----- 5. 误差状态协方差传播（均值始终为0）-----
        # 使用平方根滤波或更稳定的传播方式
        # 这里使用简化的Joseph形式
        self.P = Phi @ self.P @ Phi.T + Q_d
        
        # 保证对称性
        self.P = 0.5 * (self.P + self.P.T)
        
        # 限制协方差值，防止数值溢出
        self.P = np.clip(self.P, -self.max_cov, self.max_cov)
        
        # 确保正定性（添加小的正则化项）
        self.P += 1e-9 * self.I15
        
        # 检查并修复NaN或Inf
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            print("Warning: ESKF covariance contains NaN or Inf, resetting to identity")
            self.P = np.eye(15) * 1e-4

        return self.p.copy(), self.q.copy()

    def update(self, gps_meas):
        """
        GPS 位置观测更新
        gps_meas: 3 维位置观测
        """
        # 观测矩阵 H (3x15) ：只观测位置误差 δp
        H = np.zeros((3, 15))
        H[0:3, 0:3] = self.I3

        # 计算新息（观测与名义位置之差）
        z_pred = self.p
        y = gps_meas - z_pred

        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 误差状态更新
        delta_x = K @ y

        # 将误差状态注入名义状态
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        delta_theta = delta_x[6:9]
        self.q = quat_multiply(self.q, exp_quat(delta_theta))
        self.q = self.q / np.linalg.norm(self.q)
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        # 协方差更新（使用Joseph形式提高数值稳定性）
        I_KH = self.I15 - K @ H
        P_temp = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # 保证对称性
        P_temp = 0.5 * (P_temp + P_temp.T)
        
        # 限制协方差值，防止数值溢出
        P_temp = np.clip(P_temp, -self.max_cov, self.max_cov)

        # 误差状态重置补偿矩阵 G (姿态部分)
        G_mat = self.I15.copy()
        G_mat[6:9, 6:9] = self.I3 - 0.5 * skew(delta_theta)
        self.P = G_mat @ P_temp @ G_mat.T
        
        # 再次保证对称性和正定性
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-9 * self.I15
        
        # 检查并修复NaN或Inf
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            print("Warning: ESKF covariance contains NaN or Inf after update, resetting")
            self.P = np.eye(15) * 1e-4

        return self.p.copy()