"""
ESKF with Comprehensive Anomaly Detection
包含完整异常检测机制的ESKF实现
"""
import numpy as np
import json
from collections import deque
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


def quat_to_axis_angle(q):
    """四元数 -> 轴角 (axis, angle)"""
    qw = np.clip(q[0], -1.0, 1.0)
    angle = 2 * np.arccos(abs(qw))
    if angle < 1e-12:
        return np.array([1, 0, 0]), 0.0
    axis = q[1:] / np.sin(angle / 2)
    if qw < 0:
        axis = -axis
    return axis, angle


# ---------- 异常事件类 ----------
class AnomalyEvent:
    """记录单个异常事件的完整信息"""
    
    def __init__(self, event_type, start_frame):
        self.event_type = event_type  # 'kalman_gain', 'innovation', 'attitude_drift', 'covariance'
        self.start_frame = start_frame
        self.peak_frame = None
        self.end_frame = None
        self.duration = 0
        
        # 异常程度
        self.peak_severity = 0.0  # 峰值严重程度
        self.start_value = 0.0    # 起始值
        self.peak_value = 0.0     # 峰值
        self.end_value = 0.0      # 结束值
        
        # 滤波器状态快照
        self.state_snapshots = []
        
        # 是否已恢复
        self.is_recovered = False
        
    def add_snapshot(self, frame, state_dict):
        """添加状态快照"""
        snapshot = {
            'frame': frame,
            'timestamp': state_dict.get('timestamp', 0),
            'p': state_dict.get('p', np.zeros(3)).tolist(),
            'v': state_dict.get('v', np.zeros(3)).tolist(),
            'q': state_dict.get('q', np.array([1,0,0,0])).tolist(),
            'bg': state_dict.get('bg', np.zeros(3)).tolist(),
            'ba': state_dict.get('ba', np.zeros(3)).tolist(),
            'P_trace_rot': float(np.trace(state_dict.get('P', np.eye(15))[6:9, 6:9])),
            'P_trace_pos': float(np.trace(state_dict.get('P', np.eye(15))[0:3, 0:3])),
        }
        self.state_snapshots.append(snapshot)
        
    def update_peak(self, frame, value, severity):
        """更新峰值信息"""
        if severity > self.peak_severity:
            self.peak_frame = frame
            self.peak_value = value
            self.peak_severity = severity
            
    def finalize(self, end_frame, end_value, recovered=True):
        """结束异常事件"""
        self.end_frame = end_frame
        self.end_value = end_value
        self.duration = end_frame - self.start_frame
        self.is_recovered = recovered
        
    def to_dict(self):
        """转换为字典用于JSON序列化"""
        return {
            'event_type': self.event_type,
            'start_frame': int(self.start_frame),
            'peak_frame': int(self.peak_frame) if self.peak_frame else None,
            'end_frame': int(self.end_frame) if self.end_frame else None,
            'duration': int(self.duration),
            'peak_severity': float(self.peak_severity),
            'start_value': float(self.start_value),
            'peak_value': float(self.peak_value),
            'end_value': float(self.end_value),
            'is_recovered': bool(self.is_recovered),
            'num_snapshots': len(self.state_snapshots),
            'snapshots': self.state_snapshots[:10]  # 只保存前10个快照避免过大
        }


# ---------- 异常检测器 ----------
class AnomalyDetector:
    """
    多维度异常检测器
    
    检测类型：
    1. 卡尔曼增益异常 - 增益过大表明过度信任观测
    2. 新息异常 - 新息过大表明预测与观测严重不符
    3. 姿态偏离异常 - 与纯积分姿态的偏差过大
    4. 协方差矩阵异常 - 协方差膨胀或病态
    """
    
    def __init__(self,
                 kalman_gain_threshold=0.9,      # 卡尔曼增益阈值
                 innovation_threshold=0.5,        # 新息阈值 (rad)
                 attitude_drift_threshold=0.1,    # 姿态偏离阈值 (rad) ~5.7度
                 covariance_threshold=10.0,       # 协方差迹阈值
                 recovery_frames=100,             # 恢复确认帧数
                 snapshot_interval=10,            # 快照保存间隔
                 verbose=False):                  # 是否打印详细输出

        self.thresholds = {
            'kalman_gain': kalman_gain_threshold,
            'innovation': innovation_threshold,
            'attitude_drift': attitude_drift_threshold,
            'covariance': covariance_threshold
        }

        self.recovery_frames = recovery_frames
        self.snapshot_interval = snapshot_interval
        self.verbose = verbose
        
        # 异常事件记录
        self.anomaly_events = []
        self.active_anomalies = {}  # 当前活跃的异常 {type: AnomalyEvent}
        
        # 历史记录（用于趋势分析）
        self.history = {
            'kalman_gain': deque(maxlen=50),
            'innovation': deque(maxlen=50),
            'attitude_drift': deque(maxlen=50),
            'covariance_trace': deque(maxlen=50)
        }
        
        # 统计信息
        self.stats = {
            'total_anomalies': 0,
            'recovered_anomalies': 0,
            'persistent_anomalies': 0
        }
        
    def check_kalman_gain(self, frame, K, P_before, threshold=None):
        """
        检测卡尔曼增益是否异常
        
        Returns:
            (is_anomaly, severity, info)
        """
        if threshold is None:
            threshold = self.thresholds['kalman_gain']
            
        # 计算姿态相关的增益范数
        K_rot = K[6:9, :]
        gain_norm = np.linalg.norm(K_rot)
        
        # 计算相对于先验协方差的比例
        if P_before is not None:
            P_rot_trace = np.trace(P_before[6:9, 6:9])
            relative_gain = gain_norm / (P_rot_trace + 1e-12)
        else:
            relative_gain = gain_norm
            
        self.history['kalman_gain'].append(gain_norm)
        
        is_anomaly = gain_norm > threshold or relative_gain > threshold
        severity = max(gain_norm, relative_gain) / threshold if threshold > 0 else gain_norm
        
        info = {
            'gain_norm': float(gain_norm),
            'relative_gain': float(relative_gain),
            'threshold': threshold
        }
        
        return is_anomaly, severity, info
    
    def check_innovation(self, frame, innovation, threshold=None):
        """
        检测新息是否异常
        
        Returns:
            (is_anomaly, severity, info)
        """
        if threshold is None:
            threshold = self.thresholds['innovation']
            
        innov_norm = np.linalg.norm(innovation)
        self.history['innovation'].append(innov_norm)
        
        is_anomaly = innov_norm > threshold
        severity = innov_norm / threshold if threshold > 0 else innov_norm
        
        info = {
            'innovation_norm': float(innov_norm),
            'threshold': threshold
        }
        
        return is_anomaly, severity, info
    
    def check_attitude_drift(self, frame, q_eskf, q_integrated, threshold=None):
        """
        检测ESKF姿态与纯积分姿态的偏离
        
        Returns:
            (is_anomaly, severity, info)
        """
        if threshold is None:
            threshold = self.thresholds['attitude_drift']
            
        # 计算两个姿态之间的角度差
        q_diff = quat_multiply(quat_inverse(q_eskf), q_integrated)
        axis, angle = quat_to_axis_angle(q_diff)
        
        self.history['attitude_drift'].append(angle)
        
        is_anomaly = angle > threshold
        severity = angle / threshold if threshold > 0 else angle
        
        info = {
            'drift_angle': float(angle),
            'drift_degrees': float(np.degrees(angle)),
            'threshold': threshold
        }
        
        return is_anomaly, severity, info
    
    def check_covariance(self, frame, P, threshold=None):
        """
        检测协方差矩阵是否异常
        
        Returns:
            (is_anomaly, severity, info)
        """
        if threshold is None:
            threshold = self.thresholds['covariance']
            
        # 检查姿态协方差
        P_rot = P[6:9, 6:9]
        P_rot_trace = np.trace(P_rot)
        
        # 检查条件数
        try:
            cond_num = np.linalg.cond(P_rot)
        except:
            cond_num = np.inf
            
        # 检查是否对称
        is_symmetric = np.allclose(P_rot, P_rot.T, atol=1e-10)
        
        self.history['covariance_trace'].append(P_rot_trace)
        
        is_anomaly = P_rot_trace > threshold or cond_num > 1e6 or not is_symmetric
        severity = P_rot_trace / threshold if threshold > 0 else P_rot_trace
        
        info = {
            'P_rot_trace': float(P_rot_trace),
            'condition_number': float(cond_num),
            'is_symmetric': bool(is_symmetric),
            'threshold': threshold
        }
        
        return is_anomaly, severity, info
    
    def update_anomaly(self, event_type, frame, value, severity, state_dict, is_active):
        """
        更新异常事件状态
        
        Args:
            event_type: 异常类型
            frame: 当前帧
            value: 当前值
            severity: 严重程度
            state_dict: 滤波器状态
            is_active: 是否仍处于异常状态
        """
        if is_active:
            # 异常开始或持续
            if event_type not in self.active_anomalies:
                # 新异常开始
                event = AnomalyEvent(event_type, frame)
                event.start_value = value
                self.active_anomalies[event_type] = event
                self.stats['total_anomalies'] += 1
                if self.verbose:
                    print(f"[ANOMALY DETECTED] {event_type} at frame {frame}, severity={severity:.3f}")
            else:
                event = self.active_anomalies[event_type]

            # 更新峰值
            event.update_peak(frame, value, severity)

            # 定期保存快照
            if frame % self.snapshot_interval == 0:
                event.add_snapshot(frame, state_dict)

        else:
            # 异常结束
            if event_type in self.active_anomalies:
                event = self.active_anomalies.pop(event_type)
                event.finalize(frame, value, recovered=True)
                self.anomaly_events.append(event)
                self.stats['recovered_anomalies'] += 1
                if self.verbose:
                    print(f"[ANOMALY RECOVERED] {event_type} at frame {frame}, duration={event.duration} frames")
    
    def check_all(self, frame, state_dict, K=None, innovation=None, 
                  q_integrated=None, P_before=None):
        """
        执行所有异常检测
        
        Returns:
            dict: 包含所有检测结果的字典
        """
        results = {}
        
        # 1. 卡尔曼增益检测
        if K is not None:
            is_anomaly, severity, info = self.check_kalman_gain(frame, K, P_before)
            self.update_anomaly('kalman_gain', frame, info['gain_norm'], 
                              severity, state_dict, is_anomaly)
            results['kalman_gain'] = {'is_anomaly': is_anomaly, 'severity': severity, **info}
        
        # 2. 新息检测
        if innovation is not None:
            is_anomaly, severity, info = self.check_innovation(frame, innovation)
            self.update_anomaly('innovation', frame, info['innovation_norm'],
                              severity, state_dict, is_anomaly)
            results['innovation'] = {'is_anomaly': is_anomaly, 'severity': severity, **info}
        
        # 3. 姿态偏离检测
        if q_integrated is not None and 'q' in state_dict:
            is_anomaly, severity, info = self.check_attitude_drift(
                frame, state_dict['q'], q_integrated)
            self.update_anomaly('attitude_drift', frame, info['drift_angle'],
                              severity, state_dict, is_anomaly)
            results['attitude_drift'] = {'is_anomaly': is_anomaly, 'severity': severity, **info}
        
        # 4. 协方差检测
        if 'P' in state_dict:
            is_anomaly, severity, info = self.check_covariance(frame, state_dict['P'])
            self.update_anomaly('covariance', frame, info['P_rot_trace'],
                              severity, state_dict, is_anomaly)
            results['covariance'] = {'is_anomaly': is_anomaly, 'severity': severity, **info}
        
        return results
    
    def finalize_all(self, final_frame):
        """结束所有未完成的异常事件"""
        for event_type, event in list(self.active_anomalies.items()):
            event.finalize(final_frame, 0.0, recovered=False)
            self.anomaly_events.append(event)
            self.stats['persistent_anomalies'] += 1
            if self.verbose:
                print(f"[ANOMALY PERSISTENT] {event_type} at end, duration={event.duration} frames")
        self.active_anomalies.clear()
    
    def save_report(self, filepath):
        """保存异常检测报告到JSON"""
        report = {
            'summary': {
                'total_anomalies': self.stats['total_anomalies'],
                'recovered_anomalies': self.stats['recovered_anomalies'],
                'persistent_anomalies': self.stats['persistent_anomalies'],
                'thresholds': self.thresholds
            },
            'events': [event.to_dict() for event in self.anomaly_events]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        if self.verbose:
            print(f"\nAnomaly detection report saved to: {filepath}")
            print(f"  Total anomalies: {self.stats['total_anomalies']}")
            print(f"  Recovered: {self.stats['recovered_anomalies']}")
            print(f"  Persistent: {self.stats['persistent_anomalies']}")

        return report


# ---------- 带异常检测的ESKF ----------
class ESKF_WithAnomalyDetection:
    """
    带异常检测的ESKF
    
    在标准ESKF基础上增加：
    1. 纯积分姿态跟踪（用于对比检测）
    2. 全面的异常检测机制
    3. 异常事件记录和报告
    """
    
    def __init__(self, init_p, init_v, init_q, init_bg, init_ba,
                 sigma_g, sigma_a, sigma_bg, sigma_ba, gps_std, 
                 g=np.array([0, 0, -9.81]),
                 max_cov=1e2,
                 enable_anomaly_detection=True):
        
        # 名义状态
        self.p = init_p.copy()
        self.v = init_v.copy()
        self.q = init_q.copy() / np.linalg.norm(init_q)
        self.bg = init_bg.copy()
        self.ba = init_ba.copy()
        
        # 纯积分姿态（用于异常检测对比）
        self.q_integrated = init_q.copy() / np.linalg.norm(init_q)
        
        # 误差状态协方差
        self.P = np.eye(15) * 1e-6
        self.P[0:3, 0:3] = np.eye(3) * 0.01
        self.P[3:6, 3:6] = np.eye(3) * 0.01
        self.P[6:9, 6:9] = np.eye(3) * 0.001
        self.P[9:12, 9:12] = np.eye(3) * 1e-6
        self.P[12:15, 12:15] = np.eye(3) * 1e-6
        
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
        
        # 异常检测器（默认关闭verbose以减少输出）
        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector(verbose=False)
        else:
            self.anomaly_detector = None
            
        # 当前帧计数
        self.frame_count = 0
        
    def get_state_dict(self):
        """获取当前状态字典（用于异常检测）"""
        return {
            'p': self.p.copy(),
            'v': self.v.copy(),
            'q': self.q.copy(),
            'bg': self.bg.copy(),
            'ba': self.ba.copy(),
            'P': self.P.copy()
        }
    
    def predict(self, imu_meas, dt, sigma_a=None, sigma_g=None):
        """IMU预测步骤（带纯积分姿态更新）"""
        a_m = imu_meas[:3]
        w_m = imu_meas[3:6]
        
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
        self.q = self.q / np.linalg.norm(self.q)
        
        # 纯积分姿态更新（用于异常检测对比）
        self.q_integrated = quat_multiply(self.q_integrated, dq)
        self.q_integrated = self.q_integrated / np.linalg.norm(self.q_integrated)
        
        # ----- 2. 误差状态协方差传播 -----
        F = np.eye(15)
        F[0:3, 3:6] = self.I3 * dt
        
        a_hat = a_m - self.ba
        F[3:6, 6:9] = -R_nom @ skew(a_hat) * dt
        F[3:6, 12:15] = -R_nom * dt
        
        w_hat = w_m - self.bg
        F[6:9, 6:9] = self.I3 - skew(w_hat) * dt
        F[6:9, 9:12] = -self.I3 * dt
        
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -R_nom
        G[6:9, 3:6] = -self.I3
        G[9:12, 6:9] = self.I3
        G[12:15, 9:12] = self.I3
        
        if np.ndim(sigma_a) == 0:
            Q_a = (sigma_a ** 2) * self.I3 * dt
        else:
            Q_a = np.diag(sigma_a ** 2) * dt
            
        if np.ndim(sigma_g) == 0:
            Q_g = (sigma_g ** 2) * self.I3 * dt
        else:
            Q_g = np.diag(sigma_g ** 2) * dt
            
        # G maps the third noise block to gyro bias and the fourth block to
        # accelerometer bias. Keep Q in the same order.
        Q = block_diag(Q_a, Q_g, (self.sigma_bg ** 2) * self.I3 * dt, (self.sigma_ba ** 2) * self.I3 * dt)
        
        self.P = F @ self.P @ F.T + G @ Q @ G.T
        self._stabilize_covariance()
        
        # 异常检测
        if self.enable_anomaly_detection:
            self.anomaly_detector.check_all(
                self.frame_count, 
                self.get_state_dict(),
                q_integrated=self.q_integrated
            )
        
        self.frame_count += 1
        return self.p.copy(), self.q.copy()
    
    def update(self, gps_meas):
        """GPS更新步骤（带异常检测）"""
        P_before = self.P.copy() if self.enable_anomaly_detection else None
        
        H = np.zeros((3, 15))
        H[0:3, 0:3] = self.I3
        
        y = gps_meas - self.p
        
        S = H @ self.P @ H.T + self.R
        
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            K = np.linalg.solve(S + 1e-6 * self.I3, H @ self.P).T
        
        delta_x = K @ y
        
        # 应用更新
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        
        delta_theta = delta_x[6:9]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > 1e-12:
            axis = delta_theta / delta_theta_norm
            dq = quat_from_axis_angle(axis, delta_theta_norm)
            self.q = quat_multiply(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)
        
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]
        
        I_KH = self.I15 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self._reset_error_covariance(delta_theta)
        self._stabilize_covariance()
        
        # 异常检测
        if self.enable_anomaly_detection:
            self.anomaly_detector.check_all(
                self.frame_count,
                self.get_state_dict(),
                K=K,
                innovation=y,
                q_integrated=self.q_integrated,
                P_before=P_before
            )
        
        return self.p.copy()
    
    def update_with_attitude(self, q_meas, att_std=0.01):
        """姿态观测更新（带异常检测）"""
        P_before = self.P.copy() if self.enable_anomaly_detection else None

        # 确保四元数在相同的半球（避免双重覆盖问题）
        if np.dot(q_meas, self.q) < 0:
            q_meas = -q_meas

        # 计算相对旋转：q_err = q_meas^-1 ⊗ q_nominal
        q_meas_inv = quat_inverse(q_meas)
        q_err = quat_multiply(q_meas_inv, self.q)

        # 归一化
        q_err = q_err / (np.linalg.norm(q_err) + 1e-12)

        # 使用对数映射计算旋转向量（更稳定）
        if q_err[0] < 0:
            q_err = -q_err

        w = np.clip(q_err[0], -1.0, 1.0)
        v = q_err[1:]

        # 计算角度（使用 atan2 更稳定）
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            theta = 2 * np.arctan2(v_norm, w)
            delta_theta_meas = (v / v_norm) * theta
        else:
            delta_theta_meas = np.zeros(3)

        H = np.zeros((3, 15))
        H[0:3, 6:9] = self.I3

        y = -delta_theta_meas
        
        R_att = (att_std ** 2) * self.I3
        
        S = H @ self.P @ H.T + R_att
        
        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            K = np.linalg.solve(S + 1e-6 * self.I3, H @ self.P).T
        
        delta_x = K @ y
        
        # 应用更新
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        
        delta_theta = delta_x[6:9]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > 1e-12:
            axis = delta_theta / delta_theta_norm
            dq = quat_from_axis_angle(axis, delta_theta_norm)
            self.q = quat_multiply(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)
        
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]
        
        I_KH = self.I15 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_att @ K.T
        self._reset_error_covariance(delta_theta)
        self._stabilize_covariance()
        
        # 异常检测
        if self.enable_anomaly_detection:
            self.anomaly_detector.check_all(
                self.frame_count,
                self.get_state_dict(),
                K=K,
                innovation=y,
                q_integrated=self.q_integrated,
                P_before=P_before
            )
        
        return self.q.copy()
    
    def update_with_pose(self, pos_meas, quat_meas, pos_std, att_std):
        """
        同步位姿更新 - 同时更新位置和姿态
        
        这是解决分开更新导致偏置震荡问题的关键方法。
        当GPS和姿态观测在同一时刻到达时，应该使用此方法。
        
        Args:
            pos_meas: 位置观测 [x, y, z]
            quat_meas: 姿态四元数观测 [w, x, y, z]
            pos_std: 位置观测标准差
            att_std: 姿态观测标准差
        """
        P_before = self.P.copy() if self.enable_anomaly_detection else None
        
        # ----- 1. 构建联合观测矩阵 H (6x15) -----
        H = np.zeros((6, 15))
        H[0:3, 0:3] = self.I3   # 位置观测
        H[3:6, 6:9] = self.I3   # 姿态观测
        
        # ----- 2. 计算位置新息 -----
        y_pos = pos_meas - self.p
        
        # ----- 3. 计算姿态新息（使用更稳定的对数映射）-----
        # 确保四元数在相同的半球（避免双重覆盖问题）
        if np.dot(quat_meas, self.q) < 0:
            quat_meas = -quat_meas
        
        # 计算相对旋转：q_err = q_meas^-1 ⊗ q_nominal
        q_meas_inv = quat_inverse(quat_meas)
        q_err = quat_multiply(q_meas_inv, self.q)
        
        # 归一化
        q_err = q_err / (np.linalg.norm(q_err) + 1e-12)
        
        # 使用对数映射计算旋转向量（更稳定）
        # 对于单位四元数 [w, x, y, z]，对数映射为：
        # 如果 w > 0:  log(q) = (v / ||v||) * arccos(w) * 2
        # 如果 w < 0:  先取 -q 确保 w > 0（避免奇异性）
        if q_err[0] < 0:
            q_err = -q_err
        
        w = np.clip(q_err[0], -1.0, 1.0)
        v = q_err[1:]
        
        # 计算角度（使用 atan2 更稳定）
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-12:
            # 使用 atan2 避免 arccos 的数值问题
            theta = 2 * np.arctan2(v_norm, w)
            # 旋转向量 = 轴 * 角度
            delta_theta_meas = (v / v_norm) * theta
        else:
            delta_theta_meas = np.zeros(3)
        
        # 新息：观测相对于预测的姿态误差
        # 误差状态定义：delta_theta 是名义姿态到真实姿态的旋转
        # 观测新息应该是：delta_theta_obs = log(q_nominal^-1 ⊗ q_meas)
        # 但我们计算的是 log(q_meas^-1 ⊗ q_nominal) = -log(q_nominal^-1 ⊗ q_meas)
        # 所以新息应该是负的
        y_att = -delta_theta_meas
        
        # ----- 4. 联合新息 (6维) -----
        y = np.concatenate([y_pos, y_att])
        
        # ----- 5. 联合观测噪声协方差 (6x6) -----
        R_pose = block_diag(
            (pos_std ** 2) * self.I3,
            (att_std ** 2) * self.I3
        )
        
        # ----- 6. 计算卡尔曼增益 -----
        S = H @ self.P @ H.T + R_pose

        try:
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            I6 = np.eye(6)
            K = np.linalg.solve(S + 1e-6 * I6, H @ self.P).T
        
        # ----- 7. 计算误差状态更新 -----
        delta_x = K @ y
        
        # ----- 8. 一次性应用所有修正 -----
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        
        delta_theta = delta_x[6:9]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > 1e-12:
            axis = delta_theta / delta_theta_norm
            dq = quat_from_axis_angle(axis, delta_theta_norm)
            self.q = quat_multiply(self.q, dq)
            self.q = self.q / np.linalg.norm(self.q)
        
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]
        
        # ----- 9. 协方差更新 -----
        I_KH = self.I15 - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_pose @ K.T
        self._reset_error_covariance(delta_theta)
        self._stabilize_covariance()
        
        # ----- 10. 异常检测 -----
        if self.enable_anomaly_detection:
            self.anomaly_detector.check_all(
                self.frame_count,
                self.get_state_dict(),
                K=K,
                innovation=y,
                q_integrated=self.q_integrated,
                P_before=P_before
            )
        
        return self.p.copy(), self.q.copy()
    
    def _reset_error_covariance(self, delta_theta):
        """Apply the ESKF reset Jacobian after injecting attitude error."""
        reset_jacobian = self.I15.copy()
        reset_jacobian[6:9, 6:9] = self.I3 - 0.5 * skew(delta_theta)
        self.P = reset_jacobian @ self.P @ reset_jacobian.T

    def _stabilize_covariance(self):
        """Keep the covariance symmetric positive semidefinite.

        Element-wise clipping can turn a valid covariance into an indefinite
        matrix. Sparse updates exposed that failure after roughly ten seconds
        of propagation. Clamp eigenvalues instead so the covariance remains a
        valid PSD matrix while retaining the configured upper bound.
        """
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            self.reset_count += 1
            if self.reset_count <= 5:
                print(f"Warning: ESKF covariance contains NaN or Inf (reset #{self.reset_count}), resetting")
            self.P = np.eye(15) * 1e-4
            self.P[0:3, 0:3] = np.eye(3) * 0.01
            self.P[3:6, 3:6] = np.eye(3) * 0.01
            return
        
        symmetric_cov = 0.5 * (self.P + self.P.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric_cov)
        except np.linalg.LinAlgError:
            self.reset_count += 1
            self.P = np.eye(15) * 1e-4
            self.P[0:3, 0:3] = np.eye(3) * 0.01
            self.P[3:6, 3:6] = np.eye(3) * 0.01
            return

        eigenvalues = np.clip(eigenvalues, 1e-12, self.max_cov)
        self.P = (eigenvectors * eigenvalues) @ eigenvectors.T
        self.P = 0.5 * (self.P + self.P.T)
    
    def save_anomaly_report(self, filepath):
        """保存异常检测报告"""
        if self.enable_anomaly_detection and self.anomaly_detector:
            self.anomaly_detector.finalize_all(self.frame_count)
            return self.anomaly_detector.save_report(filepath)
        return None


# 保持向后兼容
ESKF = ESKF_WithAnomalyDetection
