# GPS + AirIMU + AirIO EKF Fusion Runner
# 融合模拟GPS信号、AirIMU模型输出和AirIO速度估计的EKF

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import argparse
import numpy as np
import pypose as pp
import tqdm
import pickle
import io
import json

import torch
import torch.utils.data as Data
from utils import move_to

from pyhocon import ConfigFactory
from datasets import imu_seq_collate, SeqDataset, SeqInfDataset

from utils import CPU_Unpickler, integrate, interp_xyz
from utils.velocity_integrator import Velocity_Integrator, integrate_pos
from utils.visualize_state import visualize_velocity, visualize_rotations

from ekf import IMUEKF
from IMUstate import IMUstate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ekfutil import plot_bias_subplots


class SingleIMUWithGPS(IMUstate):
    """
    IMU状态转移模型，支持GPS位置观测
    状态: R(3), V(3), P(3), bg(3), ba(3) = 15维
    """
    def __init__(self):
        super().__init__()

    def state_transition(self, state: torch.Tensor, input: torch.Tensor, dt: torch.Tensor, t: torch.Tensor=None):
        """
        IMU状态转移方程
        state: [R(3), V(3), P(3), bg(3), ba(3)]
        input: [gyro(3), acc(3), dbg(3), dba(3)]
        """
        init_rot = pp.so3(state[..., :3]).Exp()
        bg = input[..., 6:9]
        ba = input[..., 9:12]

        w = (input[..., 0:3] - bg)
        a = (input[..., 3:6] - init_rot.Inv() @ self.gravity.double() - ba)
        
        Dr = pp.so3(w * dt).Exp()
        Dv = Dr @ a * dt
        Dp = Dv * dt + Dr @ a * 0.5 * dt**2
        
        R = (init_rot @ Dr).Log()
        V = state[..., 3:6] + init_rot @ Dv
        P = state[..., 6:9] + state[..., 3:6] * dt + init_rot @ Dp

        return torch.cat([R, V, P, bg, ba], dim=-1).tensor()

    def observation(self, state, input, dt, t=None):
        """
        默认观测函数 - 位置观测 (用于GPS融合)
        这是IMUstate.set_refpoint所需要的
        """
        return self.observation_position(state, input, dt, t)

    def observation_velocity(self, state, input, dt, t=None):
        """
        速度观测: 将世界坐标系速度转换到IMU坐标系
        """
        nstate = self.state_transition(state, input, dt)
        rot = pp.so3(nstate[..., :3]).Exp()
        velo = rot.Inv() @ nstate[..., 3:6]
        return velo

    def observation_position(self, state, input, dt, t=None):
        """
        位置观测: 直接输出世界坐标系位置
        """
        nstate = self.state_transition(state, input, dt)
        pos = nstate[..., 6:9]
        return pos
    
    def observation_velocity_position(self, state, input, dt, t=None):
        """
        速度和位置联合观测: [V_body(3), P_world(3)]
        """
        nstate = self.state_transition(state, input, dt)
        rot = pp.so3(nstate[..., :3]).Exp()
        velo_body = rot.Inv() @ nstate[..., 3:6]
        pos_world = nstate[..., 6:9]
        return torch.cat([velo_body, pos_world], dim=-1)


class GPS_IMU_EKF_Runner:
    """
    GPS + AirIMU + AirIO EKF融合运行器
    支持多种融合模式:
    1. GPS_ONLY: 仅GPS位置观测
    2. AIRIO_ONLY: 仅AirIO速度观测
    3. GPS_AIRIO: GPS位置 + AirIO速度观测
    4. GPS_AIRIMU: GPS位置 + AirIMU校正后的IMU数据
    5. FULL_FUSION: GPS + AirIO + AirIMU
    """
    def __init__(self, model, mode='FULL_FUSION',
                 q_diag=None,
                 r_velo=torch.ones(3, dtype=torch.float64) * 0.01,
                 r_pos=torch.ones(3, dtype=torch.float64) * 0.1,
                 p_diag=None):
        self.state = torch.zeros(15, dtype=torch.float64)
        self.model = model
        self.mode = mode
        self.current_idx = 0

        # 观测噪声协方差
        self.r_velo = r_velo  # 速度观测噪声
        self.r_pos = r_pos    # 位置观测噪声
        
        # 过程噪声协方差Q (12x12对角矩阵)
        if q_diag is None:
            # 默认过程噪声: [gyro_noise(3), acc_noise(3), gyro_bias_noise(3), acc_bias_noise(3)]
            q_diag = torch.tensor([0.001, 0.001, 0.001,  # gyro
                                   0.01, 0.01, 0.01,     # acc
                                   1e-6, 1e-6, 1e-6,     # gyro bias random walk
                                   1e-5, 1e-5, 1e-5],    # acc bias random walk
                                  dtype=torch.float64)
        self.q_diag = q_diag
        Q = torch.diag(q_diag ** 2)
        
        # 根据融合模式设置观测噪声
        # 注意: r_velo和r_pos已经是方差(不是标准差)，参考IMUofflinerunner.py的实现
        if mode == 'GPS_ONLY':
            R = torch.eye(3, dtype=torch.float64) * self.r_pos
        elif mode == 'AIRIO_ONLY':
            R = torch.eye(3, dtype=torch.float64) * self.r_velo
        elif mode in ['GPS_AIRIO', 'GPS_AIRIMU', 'FULL_FUSION']:
            R = torch.eye(6, dtype=torch.float64)
            R[:3, :3] = torch.eye(3) * self.r_velo
            R[3:6, 3:6] = torch.eye(3) * self.r_pos
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        self.filter = IMUEKF(self.model, Q=Q, R=R).double()

        # 状态协方差P (15x15对角矩阵)
        if p_diag is None:
            # 默认初始协方差: [rot(3), vel(3), pos(3), gyro_bias(3), acc_bias(3)]
            # 减小旋转初始协方差，因为GPS不直接观测旋转
            p_diag = torch.tensor([0.01, 0.01, 0.01,   # rotation (减小，GPS不观测旋转)
                                   0.1, 0.1, 0.1,      # velocity
                                   0.1, 0.1, 0.1,      # position
                                   0.01, 0.01, 0.01,   # gyro bias
                                   0.1, 0.1, 0.1],     # acc bias
                                  dtype=torch.float64)
        self.p_diag = p_diag
        self.P = torch.diag(p_diag ** 2)
        self.P_list = [self.P]
        self.est_history = []
        self.gravity = torch.tensor([0., 0., 9.81007])
        
        # 记录观测使用情况
        self.gps_update_count = 0
        self.airio_update_count = 0
        
    def get_result(self):
        return torch.stack(self.est_history), torch.stack(self.P_list)

    def propagate_update(self, imu_data, observation, obs_type='velocity', Q=None, R=None):
        """
        执行预测+更新步骤
        obs_type: 'velocity', 'position', 'velo_pos'
        """
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input_tensor = torch.cat([imu_data["gyro"], imu_data["acc"], 
                                   d_bias_gyro, d_bias_acc], dim=-1)
        
        # 根据观测类型选择观测函数
        if obs_type == 'velocity':
            self.model.observation = self.model.observation_velocity
        elif obs_type == 'position':
            self.model.observation = self.model.observation_position
        elif obs_type == 'velo_pos':
            self.model.observation = self.model.observation_velocity_position
            
        self.state, self.P = self.filter(state=self.state, obs=observation, 
                                          input=input_tensor, P=self.P, 
                                          dt=imu_data["dt"], Q=Q, R=R)

    def propagate_state(self, imu_data, Q=None):
        """仅执行预测步骤"""
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input_tensor = torch.cat([imu_data["gyro"], imu_data["acc"], 
                                   d_bias_gyro, d_bias_acc], dim=-1)
        self.state, self.P = self.filter.state_propogate(state=self.state, 
                                                          input=input_tensor, 
                                                          P=self.P, 
                                                          dt=imu_data["dt"], 
                                                          Q=Q)

    def run(self, imu_data, gps_obs=None, airio_obs=None, 
            gps_cov=None, airio_cov=None, Q=None, use_imu_cov=False,
            input_weight=1e2, bias_weight=1e-12, obs_weight=1e-1):
        """
        运行EKF一步
        gps_obs: GPS位置观测 [x, y, z]
        airio_obs: AirIO速度观测 [vx, vy, vz] (body frame)
        use_imu_cov: 是否使用IMU数据中的协方差（AirIMU学习到的）
        input_weight: AirIMU加速度协方差权重 (默认100，参考IMUofflinerunner.py)
        bias_weight: bias随机游走噪声 (默认1e-12)
        obs_weight: AirIO速度协方差权重 (默认0.1，参考IMUofflinerunner.py)
        """
        # 构建过程噪声协方差 Q (12x12)
        # q = [gyro_noise(3), acc_noise(3), gyro_bias_RW(3), acc_bias_RW(3)]
        if Q is None:
            if use_imu_cov and "gyro_cov" in imu_data and "acc_cov" in imu_data:
                # 使用AirIMU学习到的协方差 (参考IMUofflinerunner.py的实现)
                # AirIMU输出的cov已经是方差(covariance)，不需要再平方
                q = torch.ones(12, dtype=torch.float64) * bias_weight  # bias随机游走
                # Gyro cov可能过小(~4e-8)，设置下限以保证旋转可以通过GPS间接校正
                gyro_cov = imu_data["gyro_cov"]
                gyro_cov = torch.clamp(gyro_cov, min=1e-6)  # 设置下限为1e-6
                q[:3] = gyro_cov                                        # AirIMU陀螺仪协方差(方差)
                q[3:6] = imu_data["acc_cov"] * input_weight             # AirIMU加速度协方差(方差) * 权重
                # cov已经是方差，直接使用，不要再平方
                Q = torch.diag(q)
            else:
                # 使用默认的过程噪声
                # q_diag是标准差，需要平方得到方差
                Q = torch.diag(self.q_diag ** 2)

        # 根据融合模式决定观测
        # 参考IMUofflinerunner.py: obs_weight = 0.1 用于平衡AirIO的不确定性
        
        if self.mode == 'GPS_ONLY':
            if gps_obs is not None:
                # r_pos已经是方差，不需要再平方
                R = torch.eye(3, dtype=torch.float64) * (gps_cov if gps_cov is not None else self.r_pos)
                self.model.observation = self.model.observation_position
                self.propagate_update(imu_data, gps_obs, obs_type='position', Q=Q, R=R)
                self.gps_update_count += 1
            else:
                self.propagate_state(imu_data, Q=Q)

        elif self.mode == 'AIRIO_ONLY':
            if airio_obs is not None:
                # AirIO速度观测噪声: 使用AirIO学习到的cov * obs_weight (参考IMUofflinerunner.py)
                if airio_cov is not None:
                    r_velo = airio_cov * obs_weight
                else:
                    r_velo = torch.ones(3, dtype=torch.float64) * self.r_velo  # r_velo已经是方差
                R = torch.diag(r_velo)
                self.model.observation = self.model.observation_velocity
                self.propagate_update(imu_data, airio_obs, obs_type='velocity', Q=Q, R=R)
                self.airio_update_count += 1
            else:
                self.propagate_state(imu_data, Q=Q)

        elif self.mode == 'GPS_AIRIO':
            if gps_obs is not None and airio_obs is not None:
                # 联合观测: [velocity_body(3), position_world(3)]
                combined_obs = torch.cat([airio_obs, gps_obs], dim=-1)
                R = torch.eye(6, dtype=torch.float64)
                # AirIO速度观测噪声
                if airio_cov is not None:
                    R[:3, :3] = torch.diag(airio_cov * obs_weight)
                else:
                    R[:3, :3] = torch.eye(3) * self.r_velo  # r_velo已经是方差
                # GPS位置观测噪声
                R[3:6, 3:6] = torch.eye(3) * (gps_cov if gps_cov is not None else self.r_pos)  # r_pos已经是方差
                self.propagate_update(imu_data, combined_obs, obs_type='velo_pos', Q=Q, R=R)
                self.gps_update_count += 1
                self.airio_update_count += 1
            elif gps_obs is not None:
                # 只有GPS
                R = torch.eye(3, dtype=torch.float64) * (gps_cov if gps_cov is not None else self.r_pos)  # r_pos已经是方差
                self.model.observation = self.model.observation_position
                self.propagate_update(imu_data, gps_obs, obs_type='position', Q=Q, R=R)
                self.gps_update_count += 1
            elif airio_obs is not None:
                # 只有AirIO
                if airio_cov is not None:
                    r_velo = airio_cov * obs_weight
                else:
                    r_velo = torch.ones(3, dtype=torch.float64) * self.r_velo  # r_velo已经是方差
                R = torch.diag(r_velo)
                self.model.observation = self.model.observation_velocity
                self.propagate_update(imu_data, airio_obs, obs_type='velocity', Q=Q, R=R)
                self.airio_update_count += 1
            else:
                self.propagate_state(imu_data, Q=Q)

        elif self.mode == 'GPS_AIRIMU':
            # GPS + AirIMU校正后的IMU (AirIMU已在输入数据中体现)
            if gps_obs is not None:
                R = torch.eye(3, dtype=torch.float64) * (gps_cov if gps_cov is not None else self.r_pos)  # r_pos已经是方差
                self.model.observation = self.model.observation_position
                self.propagate_update(imu_data, gps_obs, obs_type='position', Q=Q, R=R)
                self.gps_update_count += 1
            else:
                self.propagate_state(imu_data, Q=Q)

        elif self.mode == 'FULL_FUSION':
            # GPS + AirIO + AirIMU
            if gps_obs is not None and airio_obs is not None:
                combined_obs = torch.cat([airio_obs, gps_obs], dim=-1)
                R = torch.eye(6, dtype=torch.float64)
                # AirIO速度观测噪声 (使用obs_weight=0.1)
                if airio_cov is not None:
                    R[:3, :3] = torch.diag(airio_cov * obs_weight)
                else:
                    R[:3, :3] = torch.eye(3) * self.r_velo  # r_velo已经是方差
                # GPS位置观测噪声
                R[3:6, 3:6] = torch.eye(3) * (gps_cov if gps_cov is not None else self.r_pos)  # r_pos已经是方差
                self.propagate_update(imu_data, combined_obs, obs_type='velo_pos', Q=Q, R=R)
                self.gps_update_count += 1
                self.airio_update_count += 1
            elif gps_obs is not None:
                R = torch.eye(3, dtype=torch.float64) * (gps_cov if gps_cov is not None else self.r_pos)  # r_pos已经是方差
                self.model.observation = self.model.observation_position
                self.propagate_update(imu_data, gps_obs, obs_type='position', Q=Q, R=R)
                self.gps_update_count += 1
            elif airio_obs is not None:
                # 只有AirIO
                if airio_cov is not None:
                    r_velo = airio_cov * obs_weight
                else:
                    r_velo = torch.ones(3, dtype=torch.float64) * self.r_velo  # r_velo已经是方差
                R = torch.diag(r_velo)
                self.model.observation = self.model.observation_velocity
                self.propagate_update(imu_data, airio_obs, obs_type='velocity', Q=Q, R=R)
                self.airio_update_count += 1
            else:
                self.propagate_state(imu_data, Q=Q)

        self.est_history.append(self.state.clone())
        self.P_list.append(self.P.clone())
        self.current_idx += 1


def compute_ate(est_pos, gt_pos):
    """
    计算绝对轨迹误差 (Absolute Trajectory Error)
    ATE = sqrt(mean(||est_pos - gt_pos||^2))
    """
    errors = np.linalg.norm(est_pos - gt_pos, axis=1)
    ate = np.sqrt(np.mean(errors ** 2))
    ate_mean = np.mean(errors)
    ate_std = np.std(errors)
    ate_max = np.max(errors)
    return {'ate': ate, 'mean': ate_mean, 'std': ate_std, 'max': ate_max, 'errors': errors}


def compute_aoe(est_rot, gt_rot):
    """
    计算绝对方向误差 (Absolute Orientation Error)
    AOE = mean(angle(est_rot, gt_rot))
    输入: est_rot, gt_rot 为四元数 [N, 4] (w,x,y,z)
    """
    if isinstance(est_rot, torch.Tensor):
        est_rot = est_rot.numpy()
    if isinstance(gt_rot, torch.Tensor):
        gt_rot = gt_rot.numpy()
    
    # 确保是float64类型
    est_rot = np.array(est_rot, dtype=np.float64)
    gt_rot = np.array(gt_rot, dtype=np.float64)
    
    # 计算旋转误差
    angles = []
    for i in range(len(est_rot)):
        # 使用SO3处理四元数 (pp.SO3 expects [N, 4] quaternion)
        est_SO3 = pp.SO3(torch.tensor(est_rot[i], dtype=torch.float64))
        gt_SO3 = pp.SO3(torch.tensor(gt_rot[i], dtype=torch.float64))
        # 计算相对旋转
        rel_rot = gt_SO3.Inv() @ est_SO3
        # 获取旋转角度 (so3.Log() returns Lie algebra vector, norm gives angle)
        angle = rel_rot.Log().norm().item()
        angles.append(angle)
    
    angles = np.array(angles)
    aoe = np.mean(angles)
    aoe_std = np.std(angles)
    aoe_max = np.max(angles)
    return {'aoe': aoe, 'mean': aoe, 'std': aoe_std, 'max': aoe_max, 'angles': angles}


def compute_rte(est_pos, gt_pos, window_size=100):
    """
    计算相对轨迹误差 (Relative Trajectory Error)
    RTE = mean(||(est_pos[i+window] - est_pos[i]) - (gt_pos[i+window] - gt_pos[i])||)
    """
    if len(est_pos) < window_size:
        return {'rte': 0, 'mean': 0, 'std': 0, 'errors': np.array([0])}
    
    errors = []
    for i in range(len(est_pos) - window_size):
        est_rel = est_pos[i + window_size] - est_pos[i]
        gt_rel = gt_pos[i + window_size] - gt_pos[i]
        error = np.linalg.norm(est_rel - gt_rel)
        errors.append(error)
    
    errors = np.array(errors)
    rte = np.mean(errors)
    rte_std = np.std(errors)
    rte_max = np.max(errors)
    return {'rte': rte, 'mean': rte, 'std': rte_std, 'max': rte_max, 'errors': errors}


def compute_roe(est_rot, gt_rot, window_size=100):
    """
    计算相对方向误差 (Relative Orientation Error)
    ROE = mean(angle(est_rot[i].Inv() @ est_rot[i+window], gt_rot[i].Inv() @ gt_rot[i+window]))
    输入: est_rot, gt_rot 为四元数 [N, 4] (w,x,y,z)
    """
    if isinstance(est_rot, torch.Tensor):
        est_rot = est_rot.numpy()
    if isinstance(gt_rot, torch.Tensor):
        gt_rot = gt_rot.numpy()
    
    # 确保是float64类型
    est_rot = np.array(est_rot, dtype=np.float64)
    gt_rot = np.array(gt_rot, dtype=np.float64)
    
    if len(est_rot) < window_size:
        return {'roe': 0, 'mean': 0, 'std': 0, 'max': 0, 'angles': np.array([0])}
    
    angles = []
    for i in range(len(est_rot) - window_size):
        # 估计的相对旋转 (使用SO3处理四元数)
        est_SO3_i = pp.SO3(torch.tensor(est_rot[i], dtype=torch.float64))
        est_SO3_j = pp.SO3(torch.tensor(est_rot[i + window_size], dtype=torch.float64))
        est_rel = est_SO3_i.Inv() @ est_SO3_j
        
        # 真实的相对旋转
        gt_SO3_i = pp.SO3(torch.tensor(gt_rot[i], dtype=torch.float64))
        gt_SO3_j = pp.SO3(torch.tensor(gt_rot[i + window_size], dtype=torch.float64))
        gt_rel = gt_SO3_i.Inv() @ gt_SO3_j
        
        # 计算角度误差
        angle = (gt_rel.Inv() @ est_rel).Log().norm().item()
        angles.append(angle)
    
    angles = np.array(angles)
    roe = np.mean(angles)
    roe_std = np.std(angles)
    roe_max = np.max(angles)
    return {'roe': roe, 'mean': roe, 'std': roe_std, 'max': roe_max, 'angles': angles}


def plot_3d_trajectory(est_pos, gt_pos, save_path=None, title="3D Trajectory"):
    """
    绘制三维轨迹对比图
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制估计轨迹
    ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 
            'b-', label='Estimated', linewidth=2, alpha=0.8)
    
    # 绘制真实轨迹
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
            'r--', label='Ground Truth', linewidth=2, alpha=0.8)
    
    # 绘制起点和终点
    ax.scatter(*est_pos[0], c='blue', marker='o', s=100, label='Est Start')
    ax.scatter(*est_pos[-1], c='blue', marker='s', s=100, label='Est End')
    ax.scatter(*gt_pos[0], c='red', marker='o', s=100, label='GT Start')
    ax.scatter(*gt_pos[-1], c='red', marker='s', s=100, label='GT End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 设置等比例
    max_range = np.array([est_pos[:, 0].max() - est_pos[:, 0].min(),
                          est_pos[:, 1].max() - est_pos[:, 1].min(),
                          est_pos[:, 2].max() - est_pos[:, 2].min()]).max() / 2.0
    mid_x = (est_pos[:, 0].max() + est_pos[:, 0].min()) * 0.5
    mid_y = (est_pos[:, 1].max() + est_pos[:, 1].min()) * 0.5
    mid_z = (est_pos[:, 2].max() + est_pos[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    # 注意：不调用plt.show()以避免阻塞


def simulate_gps(gt_pos, gps_freq=1, imu_freq=200, noise_std=0.1, bias=None):
    """
    模拟GPS信号
    gt_pos: 真实位置 [N, 3]
    gps_freq: GPS频率 (Hz)
    imu_freq: IMU频率 (Hz)
    noise_std: GPS噪声标准差 (m)
    bias: GPS偏置 [3]
    
    返回: gps_data dict with 'pos', 'cov', 'indices'
    """
    N = len(gt_pos)
    ratio = imu_freq // gps_freq
    
    # GPS采样索引
    gps_indices = list(range(0, N, ratio))
    
    # 添加噪声
    noise = torch.randn(len(gps_indices), 3) * noise_std
    if bias is not None:
        noise += bias
    
    gps_pos = gt_pos[gps_indices] + noise
    gps_cov = torch.ones(len(gps_indices), 3) * (noise_std ** 2)
    
    return {
        'pos': gps_pos,
        'cov': gps_cov,
        'indices': gps_indices,
        'noise_std': noise_std
    }


def run_pure_imu_baseline(dataset_inf, init, timestamps, gt_pos_all, gt_vel_all, gt_rot_all, args):
    """
    运行纯IMU基线（无观测更新）
    """
    print("  Running Pure IMU Baseline...")
    
    model = SingleIMUWithGPS().double()
    
    # 构建过程噪声Q对角线向量
    q_diag = torch.tensor([
        args.gyro_noise, args.gyro_noise, args.gyro_noise,
        args.acc_noise, args.acc_noise, args.acc_noise,
        args.gyro_bias_noise, args.gyro_bias_noise, args.gyro_bias_noise,
        args.acc_bias_noise, args.acc_bias_noise, args.acc_bias_noise
    ], dtype=torch.float64)
    
    # 构建初始状态协方差P对角线向量
    p_diag = torch.tensor([
        args.init_rot_std, args.init_rot_std, args.init_rot_std,
        args.init_vel_std, args.init_vel_std, args.init_vel_std,
        args.init_pos_std, args.init_pos_std, args.init_pos_std,
        args.init_gyro_bias_std, args.init_gyro_bias_std, args.init_gyro_bias_std,
        args.init_acc_bias_std, args.init_acc_bias_std, args.init_acc_bias_std
    ], dtype=torch.float64)
    
    ekf = GPS_IMU_EKF_Runner(model, mode='GPS_ONLY', q_diag=q_diag, p_diag=p_diag)
    
    # 状态初始化
    initial_state = torch.zeros(15, dtype=torch.float64)
    initial_state[:3] = init["rot"].Log()
    initial_state[3:6] = init["vel"]
    initial_state[6:9] = init["pos"]
    
    # 设置初始bias（如果指定）
    if args.init_gyro_bias is not None:
        initial_state[9:12] = torch.tensor(args.init_gyro_bias, dtype=torch.float64)
    if args.init_acc_bias is not None:
        initial_state[12:15] = torch.tensor(args.init_acc_bias, dtype=torch.float64)
        
    ekf.state = initial_state

    # 调试模式：限制处理帧数
    max_frames = args.max_frames if args.debug else len(dataset_inf)
    if args.debug:
        print(f"    DEBUG MODE: Processing only first {max_frames} frames")

    # 运行纯预测
    t_range = tqdm.tqdm(range(min(max_frames, len(dataset_inf))), desc="  Pure IMU")
    for i in t_range:
        data = dataset_inf[i]
        
        imu_data = {
            "gyro": data["gyro"][0],
            "acc": data["acc"][0],
            "dt": data["dt"][0],
            "gyro_cov": data["gyro_cov"][0] if data["gyro_cov"] is not None else torch.ones(3) * args.gyro_noise,
            "acc_cov": data["acc_cov"][0] if data["acc_cov"] is not None else torch.ones(3) * args.acc_noise
        }
        
        # 纯预测，无观测更新
        ekf.propagate_state(imu_data)
        ekf.est_history.append(ekf.state.clone())
        ekf.P_list.append(ekf.P.clone())
        ekf.current_idx += 1
    
    ekf_result = torch.stack(ekf.est_history).numpy()
    
    # 计算误差
    gtpos = gt_pos_all[:len(ekf_result)].numpy()
    gtvel = gt_vel_all[:len(ekf_result)].numpy()
    
    pos_error = np.linalg.norm(gtpos - ekf_result[:, 6:9], axis=1)
    vel_error = np.linalg.norm(gtvel - ekf_result[:, 3:6], axis=1)
    
    print(f"    Mean position error: {np.mean(pos_error):.4f} m")
    print(f"    Final position error: {pos_error[-1]:.4f} m")
    
    return ekf_result, pos_error, vel_error


def run_experiment(args):
    """
    运行GPS+IMU融合实验
    """
    print("\n" + "="*60)
    print(f"Running GPS+IMU EKF Fusion Experiment")
    print(f"Mode: {args.mode}")
    print("="*60 + "\n")
    
    # 加载配置
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    
    # 加载AirIMU模型输出
    airimu_results = {}
    if args.airimu_exp is not None:
        airimu_paths = args.airimu_exp.split(',')
        for i, path in enumerate(airimu_paths):
            path = path.strip().replace('/', os.sep).replace('\\', os.sep)
            net_result_path = os.path.join(path, 'net_output.pickle')
            if os.path.isfile(net_result_path):
                with open(net_result_path, 'rb') as handle:
                    airimu_results[f'airimu_{i}'] = CPU_Unpickler(handle).load()
                print(f"Loaded AirIMU model {i} from: {net_result_path}")
            else:
                raise Exception(f"Unable to load AirIMU result: {net_result_path}")
    
    # 加载AirIO模型输出
    airio_results = {}
    if args.airio_exp is not None:
        airio_paths = args.airio_exp.split(',')
        for i, path in enumerate(airio_paths):
            path = path.strip().replace('/', os.sep).replace('\\', os.sep)
            net_result_path = os.path.join(path, 'net_output.pickle')
            if os.path.isfile(net_result_path):
                with open(net_result_path, 'rb') as handle:
                    airio_results[f'airio_{i}'] = CPU_Unpickler(handle).load()
                print(f"Loaded AirIO model {i} from: {net_result_path}")
            else:
                raise Exception(f"Unable to load AirIO result: {net_result_path}")
    
    # 创建结果保存目录
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)
    
    # 存储所有结果用于比较
    all_results = {}
    
    # 处理每个数据集
    for data_conf in dataset_conf.data_list:
        print(f"\nProcessing dataset: {data_conf.name}")
        for data_name in data_conf.data_drive:
            print(f"  Sequence: {data_name}")
            
            # 加载AirIMU数据
            if len(airimu_results) > 0:
                airimu_key = list(airimu_results.keys())[0]
                airimu_state = airimu_results[airimu_key][data_name]
            else:
                airimu_state = None
            
            # 创建数据集
            dataset_inf = SeqInfDataset(
                data_conf.data_root, data_name, airimu_state, 
                device=args.device, name=data_conf.name,
                duration=1, step_size=1, drop_last=False, 
                conf=dataset_conf, usecov=True, useraw=False
            )
            
            # 获取初始值
            init = dataset_inf.get_init_value()
            gravity = dataset_inf.get_gravity()
            
            # 获取真值
            gt_pos_all = dataset_inf.data["gt_translation"][1:]
            gt_vel_all = dataset_inf.data["velocity"][1:]
            gt_rot_all = dataset_inf.data["gt_orientation"][1:]
            timestamps = dataset_inf.data["time"][1:]
            
            # 运行纯IMU基线
            if args.run_baseline:
                baseline_result, baseline_pos_err, baseline_vel_err = run_pure_imu_baseline(
                    dataset_inf, init, timestamps, gt_pos_all, gt_vel_all, gt_rot_all, args
                )
                all_results[f"{data_name}_PURE_IMU"] = {
                    'ekf_result': baseline_result,
                    'pos_error': baseline_pos_err,
                    'mean_pos_error': np.mean(baseline_pos_err),
                    'final_pos_error': baseline_pos_err[-1]
                }
                np.save(os.path.join(folder, f"{data_name}_PURE_IMU_ekf_result.npy"), baseline_result)
            
            # 模拟GPS信号
            gps_data = simulate_gps(
                gt_pos_all, 
                gps_freq=args.gps_freq,
                imu_freq=args.imu_freq,
                noise_std=args.gps_noise,
                bias=torch.tensor(args.gps_bias) if args.gps_bias else None
            )
            print(f"  GPS updates: {len(gps_data['indices'])} / {len(gt_pos_all)} IMU steps")
            
            # 加载AirIO速度估计
            airio_vel_data = None
            if len(airio_results) > 0:
                airio_key = list(airio_results.keys())[0]
                if data_name in airio_results[airio_key]:
                    airio_data = airio_results[airio_key][data_name]
                    airio_vel_data = {
                        'ts': airio_data['ts'][:, 0],
                        'vel': airio_data['net_vel'],
                        'cov': airio_data['cov'][0] if 'cov' in airio_data else None
                    }
                    print(f"  AirIO velocity estimates loaded")
            
            # 运行不同配置的EKF
            modes_to_run = args.mode.split(',') if ',' in args.mode else [args.mode]
            
            for mode in modes_to_run:
                mode = mode.strip()
                print(f"\n  Running EKF mode: {mode}")
                
                # 创建模型和运行器
                model = SingleIMUWithGPS().double()
                
                # 构建过程噪声Q对角线向量
                q_diag = torch.tensor([
                    args.gyro_noise, args.gyro_noise, args.gyro_noise,  # gyro measurement noise
                    args.acc_noise, args.acc_noise, args.acc_noise,     # acc measurement noise
                    args.gyro_bias_noise, args.gyro_bias_noise, args.gyro_bias_noise,  # gyro bias RW
                    args.acc_bias_noise, args.acc_bias_noise, args.acc_bias_noise     # acc bias RW
                ], dtype=torch.float64)
                
                # 构建初始状态协方差P对角线向量
                p_diag = torch.tensor([
                    args.init_rot_std, args.init_rot_std, args.init_rot_std,    # rotation
                    args.init_vel_std, args.init_vel_std, args.init_vel_std,  # velocity
                    args.init_pos_std, args.init_pos_std, args.init_pos_std,   # position
                    args.init_gyro_bias_std, args.init_gyro_bias_std, args.init_gyro_bias_std,  # gyro bias
                    args.init_acc_bias_std, args.init_acc_bias_std, args.init_acc_bias_std       # acc bias
                ], dtype=torch.float64)
                
                ekf = GPS_IMU_EKF_Runner(
                    model, 
                    mode=mode,
                    q_diag=q_diag,
                    r_velo=torch.ones(3, dtype=torch.float64) * args.r_velo,
                    r_pos=torch.ones(3, dtype=torch.float64) * args.r_pos,
                    p_diag=p_diag
                )
                
                # 状态初始化
                initial_state = torch.zeros(15, dtype=torch.float64)
                initial_state[:3] = init["rot"].Log()
                initial_state[3:6] = init["vel"]
                initial_state[6:9] = init["pos"]
                
                # 设置初始bias（如果指定）
                if args.init_gyro_bias is not None:
                    initial_state[9:12] = torch.tensor(args.init_gyro_bias, dtype=torch.float64)
                if args.init_acc_bias is not None:
                    initial_state[12:15] = torch.tensor(args.init_acc_bias, dtype=torch.float64)
                    
                ekf.state = initial_state
                
                print(f"    Initial state:")
                print(f"      Position: {init['pos'].numpy()}")
                print(f"      Velocity: {init['vel'].numpy()}")
                print(f"      Gyro Bias: {initial_state[9:12].numpy() if args.init_gyro_bias else 'zero'}")
                print(f"      Acc Bias: {initial_state[12:15].numpy() if args.init_acc_bias else 'zero'}")
                
                # 运行EKF
                gps_idx = 0
                airio_idx = 0

                # 是否使用AirIMU学习到的协方差
                use_imu_cov = (len(airimu_results) > 0)

                # 调试模式：限制处理帧数
                max_frames = args.max_frames if args.debug else len(dataset_inf)
                if args.debug:
                    print(f"    DEBUG MODE: Processing only first {max_frames} frames")

                t_range = tqdm.tqdm(range(min(max_frames, len(dataset_inf))), desc=f"  {mode}")
                for i in t_range:
                    data = dataset_inf[i]
                    
                    imu_data = {
                        "gyro": data["gyro"][0],
                        "acc": data["acc"][0],
                        "dt": data["dt"][0],
                        "gyro_cov": data["gyro_cov"][0] if data["gyro_cov"] is not None else torch.ones(3) * args.gyro_noise,
                        "acc_cov": data["acc_cov"][0] if data["acc_cov"] is not None else torch.ones(3) * args.acc_noise
                    }
                    
                    # 检查是否有GPS观测
                    gps_obs = None
                    gps_cov = None
                    if gps_idx < len(gps_data['indices']):
                        if i == gps_data['indices'][gps_idx]:
                            gps_obs = gps_data['pos'][gps_idx]
                            gps_cov = gps_data['cov'][gps_idx]
                            gps_idx += 1
                    
                    # 检查是否有AirIO观测
                    airio_obs = None
                    airio_cov = None
                    if airio_vel_data is not None and airio_idx < len(airio_vel_data['ts']):
                        if abs(airio_vel_data['ts'][airio_idx] - timestamps[i]) < 0.001:
                            # 将世界坐标系速度转换为IMU坐标系
                            vel_world = airio_vel_data['vel'][airio_idx]
                            rot = data["gt_rot"][0]
                            airio_obs = rot.Inv() @ vel_world
                            if airio_vel_data['cov'] is not None:
                                airio_cov = airio_vel_data['cov'][airio_idx]
                            airio_idx += 1
                    
                    # 运行EKF一步（使用AirIMU学习到的协方差）
                    ekf.run(imu_data, gps_obs=gps_obs, airio_obs=airio_obs,
                           gps_cov=gps_cov, airio_cov=airio_cov,
                           use_imu_cov=use_imu_cov,
                           input_weight=args.input_weight,
                           bias_weight=args.bias_weight,
                           obs_weight=args.obs_weight)
                
                # 获取结果
                ekf_result, ekf_cov = ekf.get_result()
                ekf_result = ekf_result.numpy()
                
                print(f"    GPS updates used: {ekf.gps_update_count}")
                print(f"    AirIO updates used: {ekf.airio_update_count}")
                
                # 计算误差
                gtpos = gt_pos_all[:len(ekf_result)].numpy()
                gtvel = gt_vel_all[:len(ekf_result)].numpy()
                gtrot = gt_rot_all[:len(ekf_result)].numpy()
                
                est_pos = ekf_result[:, 6:9]
                est_rot = pp.so3(ekf_result[:, :3]).Exp().numpy()
                
                # 计算评估指标
                ate_metrics = compute_ate(est_pos, gtpos)
                aoe_metrics = compute_aoe(est_rot, gtrot)
                rte_metrics = compute_rte(est_pos, gtpos, window_size=args.rte_window)
                roe_metrics = compute_roe(est_rot, gtrot, window_size=args.rte_window)
                
                pos_error = ate_metrics['errors']
                vel_error = np.linalg.norm(gtvel - ekf_result[:, 3:6], axis=1)
                
                print(f"    ===== Evaluation Metrics =====")
                print(f"    ATE: {ate_metrics['ate']:.4f} m (mean: {ate_metrics['mean']:.4f}, std: {ate_metrics['std']:.4f}, max: {ate_metrics['max']:.4f})")
                print(f"    AOE: {np.degrees(aoe_metrics['aoe']):.4f} deg (mean: {np.degrees(aoe_metrics['mean']):.4f}, std: {np.degrees(aoe_metrics['std']):.4f}, max: {np.degrees(aoe_metrics['max']):.4f})")
                print(f"    RTE ({args.rte_window} frames): {rte_metrics['rte']:.4f} m (mean: {rte_metrics['mean']:.4f}, std: {rte_metrics['std']:.4f}, max: {rte_metrics['max']:.4f})")
                print(f"    ROE ({args.rte_window} frames): {np.degrees(roe_metrics['roe']):.4f} deg (mean: {np.degrees(roe_metrics['mean']):.4f}, std: {np.degrees(roe_metrics['std']):.4f}, max: {np.degrees(roe_metrics['max']):.4f})")
                print(f"    Mean velocity error: {np.mean(vel_error):.4f} m/s")
                print(f"    Final position error: {pos_error[-1]:.4f} m")
                
                # 保存结果
                result_key = f"{data_name}_{mode}"
                all_results[result_key] = {
                    'ekf_result': ekf_result,
                    'pos_error': pos_error,
                    'vel_error': vel_error,
                    'mean_pos_error': np.mean(pos_error),
                    'final_pos_error': pos_error[-1],
                    'gps_count': ekf.gps_update_count,
                    'airio_count': ekf.airio_update_count,
                    # 评估指标
                    'ate': ate_metrics['ate'],
                    'ate_mean': ate_metrics['mean'],
                    'ate_std': ate_metrics['std'],
                    'ate_max': ate_metrics['max'],
                    'aoe': aoe_metrics['aoe'],
                    'aoe_deg': np.degrees(aoe_metrics['aoe']),
                    'rte': rte_metrics['rte'],
                    'rte_mean': rte_metrics['mean'],
                    'rte_std': rte_metrics['std'],
                    'roe': roe_metrics['roe'],
                    'roe_deg': np.degrees(roe_metrics['roe'])
                }
                
                # 保存npy文件
                np.save(os.path.join(folder, f"{result_key}_ekf_result.npy"), ekf_result)
                
                # 保存为TUM格式轨迹
                ekf_poses = np.zeros((ekf_result.shape[0], 8))
                ekf_poses[:, 0] = timestamps[:len(ekf_result)].numpy()
                ekf_poses[:, 1:4] = ekf_result[:, 6:9]
                ekf_poses[:, 4:] = pp.so3(ekf_result[:, :3]).Exp().numpy()
                np.save(os.path.join(folder, f"{result_key}_ekf_poses.npy"), ekf_poses)
                
                # 绘制轨迹对比图
                plt.figure(figsize=(10, 8))
                plt.plot(ekf_result[:, 6], ekf_result[:, 7], label=f"EKF {mode}", linewidth=2)
                plt.plot(gtpos[:, 0], gtpos[:, 1], label="GT", linewidth=2, linestyle='--')
                
                # 如果有基线，也绘制
                if args.run_baseline and f"{data_name}_PURE_IMU" in all_results:
                    baseline_res = all_results[f"{data_name}_PURE_IMU"]['ekf_result']
                    plt.plot(baseline_res[:, 6], baseline_res[:, 7], label="Pure IMU", 
                            linewidth=1, alpha=0.7, linestyle=':')
                
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title(f'Trajectory Comparison - {data_name} - {mode}')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.savefig(os.path.join(folder, f"{result_key}_trajectory.png"), dpi=150)
                plt.close()
                
                # 绘制位置误差图
                plt.figure(figsize=(12, 4))
                plt.plot(pos_error, label=f"{mode}")
                if args.run_baseline and f"{data_name}_PURE_IMU" in all_results:
                    plt.plot(all_results[f"{data_name}_PURE_IMU"]['pos_error'], 
                            label="Pure IMU", alpha=0.7, linestyle=':')
                plt.xlabel('Frame')
                plt.ylabel('Position Error (m)')
                plt.title(f'Position Error - {data_name} - {mode}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(folder, f"{result_key}_pos_error.png"), dpi=150)
                plt.close()
                
                # 绘制bias估计
                plot_bias_subplots(ekf_result[:, 9:12], 
                                 title=f"Gyro Bias - {data_name} - {mode}", 
                                 save_path=os.path.join(folder, f"{result_key}_gyro_bias.png"))
                plot_bias_subplots(ekf_result[:, 12:15], 
                                 title=f"Acc Bias - {data_name} - {mode}", 
                                 save_path=os.path.join(folder, f"{result_key}_acc_bias.png"))
                
                # 绘制三维轨迹
                plot_3d_trajectory(est_pos, gtpos, 
                                 save_path=os.path.join(folder, f"{result_key}_3d_trajectory.png"),
                                 title=f"3D Trajectory - {data_name} - {mode}\nATE: {ate_metrics['ate']:.3f}m, AOE: {np.degrees(aoe_metrics['aoe']):.3f}°")
            
            # 绘制所有模式的对比图
            if len(modes_to_run) > 1 or args.run_baseline:
                plt.figure(figsize=(12, 5))
                
                # 轨迹对比
                plt.subplot(1, 2, 1)
                for mode in modes_to_run:
                    mode = mode.strip()
                    result_key = f"{data_name}_{mode}"
                    if result_key in all_results:
                        res = all_results[result_key]['ekf_result']
                        plt.plot(res[:, 6], res[:, 7], label=mode, linewidth=1.5)
                if args.run_baseline and f"{data_name}_PURE_IMU" in all_results:
                    baseline_res = all_results[f"{data_name}_PURE_IMU"]['ekf_result']
                    plt.plot(baseline_res[:, 6], baseline_res[:, 7], 
                            label="Pure IMU", linewidth=1, linestyle=':', alpha=0.7)
                plt.plot(gtpos[:, 0], gtpos[:, 1], label="GT", linewidth=2, linestyle='--', color='black')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title('Trajectory Comparison')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                
                # 误差对比
                plt.subplot(1, 2, 2)
                for mode in modes_to_run:
                    mode = mode.strip()
                    result_key = f"{data_name}_{mode}"
                    if result_key in all_results:
                        err = all_results[result_key]['pos_error']
                        plt.plot(err, label=mode, linewidth=1.5)
                if args.run_baseline and f"{data_name}_PURE_IMU" in all_results:
                    baseline_err = all_results[f"{data_name}_PURE_IMU"]['pos_error']
                    plt.plot(baseline_err, label="Pure IMU", linewidth=1, linestyle=':', alpha=0.7)
                plt.xlabel('Frame')
                plt.ylabel('Position Error (m)')
                plt.title('Position Error Comparison')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(folder, f"{data_name}_comparison.png"), dpi=150)
                plt.close()
    
    # 保存总结报告
    with open(os.path.join(folder, 'summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("GPS+IMU EKF Fusion Experiment Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"GPS Frequency: {args.gps_freq} Hz\n")
        f.write(f"GPS Noise: {args.gps_noise} m\n")
        f.write(f"IMU Frequency: {args.imu_freq} Hz\n")
        f.write(f"RTE/ROE Window: {args.rte_window} frames\n\n")
        
        for key, result in all_results.items():
            f.write(f"\n{key}:\n")
            f.write(f"  Mean Position Error: {result['mean_pos_error']:.4f} m\n")
            f.write(f"  Final Position Error: {result['final_pos_error']:.4f} m\n")
            if 'gps_count' in result:
                f.write(f"  GPS Updates: {result['gps_count']}\n")
            if 'airio_count' in result:
                f.write(f"  AirIO Updates: {result['airio_count']}\n")
            # 添加评估指标
            if 'ate' in result:
                f.write(f"  ATE: {result['ate']:.4f} m\n")
                f.write(f"  ATE (mean/std/max): {result['ate_mean']:.4f} / {result['ate_std']:.4f} / {result['ate_max']:.4f} m\n")
            if 'aoe_deg' in result:
                f.write(f"  AOE: {result['aoe_deg']:.4f} deg\n")
            if 'rte' in result:
                f.write(f"  RTE ({args.rte_window} frames): {result['rte']:.4f} m\n")
                f.write(f"  RTE (mean/std/max): {result['rte_mean']:.4f} / {result['rte_std']:.4f} m\n")
            if 'roe_deg' in result:
                f.write(f"  ROE ({args.rte_window} frames): {result['roe_deg']:.4f} deg\n")
    
    # 保存JSON格式的详细结果
    json_results = {}
    for key, result in all_results.items():
        json_results[key] = {
            'mean_pos_error': float(result['mean_pos_error']),
            'final_pos_error': float(result['final_pos_error']),
            'gps_count': int(result.get('gps_count', 0)),
            'airio_count': int(result.get('airio_count', 0)),
            'ate': float(result.get('ate', 0)),
            'ate_mean': float(result.get('ate_mean', 0)),
            'ate_std': float(result.get('ate_std', 0)),
            'ate_max': float(result.get('ate_max', 0)),
            'aoe_deg': float(result.get('aoe_deg', 0)),
            'rte': float(result.get('rte', 0)),
            'rte_mean': float(result.get('rte_mean', 0)),
            'rte_std': float(result.get('rte_std', 0)),
            'roe_deg': float(result.get('roe_deg', 0))
        }
    
    with open(os.path.join(folder, 'results.json'), 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print("\n" + "="*60)
    print("Experiment completed! Results saved to:", folder)
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPS + AirIMU + AirIO EKF Fusion')
    
    # 设备配置
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    
    # 数据配置
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", 
                       help="Dataset configuration file")
    
    # AirIMU模型输出路径 (支持多个，用逗号分隔)
    parser.add_argument("--airimu_exp", type=str, 
                       default="experiments/euroc/airimu_code",
                       help="Path(s) to AirIMU net_output.pickle (comma separated for multiple)")
    
    # AirIO模型输出路径 (支持多个，用逗号分隔)
    parser.add_argument("--airio_exp", type=str,
                       default="experiments/euroc/motion_body_rot",
                       help="Path(s) to AirIO net_output.pickle (comma separated for multiple)")
    
    # 融合模式
    parser.add_argument("--mode", type=str, default="FULL_FUSION",
                       choices=['GPS_ONLY', 'AIRIO_ONLY', 'GPS_AIRIO', 'GPS_AIRIMU', 
                               'FULL_FUSION', 'ALL'],
                       help="Fusion mode. Use comma to run multiple modes (e.g., 'GPS_ONLY,FULL_FUSION')")
    
    # GPS模拟参数
    parser.add_argument("--gps_freq", type=int, default=1, help="GPS frequency (Hz)")
    parser.add_argument("--imu_freq", type=int, default=200, help="IMU frequency (Hz)")
    parser.add_argument("--gps_noise", type=float, default=0.1, help="GPS noise std (m)")
    parser.add_argument("--gps_bias", type=float, nargs=3, default=None, 
                       help="GPS bias [x, y, z] (m)")
    
    # 评估指标参数
    parser.add_argument("--rte_window", type=int, default=100, 
                       help="Window size for RTE/ROE calculation (frames)")
    
    # 快速测试模式
    parser.add_argument("--debug", action="store_true", 
                       help="Debug mode: only process first 1000 frames for quick testing")
    parser.add_argument("--max_frames", type=int, default=1000, 
                       help="Maximum number of frames to process in debug mode (default: 1000)")
    
    # EKF噪声参数
    parser.add_argument("--q_init", type=float, default=0.01, help="Initial process noise")
    parser.add_argument("--r_velo", type=float, default=0.1, help="Velocity observation noise")
    parser.add_argument("--r_pos", type=float, default=0.1, help="Position observation noise (should match gps_noise)")
    parser.add_argument("--p_init", type=float, default=0.1, help="Initial state covariance")
    
    # IMU噪声参数（过程噪声Q的各分量）
    parser.add_argument("--gyro_noise", type=float, default=0.001, 
                       help="Gyroscope measurement noise (rad/s)")
    parser.add_argument("--acc_noise", type=float, default=0.01, 
                       help="Accelerometer measurement noise (m/s^2)")
    parser.add_argument("--gyro_bias_noise", type=float, default=1e-6, 
                       help="Gyroscope bias random walk (rad/s^2)")
    parser.add_argument("--acc_bias_noise", type=float, default=1e-5, 
                       help="Accelerometer bias random walk (m/s^3)")
    
    # 初始状态协方差（P矩阵对角线各分量）
    parser.add_argument("--init_rot_std", type=float, default=0.01, 
                       help="Initial rotation uncertainty (rad), reduced because GPS doesn't observe rotation")
    parser.add_argument("--init_vel_std", type=float, default=0.1, 
                       help="Initial velocity uncertainty (m/s)")
    parser.add_argument("--init_pos_std", type=float, default=0.1, 
                       help="Initial position uncertainty (m)")
    parser.add_argument("--init_gyro_bias_std", type=float, default=0.01, 
                       help="Initial gyroscope bias uncertainty (rad/s)")
    parser.add_argument("--init_acc_bias_std", type=float, default=0.1, 
                       help="Initial accelerometer bias uncertainty (m/s^2)")
    
    # 初始bias设置
    parser.add_argument("--init_gyro_bias", type=float, nargs=3, default=None,
                       help="Initial gyroscope bias [x, y, z] (rad/s)")
    parser.add_argument("--init_acc_bias", type=float, nargs=3, default=None,
                       help="Initial accelerometer bias [x, y, z] (m/s^2)")
    
    # AirIMU/AirIO不确定性权重 (参考IMUofflinerunner.py)
    parser.add_argument("--input_weight", type=float, default=1e2,
                       help="Weight for AirIMU acceleration covariance (default: 100)")
    parser.add_argument("--obs_weight", type=float, default=1e-1,
                       help="Weight for AirIO velocity covariance (default: 0.1)")
    parser.add_argument("--bias_weight", type=float, default=1e-12,
                       help="Bias random walk noise (default: 1e-12)")
    
    # 输出配置
    parser.add_argument("--savedir", type=str, default="./EKFresult/gps_imu_fusion",
                       help="Directory to save results")
    
    # 运行基线
    parser.add_argument("--run_baseline", action="store_true", default=True,
                       help="Run pure IMU baseline for comparison")
    
    args = parser.parse_args()
    
    # 如果模式是ALL，运行所有模式
    if args.mode == 'ALL':
        args.mode = 'GPS_ONLY,AIRIO_ONLY,GPS_AIRIO,FULL_FUSION'
    
    print("\nArguments:")
    print(args)
    print()
    
    run_experiment(args)
