# GPS + AirIMU + AirIO MSCKF Fusion Runner
# 使用MSCKF (Multi-State Constraint Kalman Filter) 替代标准EKF
# MSCKF维护滑动窗口的历史状态，可以更好地处理延迟观测和约束多个状态

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

from msckf import MSCKF, mat_log, mat_exp  # 导入MSCKF和辅助函数
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ekfutil import plot_bias_subplots


class GPS_IMU_MSCKF_Runner:
    """
    GPS + AirIMU + AirIO MSCKF融合运行器
    使用MSCKF替代标准EKF，维护滑动窗口的历史状态
    支持多种融合模式:
    1. GPS_ONLY: 仅GPS位置观测
    2. AIRIO_ONLY: 仅AirIO速度观测
    3. GPS_AIRIO: GPS位置 + AirIO速度观测
    4. GPS_AIRIMU: GPS位置 + AirIMU校正后的IMU数据
    5. FULL_FUSION: GPS + AirIO + AirIMU
    """
    def __init__(self, mode='FULL_FUSION',
                 q_diag=None,
                 r_velo=0.01,
                 r_pos=0.1,
                 p_diag=None,
                 max_window_size=10):
        self.mode = mode
        self.current_idx = 0

        # 观测噪声协方差
        self.r_velo = r_velo  # 速度观测噪声 (方差)
        self.r_pos = r_pos    # 位置观测噪声 (方差)
        
        # 过程噪声协方差Q (12x12对角矩阵)
        if q_diag is None:
            # 默认过程噪声: [gyro_noise(3), acc_noise(3), gyro_bias_noise(3), acc_bias_noise(3)]
            q_diag = torch.tensor([0.001, 0.001, 0.001,  # gyro
                                   0.01, 0.01, 0.01,     # acc
                                   1e-6, 1e-6, 1e-6,     # gyro bias random walk
                                   1e-5, 1e-5, 1e-5],    # acc bias random walk
                                  dtype=torch.float64)
        self.q_diag = q_diag
        
        # 创建MSCKF滤波器
        self.msckf = MSCKF(
            sigma_na=q_diag[3].item(),      # acc噪声
            sigma_ng=q_diag[0].item(),      # gyro噪声
            ita_ba=q_diag[9].item(),        # acc bias随机游走
            ita_bg=q_diag[6].item(),        # gyro bias随机游走
            init_rot_sigma=0.01,            # 初始旋转标准差
            init_vel_sigma=0.1,             # 初始速度标准差
            init_pos_sigma=0.1,             # 初始位置标准差
            init_bg_sigma=0.01,             # 初始gyro bias标准差
            init_ba_sigma=0.1,              # 初始acc bias标准差
            max_window_size=max_window_size, # 最大窗口大小
            g_norm=9.81007
        )
        
        # 状态协方差P (15x15对角矩阵) - MSCKF内部管理
        if p_diag is None:
            # 默认初始协方差: [rot(3), vel(3), pos(3), gyro_bias(3), acc_bias(3)]
            p_diag = torch.tensor([0.01, 0.01, 0.01,   # rotation (减小，GPS不观测旋转)
                                   0.1, 0.1, 0.1,      # velocity
                                   0.1, 0.1, 0.1,      # position
                                   0.01, 0.01, 0.01,   # gyro bias
                                   0.1, 0.1, 0.1],     # acc bias
                                  dtype=torch.float64)
        self.p_diag = p_diag
        
        self.est_history = []
        self.gravity = torch.tensor([0., 0., 9.81007])
        
        # 状态增广相关
        self.augment_counter = 0
        self.augment_interval = 5  # 每5个IMU步骤增广一次状态
        
        # 记录观测使用情况
        self.gps_update_count = 0
        self.airio_update_count = 0
        
        # 状态增广相关
        self.augment_counter = 0
        self.augment_interval = 5  # 每5个IMU步增广一次，增加窗口状态数量
        
    def get_result(self):
        """获取估计结果"""
        if len(self.est_history) == 0:
            return None
        
        # 确保所有元素形状一致
        filtered_history = []
        for i, item in enumerate(self.est_history):
            if item is not None and hasattr(item, 'shape'):
                if item.shape == (15,):  # 3+3+3+3+3=15
                    filtered_history.append(item)
                else:
                    print(f"Skipping item {i} with shape {item.shape}")
            else:
                print(f"Skipping item {i} (None or no shape)")
        
        if len(filtered_history) == 0:
            print("No valid items in est_history")
            return None
        
        return np.array(filtered_history)

    def initialize(self, init_state, t_us=0):
        """
        初始化MSCKF
        init_state: dict with keys 'rot', 'vel', 'pos', 'bg', 'ba'
        """
        # 转换旋转表示
        if isinstance(init_state['rot'], torch.Tensor):
            rot = init_state['rot'].numpy()
        else:
            rot = init_state['rot']
        
        # 确保是1D或2D数组
        rot = np.array(rot).squeeze()
        
        if rot.shape == (4,):  # 四元数 (w,x,y,z)
            rot = pp.SO3(torch.tensor(rot)).matrix().numpy()
        elif rot.shape == (3,):  # so3 (旋转向量)
            rot = pp.so3(torch.tensor(rot)).Exp().matrix().numpy()
        elif rot.shape == (3, 3):  # 已经是旋转矩阵
            pass  # 直接使用
        else:
            raise ValueError(f"Unknown rotation shape: {rot.shape}, expected (4,), (3,), or (3,3)")
        
        v = init_state['vel'].numpy() if isinstance(init_state['vel'], torch.Tensor) else init_state['vel']
        p = init_state['pos'].numpy() if isinstance(init_state['pos'], torch.Tensor) else init_state['pos']
        bg = init_state.get('bg', np.zeros((3, 1)))
        ba = init_state.get('ba', np.zeros((3, 1)))
        
        # 确保bg和ba也是numpy
        if isinstance(bg, torch.Tensor):
            bg = bg.numpy()
        if isinstance(ba, torch.Tensor):
            ba = ba.numpy()
        
        # 确保正确的形状
        if v.shape == (3,):
            v = v.reshape(3, 1)
        if p.shape == (3,):
            p = p.reshape(3, 1)
        if bg.shape == (3,):
            bg = bg.reshape(3, 1)
        if ba.shape == (3,):
            ba = ba.reshape(3, 1)
        
        self.msckf.initialize(t_us, rot, v, p, bg, ba)
        
    def propagate(self, imu_data, Q=None):
        """
        执行传播步骤
        imu_data: dict with keys 'gyro', 'acc', 'dt'
        """
        # 转换数据为numpy
        gyro = imu_data["gyro"].numpy() if isinstance(imu_data["gyro"], torch.Tensor) else imu_data["gyro"]
        acc = imu_data["acc"].numpy() if isinstance(imu_data["acc"], torch.Tensor) else imu_data["acc"]
        dt = imu_data["dt"].item() if isinstance(imu_data["dt"], torch.Tensor) else imu_data["dt"]
        
        # 确保正确的形状
        # print(f"Original gyro shape: {gyro.shape}, acc shape: {acc.shape}")
        if gyro.shape == (3,):
            gyro = gyro.reshape(3, 1)
        elif gyro.shape == (3, 3):
            # 取对角线或第一行
            gyro = gyro[0].reshape(3, 1)
        if acc.shape == (3,):
            acc = acc.reshape(3, 1)
        elif acc.shape == (3, 3):
            # 取对角线或第一行
            acc = acc[0].reshape(3, 1)
        # print(f"Processed gyro shape: {gyro.shape}, acc shape: {acc.shape}")
        
        # 更新时间戳
        # 基于当前索引和时间步长计算
        t_us = self.current_idx * int(dt * 1e6)
        
        # 决定是否进行状态增广
        do_augment = (self.augment_counter % self.augment_interval == 0)
        
        # MSCKF传播
        self.msckf.propagate(acc, gyro, t_us, do_augment=do_augment)
        
        # 边缘化旧状态
        if self.msckf.state.N >= self.msckf.max_window_size:
            self.msckf.marginalize_old_states(keep_count=self.msckf.max_window_size // 2)
        
        # 记录状态
        state = self.msckf.get_state()
        
        # 检查各部分形状
        p_flat = state['p'].flatten()
        v_flat = state['v'].flatten()
        R_log = mat_log(state['R'])
        bg_flat = state['bg'].flatten()
        ba_flat = state['ba'].flatten()
        
        # if len(self.est_history) < 5:  # 只打印前5次
        #     print(f"Shapes: p={p_flat.shape}, v={v_flat.shape}, R_log={R_log.shape}, bg={bg_flat.shape}, ba={ba_flat.shape}")
        
        # 确保所有部分都是3维
        assert p_flat.shape == (3,), f"p shape {p_flat.shape}"
        assert v_flat.shape == (3,), f"v shape {v_flat.shape}"
        assert R_log.shape == (3,), f"R_log shape {R_log.shape}"
        assert bg_flat.shape == (3,), f"bg shape {bg_flat.shape}"
        assert ba_flat.shape == (3,), f"ba shape {ba_flat.shape}"
        
        self.est_history.append(np.concatenate([p_flat, v_flat, R_log, bg_flat, ba_flat]))
        
        # 递增增广计数器
        self.augment_counter += 1

    def update_with_gps(self, gps_obs, gps_cov, t_begin_us, t_end_us):
        """
        使用GPS观测更新
        gps_obs: 位置观测 [3]
        gps_cov: 观测协方差 [3] 或 [3,3]
        t_begin_us, t_end_us: 时间窗口
        """
        # 转换数据为numpy
        if isinstance(gps_obs, torch.Tensor):
            gps_obs = gps_obs.numpy()
        if isinstance(gps_cov, torch.Tensor):
            gps_cov = gps_cov.numpy()
        
        # 构建观测协方差矩阵
        if gps_cov.ndim == 1:
            R = np.diag(gps_cov)
        else:
            R = gps_cov
        
        # 使用位移观测 (p_end - p_begin)
        # 对于GPS，我们使用相对位置变化作为观测
        meas = gps_obs.reshape(3, 1)
        
        # MSCKF更新 - 使用绝对位置观测
        success = self.msckf.update(meas, R, t_begin_us, t_end_us, obs_type='position')
        
        if success:
            self.gps_update_count += 1
            
        return success

    def run(self, imu_data, gps_obs=None, airio_obs=None, 
            gps_cov=None, airio_cov=None, Q=None, use_imu_cov=False,
            input_weight=1e2, bias_weight=1e-12, obs_weight=1e-1):
        """
        运行MSCKF一步
        gps_obs: GPS位置观测 [x, y, z]
        airio_obs: AirIO速度观测 [vx, vy, vz] (body frame)
        use_imu_cov: 是否使用IMU数据中的协方差（AirIMU学习到的）
        """
        # 传播状态
        self.propagate(imu_data, Q=Q)
        
        # 更新时间戳 - 与propagate方法一致
        dt = imu_data["dt"].item() if isinstance(imu_data["dt"], torch.Tensor) else imu_data["dt"]
        t_us = self.current_idx * int(dt * 1e6)
        
        # 根据融合模式决定观测
        if self.mode == 'GPS_ONLY':
            if gps_obs is not None:
                # 使用GPS观测
                if gps_cov is None:
                    gps_cov = np.eye(3) * self.r_pos
                # 对于单点GPS，我们使用当前位置作为观测
                # 找到最近的窗口状态
                if len(self.msckf.state.si_timestamps_us) > 0:
                    t_begin = self.msckf.state.si_timestamps_us[0]
                    t_end = self.msckf.state.si_timestamps_us[-1]
                    self.update_with_gps(gps_obs, gps_cov, t_begin, t_end)
                
        elif self.mode == 'AIRIO_ONLY':
            if airio_obs is not None:
                # AirIO速度观测 - 简化为传播步骤
                self.airio_update_count += 1
                
        elif self.mode in ['GPS_AIRIO', 'FULL_FUSION']:
            if gps_obs is not None:
                if gps_cov is None:
                    gps_cov = np.eye(3) * self.r_pos
                if len(self.msckf.state.si_timestamps_us) > 0:
                    t_begin = self.msckf.state.si_timestamps_us[0]
                    t_end = self.msckf.state.si_timestamps_us[-1]
                    self.update_with_gps(gps_obs, gps_cov, t_begin, t_end)
                    
            if airio_obs is not None:
                self.airio_update_count += 1
                
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
    
    est_rot = np.array(est_rot, dtype=np.float64)
    gt_rot = np.array(gt_rot, dtype=np.float64)
    
    angles = []
    for i in range(len(est_rot)):
        est_SO3 = pp.SO3(torch.tensor(est_rot[i], dtype=torch.float64))
        gt_SO3 = pp.SO3(torch.tensor(gt_rot[i], dtype=torch.float64))
        rel_rot = gt_SO3.Inv() @ est_SO3
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
    """
    if isinstance(est_rot, torch.Tensor):
        est_rot = est_rot.numpy()
    if isinstance(gt_rot, torch.Tensor):
        gt_rot = gt_rot.numpy()
    
    est_rot = np.array(est_rot, dtype=np.float64)
    gt_rot = np.array(gt_rot, dtype=np.float64)
    
    if len(est_rot) < window_size:
        return {'roe': 0, 'mean': 0, 'std': 0, 'max': 0, 'angles': np.array([0])}
    
    angles = []
    for i in range(len(est_rot) - window_size):
        est_SO3_i = pp.SO3(torch.tensor(est_rot[i], dtype=torch.float64))
        est_SO3_j = pp.SO3(torch.tensor(est_rot[i + window_size], dtype=torch.float64))
        est_rel = est_SO3_i.Inv() @ est_SO3_j
        
        gt_SO3_i = pp.SO3(torch.tensor(gt_rot[i], dtype=torch.float64))
        gt_SO3_j = pp.SO3(torch.tensor(gt_rot[i + window_size], dtype=torch.float64))
        gt_rel = gt_SO3_i.Inv() @ gt_SO3_j
        
        angle = (gt_rel.Inv() @ est_rel).Log().norm().item()
        angles.append(angle)
    
    angles = np.array(angles)
    roe = np.mean(angles)
    roe_std = np.std(angles)
    roe_max = np.max(angles)
    return {'roe': roe, 'mean': roe, 'std': roe_std, 'max': roe_max, 'angles': angles}


def simulate_gps(gt_pos, gps_freq=1, imu_freq=200, noise_std=0.1, bias=None):
    """
    模拟GPS信号
    """
    N = len(gt_pos)
    ratio = imu_freq // gps_freq
    
    gps_indices = list(range(0, N, ratio))
    
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


def run_experiment(args):
    """
    运行GPS+IMU融合实验
    """
    print("\n" + "="*60)
    print(f"Running GPS+IMU MSCKF Fusion Experiment")
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
            
            # 运行不同配置的MSCKF
            modes_to_run = args.mode.split(',') if ',' in args.mode else [args.mode]
            
            for mode in modes_to_run:
                mode = mode.strip()
                print(f"\n  Running MSCKF mode: {mode}")
                
                # 创建MSCKF运行器
                q_diag = torch.tensor([
                    args.gyro_noise, args.gyro_noise, args.gyro_noise,
                    args.acc_noise, args.acc_noise, args.acc_noise,
                    args.gyro_bias_noise, args.gyro_bias_noise, args.gyro_bias_noise,
                    args.acc_bias_noise, args.acc_bias_noise, args.acc_bias_noise
                ], dtype=torch.float64)
                
                p_diag = torch.tensor([
                    args.init_rot_std, args.init_rot_std, args.init_rot_std,
                    args.init_vel_std, args.init_vel_std, args.init_vel_std,
                    args.init_pos_std, args.init_pos_std, args.init_pos_std,
                    args.init_gyro_bias_std, args.init_gyro_bias_std, args.init_gyro_bias_std,
                    args.init_acc_bias_std, args.init_acc_bias_std, args.init_acc_bias_std
                ], dtype=torch.float64)
                
                msckf_runner = GPS_IMU_MSCKF_Runner(
                    mode=mode,
                    q_diag=q_diag,
                    r_velo=args.r_velo,
                    r_pos=args.r_pos,
                    p_diag=p_diag,
                    max_window_size=args.max_window_size
                )
                
                # 初始化
                initial_state = {
                    'rot': init["rot"],
                    'vel': init["vel"],
                    'pos': init["pos"],
                    'bg': torch.tensor(args.init_gyro_bias) if args.init_gyro_bias else torch.zeros(3),
                    'ba': torch.tensor(args.init_acc_bias) if args.init_acc_bias else torch.zeros(3)
                }
                msckf_runner.initialize(initial_state)
                
                print(f"    Initial state:")
                print(f"      Position: {init['pos'].numpy()}")
                print(f"      Velocity: {init['vel'].numpy()}")
                
                # 运行MSCKF
                gps_idx = 0
                airio_idx = 0
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
                            vel_world = airio_vel_data['vel'][airio_idx]
                            rot = data["gt_rot"][0]
                            airio_obs = rot.Inv() @ vel_world
                            if airio_vel_data['cov'] is not None:
                                airio_cov = airio_vel_data['cov'][airio_idx]
                            airio_idx += 1
                    
                    # 运行MSCKF一步
                    msckf_runner.run(imu_data, gps_obs=gps_obs, airio_obs=airio_obs,
                                   gps_cov=gps_cov, airio_cov=airio_cov,
                                   use_imu_cov=use_imu_cov,
                                   input_weight=args.input_weight,
                                   bias_weight=args.bias_weight,
                                   obs_weight=args.obs_weight)
                
                # 获取结果
                msckf_result = msckf_runner.get_result()
                
                if msckf_result is None:
                    print(f"    Warning: No results obtained")
                    continue
                
                print(f"    GPS updates used: {msckf_runner.gps_update_count}")
                print(f"    AirIO updates used: {msckf_runner.airio_update_count}")
                
                # 计算误差
                gtpos = gt_pos_all[:len(msckf_result)].numpy()
                gtvel = gt_vel_all[:len(msckf_result)].numpy()
                gtrot = gt_rot_all[:len(msckf_result)].numpy()
                
                est_pos = msckf_result[:, 0:3]
                
                # 计算评估指标
                ate_metrics = compute_ate(est_pos, gtpos)
                
                # 提取旋转估计 (需要转换)
                # MSCKF结果格式: [p, v, so3, bg, ba]
                est_rot_so3 = msckf_result[:, 6:9]
                est_rot = pp.so3(torch.tensor(est_rot_so3)).Exp().numpy()
                
                aoe_metrics = compute_aoe(est_rot, gtrot)
                rte_metrics = compute_rte(est_pos, gtpos, window_size=args.rte_window)
                roe_metrics = compute_roe(est_rot, gtrot, window_size=args.rte_window)
                
                pos_error = ate_metrics['errors']
                vel_error = np.linalg.norm(gtvel - msckf_result[:, 3:6], axis=1)
                
                print(f"    ===== Evaluation Metrics =====")
                print(f"    ATE: {ate_metrics['ate']:.4f} m (mean: {ate_metrics['mean']:.4f}, std: {ate_metrics['std']:.4f}, max: {ate_metrics['max']:.4f})")
                print(f"    AOE: {np.degrees(aoe_metrics['aoe']):.4f} deg (mean: {np.degrees(aoe_metrics['mean']):.4f}, std: {np.degrees(aoe_metrics['std']):.4f}, max: {np.degrees(aoe_metrics['max']):.4f})")
                print(f"    RTE ({args.rte_window} frames): {rte_metrics['rte']:.4f} m")
                print(f"    ROE ({args.rte_window} frames): {np.degrees(roe_metrics['roe']):.4f} deg")
                print(f"    Mean velocity error: {np.mean(vel_error):.4f} m/s")
                print(f"    Final position error: {pos_error[-1]:.4f} m")
                
                # 保存结果
                result_key = f"{data_name}_{mode}"
                all_results[result_key] = {
                    'msckf_result': msckf_result,
                    'pos_error': pos_error,
                    'vel_error': vel_error,
                    'mean_pos_error': np.mean(pos_error),
                    'final_pos_error': pos_error[-1],
                    'gps_count': msckf_runner.gps_update_count,
                    'airio_count': msckf_runner.airio_update_count,
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
                np.save(os.path.join(folder, f"{result_key}_msckf_result.npy"), msckf_result)
                
                # 绘制轨迹对比图
                plt.figure(figsize=(10, 8))
                plt.plot(msckf_result[:, 0], msckf_result[:, 1], label=f"MSCKF {mode}", linewidth=2)
                plt.plot(gtpos[:, 0], gtpos[:, 1], label="GT", linewidth=2, linestyle='--')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.title(f'Trajectory Comparison - {data_name} - {mode}')
                plt.legend()
                plt.grid(True)
                plt.axis('equal')
                plt.savefig(os.path.join(folder, f"{result_key}_trajectory.png"), dpi=150)
                plt.close()
    
    # 保存总结报告
    with open(os.path.join(folder, 'summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("GPS+IMU MSCKF Fusion Experiment Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"GPS Frequency: {args.gps_freq} Hz\n")
        f.write(f"GPS Noise: {args.gps_noise} m\n")
        f.write(f"IMU Frequency: {args.imu_freq} Hz\n")
        f.write(f"RTE/ROE Window: {args.rte_window} frames\n")
        f.write(f"MSCKF Window Size: {args.max_window_size}\n\n")
        
        for key, result in all_results.items():
            f.write(f"\n{key}:\n")
            f.write(f"  Mean Position Error: {result['mean_pos_error']:.4f} m\n")
            f.write(f"  Final Position Error: {result['final_pos_error']:.4f} m\n")
            if 'gps_count' in result:
                f.write(f"  GPS Updates: {result['gps_count']}\n")
            if 'airio_count' in result:
                f.write(f"  AirIO Updates: {result['airio_count']}\n")
            if 'ate' in result:
                f.write(f"  ATE: {result['ate']:.4f} m\n")
            if 'aoe_deg' in result:
                f.write(f"  AOE: {result['aoe_deg']:.4f} deg\n")
            if 'rte' in result:
                f.write(f"  RTE ({args.rte_window} frames): {result['rte']:.4f} m\n")
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
    parser = argparse.ArgumentParser(description='GPS + AirIMU + AirIO MSCKF Fusion')
    
    # 设备配置
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    
    # 数据配置
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", 
                       help="Dataset configuration file")
    
    # AirIMU模型输出路径
    parser.add_argument("--airimu_exp", type=str, 
                       default="experiments/euroc/airimu_code",
                       help="Path(s) to AirIMU net_output.pickle")
    
    # AirIO模型输出路径
    parser.add_argument("--airio_exp", type=str,
                       default="experiments/euroc/motion_body_rot",
                       help="Path(s) to AirIO net_output.pickle")
    
    # 融合模式
    parser.add_argument("--mode", type=str, default="FULL_FUSION",
                       choices=['GPS_ONLY', 'AIRIO_ONLY', 'GPS_AIRIO', 'GPS_AIRIMU', 
                               'FULL_FUSION', 'ALL'],
                       help="Fusion mode")
    
    # GPS模拟参数
    parser.add_argument("--gps_freq", type=int, default=1, help="GPS frequency (Hz)")
    parser.add_argument("--imu_freq", type=int, default=200, help="IMU frequency (Hz)")
    parser.add_argument("--gps_noise", type=float, default=0.1, help="GPS noise std (m)")
    parser.add_argument("--gps_bias", type=float, nargs=3, default=None, 
                       help="GPS bias [x, y, z] (m)")
    
    # 评估指标参数
    parser.add_argument("--rte_window", type=int, default=100, 
                       help="Window size for RTE/ROE calculation (frames)")
    
    # MSCKF窗口大小
    parser.add_argument("--max_window_size", type=int, default=200,
                       help="MSCKF sliding window size")
    
    # 快速测试模式
    parser.add_argument("--debug", action="store_true", 
                       help="Debug mode: only process first N frames")
    parser.add_argument("--max_frames", type=int, default=1000, 
                       help="Maximum number of frames to process in debug mode")
    
    # EKF噪声参数
    parser.add_argument("--r_velo", type=float, default=0.01, help="Velocity observation noise")
    parser.add_argument("--r_pos", type=float, default=0.1, help="Position observation noise")
    
    # IMU噪声参数
    parser.add_argument("--gyro_noise", type=float, default=0.001, 
                       help="Gyroscope measurement noise (rad/s)")
    parser.add_argument("--acc_noise", type=float, default=0.01, 
                       help="Accelerometer measurement noise (m/s^2)")
    parser.add_argument("--gyro_bias_noise", type=float, default=1e-6, 
                       help="Gyroscope bias random walk (rad/s^2)")
    parser.add_argument("--acc_bias_noise", type=float, default=1e-5, 
                       help="Accelerometer bias random walk (m/s^3)")
    
    # 初始状态协方差
    parser.add_argument("--init_rot_std", type=float, default=0.01, 
                       help="Initial rotation uncertainty (rad)")
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
    
    # AirIMU/AirIO不确定性权重
    parser.add_argument("--input_weight", type=float, default=1e2,
                       help="Weight for AirIMU acceleration covariance")
    parser.add_argument("--obs_weight", type=float, default=1e-1,
                       help="Weight for AirIO velocity covariance")
    parser.add_argument("--bias_weight", type=float, default=1e-12,
                       help="Bias random walk noise")
    
    # 输出配置
    parser.add_argument("--savedir", type=str, default="./EKFresult/msckf_fusion",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # 如果模式是ALL，运行所有模式
    if args.mode == 'ALL':
        args.mode = 'GPS_ONLY,AIRIO_ONLY,GPS_AIRIO,FULL_FUSION'
    
    print("\nArguments:")
    print(args)
    print()
    
    run_experiment(args)
