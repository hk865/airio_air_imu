# GPS + AirIMU ESKF Fusion
# 使用AirIMU校正后的IMU数据和协方差，融合1Hz GPS观测
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 将项目根目录插入到 sys.path 的最前面
sys.path.insert(0, project_root)

import json
import argparse
import numpy as np
import pypose as pp
import pickle

import torch
import torch.utils.data as Data

from pyhocon import ConfigFactory

from datasets import SeqInfDataset, SeqDataset
from datasets.dataset_utils import imu_seq_collate

from utils import CPU_Unpickler, integrate
from utils.visualize_state import visualize_rotations, visualize_state_error
from evaluation.ESKF_with_anomaly_detection import ESKF, quat_multiply, quat_inverse, quat_rotate, quat_to_rot, exp_quat


def integrate_with_eskf(seq_dataset, inference_state, gps_freq=1.0, att_freq=None, gps_std=0.1, 
                        device='cpu', use_gt_rot=False, savedir='./ekf_result', data_name='unknown'):
    """
    使用ESKF融合AirIMU校正后的IMU数据和模拟GPS观测
    
    Args:
        seq_dataset: SeqDataset对象
        inference_state: AirIMU网络输出结果，包含校正后的IMU和协方差
        gps_freq: GPS频率(Hz)
        gps_std: GPS观测标准差(m)
        device: 计算设备
        use_gt_rot: 是否使用真值旋转进行重力补偿
        savedir: 异常检测报告保存目录
        data_name: 序列名称
    
    Returns:
        包含轨迹和误差的结果字典
    """
    # 获取数据
    data = seq_dataset.data
    seqlen = seq_dataset.seqlen
    gravity = seq_dataset.get_gravity()
    
    # 应用AirIMU校正
    acc_corrected = data['acc'][:-1].clone()
    gyro_corrected = data['gyro'][:-1].clone()

    # 获取协方差（如果存在）
    # 对于Raw IMU，使用VinsMono的EuRoc配置
    # VinsMono: acc_noise = 0.08, gyro_noise = 0.004 (连续时间噪声密度)
    # 这里存储的是噪声密度的平方，ESKF会乘以dt得到离散协方差
    acc_cov = inference_state.get('acc_cov', [(0.08**2) * torch.ones_like(acc_corrected)])[0]
    gyro_cov = inference_state.get('gyro_cov', [(0.004**2) * torch.ones_like(gyro_corrected)])[0]
    
    # 标记是否使用AirIMU（检查correction是否存在且非零）
    use_airimu = False

    # 应用AirIMU校正并获取协方差
    if 'correction_acc' in inference_state:
        correction_acc = inference_state['correction_acc'].cpu()[0]
        acc_corrected += correction_acc
        acc_cov = inference_state.get('acc_cov', [0.08 * torch.ones_like(acc_corrected)])[0]
        # 只有当correction非零时才认为是AirIMU
        correction_max = torch.abs(correction_acc).max().item()
        if correction_max > 1e-10:
            use_airimu = True
    
    if 'correction_gyro' in inference_state:
        gyro_corrected += inference_state['correction_gyro'].cpu()[0]
        gyro_cov = inference_state.get('gyro_cov', [0.006 * torch.ones_like(gyro_corrected)])[0]
    
    # 转换为numpy用于ESKF
    acc_np = acc_corrected.cpu().numpy()
    gyro_np = gyro_corrected.cpu().numpy()
    dt_np = data['dt'][:-1].cpu().numpy().flatten()
    
    # 真值数据用于计算误差
    gt_pos = data['gt_translation'].cpu().numpy()
    gt_rot = data['gt_orientation'].cpu().numpy()
    gt_vel = data['velocity'].cpu().numpy()
    
    # 获取协方差（逐帧）
    acc_cov_np = acc_cov.cpu().numpy()  # (N, 3)
    gyro_cov_np = gyro_cov.cpu().numpy()  # (N, 3)
    
    # 如果是AirIMU，协方差乘以100作为损失函数的补偿
    if use_airimu:
        acc_cov_np = acc_cov_np * 100
        gyro_cov_np = gyro_cov_np * 1
    
    # 对于ESKF，我们需要连续时间噪声密度
    # acc_cov_np和gyro_cov_np存储的是噪声密度的平方（sigma^2）
    # 因此：sigma = sqrt(cov)
    
    # 计算逐帧的噪声参数（每个轴独立）
    # sigma_a: (N, 3), sigma_g: (N, 3)
    sigma_a = np.sqrt(acc_cov_np)
    sigma_g = np.sqrt(gyro_cov_np)
    
    # 偏置随机游走参数（根据是否使用AirIMU选择不同值）
    # AirIMU模式：使用小噪声抑制姿态协方差膨胀（参考AirIO: 1e-12）
    # 纯IMU模式：使用较大的噪声以适应原始IMU的不确定性
    if use_airimu:
        sigma_ba = 1e-12  # AirIMU模式：加速度计偏置随机游走
        sigma_bg = 1e-12  # AirIMU模式：陀螺仪偏置随机游走
    else:
        sigma_ba = 1e-4   # 纯IMU模式：加速度计偏置随机游走
        sigma_bg = 1e-6   # 纯IMU模式：陀螺仪偏置随机游走
    
    cov_label = "(scaled x100)" if use_airimu else "(not scaled)"
    print(f"  AirIMU cov {cov_label}: acc_cov={acc_cov_np.mean():.6f}, gyro_cov={gyro_cov_np.mean():.6f}")
    print(f"  ESKF noise: sigma_a={sigma_a.mean():.6f}, sigma_g={sigma_g.mean():.6f}, sigma_ba={sigma_ba:.6e}, sigma_bg={sigma_bg:.6e}")
    
    # 初始化ESKF
    init_p = gt_pos[0].copy()
    init_v = gt_vel[0].copy()
    # gt_rot是[x,y,z,w]格式（pypose SO3），ESKF需要[w,x,y,z]格式
    init_q = np.array([gt_rot[0, 3], gt_rot[0, 0], gt_rot[0, 1], gt_rot[0, 2]])  # [w, x, y, z]
    init_bg = np.zeros(3)
    init_ba = np.zeros(3)
    
    eskf = ESKF(init_p, init_v, init_q, init_bg, init_ba,
                sigma_g, sigma_a, sigma_bg, sigma_ba, gps_std, 
                g=np.array([0, 0, -gravity]))
    
    # 存储结果
    est_pos = [init_p.copy()]
    est_vel = [init_v.copy()]
    est_rot = [init_q.copy()]
    
    # GPS和姿态更新间隔（根据IMU数据频率计算）
    imu_freq = 1.0 / np.mean(dt_np)
    gps_interval = int(imu_freq / gps_freq)
    
    # 姿态观测频率（默认与GPS相同，可以独立设置）
    if att_freq is None:
        att_freq = gps_freq
    att_interval = int(imu_freq / att_freq)
    
    # 判断是否可以使用同步更新（当频率相同或成整数倍时）
    use_sync_update = (gps_interval == att_interval) or (max(gps_interval, att_interval) % min(gps_interval, att_interval) == 0)
    if use_sync_update:
        print(f"  Using synchronized pose update (GPS interval: {gps_interval}, Attitude interval: {att_interval})")
    else:
        print(f"  Using separate updates (GPS interval: {gps_interval}, Attitude interval: {att_interval})")
    
    # 时间累积
    time_accum = 0.0
    last_gps_time = 0.0
    
    # 注意：dt_np长度是seqlen，gt_pos长度是seqlen+1（包含初始帧）
    # 我们只积分seqlen步，所以est_pos会有seqlen+1个元素（包含初始帧）
    for i in range(len(dt_np)):
        # IMU测量值 [a_m, w_m]
        imu_meas = np.concatenate([acc_np[i], gyro_np[i]])
        dt = dt_np[i]
        
        # ESKF预测步骤（传递逐时刻的噪声参数）
        p, q = eskf.predict(imu_meas, dt, sigma_a=sigma_a[i], sigma_g=sigma_g[i])
        
        time_accum += dt
        
        # 判断当前时刻的更新类型
        has_gps = (i % gps_interval == 0 and i > 0)
        has_attitude = (i % att_interval == 0 and i > 0)
        
        if use_sync_update and has_gps and has_attitude:
            # ===== 同步位姿更新 =====
            # 当GPS和姿态观测在同一时刻到达时，使用同步更新
            # 这避免了分开更新导致的偏置震荡问题
            
            # GPS观测
            gps_noise = np.random.normal(0, gps_std, 3)
            gps_meas = gt_pos[i+1] + gps_noise
            
            # 姿态观测
            att_noise = np.random.normal(0, 0.03, 3)
            q_true_xyzw = gt_rot[i+1]
            q_true = np.array([q_true_xyzw[3], q_true_xyzw[0], q_true_xyzw[1], q_true_xyzw[2]])
            if np.linalg.norm(att_noise) > 1e-12:
                axis = att_noise / np.linalg.norm(att_noise)
                angle = np.linalg.norm(att_noise)
                dq_noise = exp_quat(axis * angle)
                q_meas = quat_multiply(q_true, dq_noise)
                q_meas = q_meas / np.linalg.norm(q_meas)
            else:
                q_meas = q_true
            
            # 同步更新
            eskf.update_with_pose(gps_meas, q_meas, gps_std, 0.03)
            last_gps_time = time_accum
            
        else:
            # ===== 分开更新（频率不同或不成整数倍时） =====
            
            # GPS更新步骤
            if has_gps:
                gps_noise = np.random.normal(0, gps_std, 3)
                gps_meas = gt_pos[i+1] + gps_noise
                eskf.update(gps_meas)
                last_gps_time = time_accum
            
            # 姿态观测更新步骤
            if has_attitude:
                att_noise = np.random.normal(0, 0.03, 3)
                q_true_xyzw = gt_rot[i+1]
                q_true = np.array([q_true_xyzw[3], q_true_xyzw[0], q_true_xyzw[1], q_true_xyzw[2]])
                if np.linalg.norm(att_noise) > 1e-12:
                    axis = att_noise / np.linalg.norm(att_noise)
                    angle = np.linalg.norm(att_noise)
                    dq_noise = exp_quat(axis * angle)
                    q_meas = quat_multiply(q_true, dq_noise)
                    q_meas = q_meas / np.linalg.norm(q_meas)
                else:
                    q_meas = q_true
                eskf.update_with_attitude(q_meas, att_std=0.03)
        
        # 保存估计结果
        est_pos.append(eskf.p.copy())
        est_vel.append(eskf.v.copy())
        est_rot.append(eskf.q.copy())
    
    # 转换为tensor - 确保长度与gt一致
    est_pos = torch.tensor(np.array(est_pos), dtype=torch.float64)
    est_vel = torch.tensor(np.array(est_vel), dtype=torch.float64)
    est_rot = torch.tensor(np.array(est_rot), dtype=torch.float64)
    
    # 截取与估计结果相同长度的真值（从开头截取）
    gt_pos_tensor = torch.tensor(gt_pos[:len(est_pos)], dtype=torch.float64)
    gt_vel_tensor = torch.tensor(gt_vel[:len(est_vel)], dtype=torch.float64)
    gt_rot_tensor = torch.tensor(gt_rot[:len(est_rot)], dtype=torch.float64)
    
    # 计算误差
    pos_dist = torch.norm(est_pos - gt_pos_tensor, dim=-1)
    vel_dist = torch.norm(est_vel - gt_vel_tensor, dim=-1)
    
    # 旋转误差（使用pypose计算）
    # ESKF输出的是[w,x,y,z]格式，需要转换为[x,y,z,w]格式以匹配pypose SO3
    # gt_rot_tensor已经是[x,y,z,w]格式（来自pypose SO3），无需转换
    est_rot_xyzw = torch.stack([est_rot[:, 1], est_rot[:, 2], est_rot[:, 3], est_rot[:, 0]], dim=-1)
    
    est_rot_SO3 = pp.SO3(est_rot_xyzw)
    gt_rot_SO3 = pp.SO3(gt_rot_tensor)  # gt_rot_tensor已经是[x,y,z,w]格式
    rot_dist = (est_rot_SO3.Inv() @ gt_rot_SO3).Log().norm(dim=-1)
    
    # 为了与integrate函数返回格式保持一致
    # 添加一个batch维度，并在第1维添加初始帧（与integrate一致）
    est_pos_batch = est_pos.unsqueeze(0)
    est_vel_batch = est_vel.unsqueeze(0)
    # 转换四元数格式从[w,x,y,z]到[x,y,z,w]以匹配pypose SO3
    est_rot_batch = est_rot_xyzw.unsqueeze(0)
    gt_pos_batch = gt_pos_tensor.unsqueeze(0)
    gt_vel_batch = gt_vel_tensor.unsqueeze(0)
    # gt_rot_tensor已经是[x,y,z,w]格式，无需转换
    gt_rot_batch = gt_rot_tensor.unsqueeze(0)
    
    # 保存异常检测报告
    anomaly_report_path = os.path.join(savedir, f"{data_name}_anomaly_report.json")
    eskf.save_anomaly_report(anomaly_report_path)
    
    return {
        'poses': est_pos_batch,
        'vel': est_vel_batch,
        'orientations': est_rot_batch,
        'poses_gt': gt_pos_batch,
        'vel_gt': gt_vel_batch,
        'orientations_gt': gt_rot_batch,
        'pos_dist': pos_dist.unsqueeze(0),
        'vel_dist': vel_dist.unsqueeze(0),
        'rot_dist': rot_dist.unsqueeze(0),
    }


def integrate_raw_imu(seq_dataset, device='cpu', use_gt_rot=False):
    """
    使用原始IMU数据进行积分（不使用AirIMU校正）
    用于对比
    """
    loader = Data.DataLoader(
        dataset=seq_dataset, 
        batch_size=1, 
        collate_fn=imu_seq_collate, 
        shuffle=False, 
        drop_last=False
    )
    
    init = seq_dataset.get_init_value()
    gravity = seq_dataset.get_gravity()
    
    integrator = pp.module.IMUPreintegrator(
        init['pos'], init['rot'], init['vel'],
        gravity=gravity,
        reset=False
    ).to(device).double()
    
    return integrate(
        integrator, loader, init, 
        device=device, gtinit=False, save_full_traj=True,
        use_gt_rot=use_gt_rot
    )


def integrate_airimu(seq_dataset, inference_state, device='cpu', use_gt_rot=False):
    """
    使用AirIMU校正后的IMU数据进行积分（不使用ESKF，仅积分）
    用于对比
    """
    # 直接使用seq_dataset的数据，应用AirIMU校正
    # 创建一个新的数据集对象，但不重新加载数据
    class SimpleDataset:
        def __init__(self, data, seqlen, index_map, gravity):
            self.data = data
            self.seqlen = seqlen
            self.index_map = index_map
            self._gravity = gravity
        
        def __len__(self):
            return len(self.index_map)
        
        def __getitem__(self, i):
            frame_id, end_frame_id = self.index_map[i]
            return {
                'dt': self.data['dt'][frame_id: end_frame_id],
                'acc': self.data['acc'][frame_id: end_frame_id],
                'gyro': self.data['gyro'][frame_id: end_frame_id],
                'rot': self.data['gt_orientation'][frame_id: end_frame_id],
                'gt_pos': self.data['gt_translation'][frame_id+1: end_frame_id+1],
                'gt_rot': self.data['gt_orientation'][frame_id+1: end_frame_id+1],
                'gt_vel': self.data['velocity'][frame_id+1: end_frame_id+1],
                'init_pos': self.data['gt_translation'][frame_id][None, ...],
                'init_rot': self.data['gt_orientation'][frame_id: end_frame_id],
                'init_vel': self.data['velocity'][frame_id][None, ...],
            }
        
        def get_init_value(self):
            return {'pos': self.data['gt_translation'][:1],
                    'rot': self.data['gt_orientation'][:1],
                    'vel': self.data['velocity'][:1]}
        
        def get_gravity(self):
            return self._gravity
    
    # 复制数据并应用校正
    data_corrected = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in seq_dataset.data.items()}
    data_corrected['acc'][:-1] += inference_state['correction_acc'].cpu()[0]
    data_corrected['gyro'][:-1] += inference_state['correction_gyro'].cpu()[0]
    
    dataset_inf = SimpleDataset(
        data_corrected,
        seq_dataset.seqlen,
        seq_dataset.index_map,
        seq_dataset.get_gravity()
    )
    
    loader = Data.DataLoader(
        dataset=dataset_inf,
        batch_size=1,
        collate_fn=imu_seq_collate,
        shuffle=False,
        drop_last=False
    )
    
    init = seq_dataset.get_init_value()
    gravity = seq_dataset.get_gravity()
    
    integrator = pp.module.IMUPreintegrator(
        init['pos'], init['rot'], init['vel'],
        gravity=gravity,
        reset=False
    ).to(device).double()
    
    return integrate(
        integrator, loader, init,
        device=device, gtinit=False, save_full_traj=True,
        use_gt_rot=use_gt_rot
    )


def compute_relative_error(outstate, seq_dataset):
    """
    计算相对误差（与evaluate_state.py中的逻辑一致）
    """
    index_id = seq_dataset.index_map[:, -1]
    select_mask = torch.ones_like(seq_dataset.get_mask()[index_id], dtype=torch.bool)
    select_mask[-1] = False
    
    return {
        'pos_dist': outstate['pos_dist'][0, select_mask],
        'vel_dist': outstate['vel_dist'][0, select_mask],
        'rot_dist': outstate['rot_dist'][0, select_mask],
        'mask': select_mask
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu, Default is cpu")
    parser.add_argument("--exp", type=str, default="experiments/EuRoC/freqnet_improved_0", 
                        help="the directory path where your network output pickle file is stored")
    parser.add_argument("--seqlen", type=int, default="200", help="the length of the integration sequence")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/BaselineEuroc/Euroc_1000.conf", 
                        help="the configuration of the dataset")
    parser.add_argument("--savedir", type=str, default="./ekf_result", 
                        help="the save directory for the evaluation results")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")
    parser.add_argument("--gps_freq", type=float, default=1.0, help="GPS frequency in Hz")
    parser.add_argument("--att_freq", type=float, default=None, help="Attitude observation frequency in Hz (default: same as gps_freq)")
    parser.add_argument("--gps_std", type=float, default=0.1, help="GPS observation standard deviation in meters")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for GPS noise")
    
    args = parser.parse_args()
    print("\n" * 3 + str(args) + "\n" * 3)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载配置
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    
    # 加载网络结果
    net_result_path = os.path.join(args.exp, 'net_output.pickle')
    if os.path.isfile(net_result_path):
        with open(net_result_path, 'rb') as handle:
            inference_state_load = CPU_Unpickler(handle).load()
        print(f"Loaded network results from: {net_result_path}")
    else:
        raise Exception(f"Unable to load the network result: {net_result_path}")
    
    # 创建保存目录
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)
    
    AllResults = []
    
    # 处理每个数据序列
    for data_conf in dataset_conf.data_list:
        print(f"Processing data configuration: {data_conf}")
        for data_name in data_conf.data_drive:
            print(f"\n{'='*60}")
            print(f"Processing sequence: {data_name}")
            print(f"Data root: {data_conf.data_root}")
            print(f"{'='*60}")
            
            # 创建数据集
            dataset = SeqDataset(
                data_conf.data_root, 
                data_name, 
                args.device, 
                name=data_conf.name, 
                duration=args.seqlen, 
                step_size=args.seqlen, 
                drop_last=False, 
                conf=dataset_conf
            )
            
            # 获取真值数据用于生成GPS观测
            gt_pos = dataset.data['gt_translation'].cpu().numpy()
            gt_rot = dataset.data['gt_orientation'].cpu().numpy()
            gt_vel = dataset.data['velocity'].cpu().numpy()
            
            # 1. 原始IMU积分结果
            print("\n[1/4] Integrating raw IMU data...")
            outstate_raw = integrate_raw_imu(dataset, device=args.device, use_gt_rot=args.usegtrot)
            
            # 2. AirIMU校正后的积分结果
            print("[2/4] Integrating AirIMU corrected data...")
            inference_state = inference_state_load[data_name]
            outstate_airimu = integrate_airimu(dataset, inference_state, device=args.device, use_gt_rot=args.usegtrot)
            
            # 3. ESKF + 原始IMU + GPS
            print(f"[3/4] Running ESKF with raw IMU + {args.gps_freq}Hz GPS (std={args.gps_std}m)...")
            # 创建空的inference_state（无校正）
            empty_state = {
                'correction_acc': torch.zeros_like(inference_state['correction_acc']),
                'correction_gyro': torch.zeros_like(inference_state['correction_gyro']),
            }
            if 'acc_cov' in inference_state:
                empty_state['acc_cov'] = [torch.ones_like(inference_state['acc_cov'][0]) * 0.08]
                empty_state['gyro_cov'] = [torch.ones_like(inference_state['gyro_cov'][0]) * 0.006]
            
            outstate_eskf_raw = integrate_with_eskf(
                dataset, empty_state,
                gps_freq=args.gps_freq, att_freq=args.att_freq, gps_std=args.gps_std,
                device=args.device, use_gt_rot=args.usegtrot,
                savedir=folder, data_name=f"{data_name}_raw"
            )

            # 4. ESKF + AirIMU + GPS
            att_freq_str = f"{args.att_freq}Hz" if args.att_freq else f"{args.gps_freq}Hz (same as GPS)"
            print(f"[4/4] Running ESKF with AirIMU + {args.gps_freq}Hz GPS (std={args.gps_std}m), attitude {att_freq_str}...")
            outstate_eskf_airimu = integrate_with_eskf(
                dataset, inference_state,
                gps_freq=args.gps_freq, att_freq=args.att_freq, gps_std=args.gps_std,
                device=args.device, use_gt_rot=args.usegtrot,
                savedir=folder, data_name=f"{data_name}_airimu"
            )
            
            # 计算mask
            # 注意：integrate函数返回的是分段积分结果，而integrate_with_eskf返回的是完整轨迹
            # 对于ESKF结果，我们使用不同的mask逻辑
            full_length = dataset.seqlen
            mask = torch.ones(full_length, dtype=torch.bool)
            
            # 计算指标
            print("\n" + "="*60)
            print("Computing evaluation metrics...")
            print("="*60)
            
            # 绝对误差（使用全部数据）
            def compute_metrics(outstate, name, use_full_mask=False):
                print(f"\n{name}:")
                if use_full_mask:
                    # 对于ESKF结果，使用完整长度的mask
                    pos_err = outstate['pos_dist'][0].mean().item()
                    rot_err = outstate['rot_dist'][0].mean().item()
                    vel_err = outstate['vel_dist'][0].mean().item()
                else:
                    pos_err = outstate['pos_dist'][0, mask].mean().item()
                    rot_err = outstate['rot_dist'][0, mask].mean().item()
                    vel_err = outstate['vel_dist'][0, mask].mean().item()
                print(f"  ATE: {pos_err:.4f} m")
                print(f"  AOE: {180./np.pi * rot_err:.4f} deg")
                print(f"  AVE: {vel_err:.4f} m/s")
                return pos_err, rot_err, vel_err
            
            # 相对误差 - 计算短时间窗口内的漂移（真正的RTE/ROE）
            def compute_relative_metrics(outstate, name, use_full_mask=False):
                """
                计算相对轨迹误差（RTE）和相对姿态误差（ROE）
                
                与ATE/AOE不同，RTE/ROE测量的是短时间内的局部一致性，
                不需要全局对齐。计算方法：
                - 使用滑动窗口（如1秒=200帧）
                - 在每个窗口内，计算起点到终点的漂移
                - 对所有窗口的漂移取平均
                """
                # 使用1秒的窗口（假设200Hz IMU）
                window_size = min(200, len(outstate['poses'][0]) // 2)
                if window_size < 10:
                    window_size = 10
                
                poses = outstate['poses'][0]  # (N, 3)
                gt_poses = outstate['poses_gt'][0]  # (N, 3)
                rots = outstate['orientations'][0]  # (N, 4) [x,y,z,w]
                gt_rots = outstate['orientations_gt'][0]  # (N, 4) [x,y,z,w]
                
                rel_pos_errors = []
                rel_rot_errors = []
                
                # 滑动窗口计算相对误差
                for i in range(0, len(poses) - window_size, window_size // 2):  # 50%重叠
                    start = i
                    end = i + window_size
                    
                    # 获取窗口起点和终点
                    p_start = poses[start]
                    p_end = poses[end]
                    gt_start = gt_poses[start]
                    gt_end = gt_poses[end]
                    # 转换为普通tensor（避免pypose LieTensor的问题）
                    q_start = rots[start].tensor() if hasattr(rots[start], 'tensor') else torch.tensor(rots[start])  # [x,y,z,w]
                    q_end = rots[end].tensor() if hasattr(rots[end], 'tensor') else torch.tensor(rots[end])
                    q_gt_start = gt_rots[start].tensor() if hasattr(gt_rots[start], 'tensor') else torch.tensor(gt_rots[start])
                    q_gt_end = gt_rots[end].tensor() if hasattr(gt_rots[end], 'tensor') else torch.tensor(gt_rots[end])
                    
                    # 四元数乘法辅助函数
                    def quat_mult(q1, q2):
                        x1, y1, z1, w1 = q1
                        x2, y2, z2, w2 = q2
                        return torch.tensor([
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2,
                            w1*w2 - x1*x2 - y1*y2 - z1*z2
                        ])
                    
                    def quat_conj(q):
                        return torch.tensor([-q[0], -q[1], -q[2], q[3]])
                    
                    def quat_to_rotmat(q):
                        """四元数转旋转矩阵 (3x3)"""
                        x, y, z, w = q
                        return torch.tensor([
                            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
                        ])
                    
                    # 计算相对姿态漂移 = angle( (q_start^-1 * q_end) vs (q_gt_start^-1 * q_gt_end) )
                    q_rel_est = quat_mult(quat_conj(q_start), q_end)
                    q_rel_gt = quat_mult(quat_conj(q_gt_start), q_gt_end)
                    q_error = quat_mult(q_rel_est, quat_conj(q_rel_gt))
                    w_error = torch.clamp(q_error[3], -1.0, 1.0)
                    rot_err = 2 * torch.acos(torch.abs(w_error))
                    rel_rot_errors.append(rot_err.item())
                    
                    # RTE: 相对位置漂移（需要对齐姿态）
                    # 步骤：
                    # 1. 计算姿态对齐旋转: R_align = R_gt_start * R_est_start^T
                    # 2. 将估计轨迹对齐到真值坐标系
                    # 3. 计算对齐后的位置漂移
                    R_est_start = quat_to_rotmat(q_start)
                    R_gt_start = quat_to_rotmat(q_gt_start)
                    R_align = R_gt_start @ R_est_start.T
                    
                    # 在窗口起点对齐位置
                    p_start_aligned = p_start
                    gt_start_aligned = gt_start
                    
                    # 计算相对位移（在各自坐标系中）
                    est_delta = p_end - p_start
                    gt_delta = gt_end - gt_start
                    
                    # 将估计的相对位移旋转到真值坐标系
                    est_delta_aligned = R_align @ est_delta
                    
                    # 相对位置漂移 = |对齐后的估计位移 - 真值位移|
                    rel_pos_err = torch.norm(est_delta_aligned - gt_delta).item()
                    rel_pos_errors.append(rel_pos_err)
                
                if len(rel_pos_errors) > 0:
                    pos_err = np.mean(rel_pos_errors)
                    rot_err = np.mean(rel_rot_errors)
                    rp_rmse = np.sqrt(np.mean([e**2 for e in rel_pos_errors]))
                    ro_rmse = 180./np.pi * np.sqrt(np.mean([e**2 for e in rel_rot_errors]))
                else:
                    pos_err = rot_err = rp_rmse = ro_rmse = 0.0
                
                # 速度误差使用全局误差的滑动窗口平均（速度本身就是瞬时的）
                vel_dists = outstate['vel_dist'][0]
                num_windows = len(vel_dists) // window_size
                rel_vel_errors = []
                for i in range(num_windows):
                    start = i * window_size
                    end = min((i + 1) * window_size, len(vel_dists))
                    if end > start:
                        rel_vel_errors.append(vel_dists[start:end].mean().item())
                vel_err = np.mean(rel_vel_errors) if rel_vel_errors else 0.0
                rv_rmse = np.sqrt(np.mean([e**2 for e in rel_vel_errors])) if rel_vel_errors else 0.0
                
                print(f"  RTE: {pos_err:.4f} m")
                print(f"  ROE: {180./np.pi * rot_err:.4f} deg")
                print(f"  RVE: {vel_err:.4f} m/s")
                print(f"  RP_RMSE: {rp_rmse:.4f} m")
                print(f"  RV_RMSE: {rv_rmse:.4f} m/s")
                print(f"  RO_RMSE: {ro_rmse:.4f} deg")
                return pos_err, rot_err, vel_err, rp_rmse, rv_rmse, ro_rmse
            
            print("\n--- Absolute Errors ---")
            raw_ate, raw_aoe, raw_ave = compute_metrics(outstate_raw, "Raw IMU")
            airimu_ate, airimu_aoe, airimu_ave = compute_metrics(outstate_airimu, "AirIMU")
            eskf_raw_ate, eskf_raw_aoe, eskf_raw_ave = compute_metrics(outstate_eskf_raw, "ESKF + Raw IMU + GPS", use_full_mask=True)
            eskf_airimu_ate, eskf_airimu_aoe, eskf_airimu_ave = compute_metrics(outstate_eskf_airimu, "ESKF + AirIMU + GPS", use_full_mask=True)
            
            print("\n--- Relative Errors ---")
            raw_rte, raw_roe, raw_rve, raw_rp_rmse, raw_rv_rmse, raw_ro_rmse = compute_relative_metrics(outstate_raw, "Raw IMU")
            airimu_rte, airimu_roe, airimu_rve, airimu_rp_rmse, airimu_rv_rmse, airimu_ro_rmse = compute_relative_metrics(outstate_airimu, "AirIMU")
            eskf_raw_rte, eskf_raw_roe, eskf_raw_rve, eskf_raw_rp_rmse, eskf_raw_rv_rmse, eskf_raw_ro_rmse = compute_relative_metrics(outstate_eskf_raw, "ESKF + Raw IMU + GPS", use_full_mask=True)
            eskf_airimu_rte, eskf_airimu_roe, eskf_airimu_rve, eskf_airimu_rp_rmse, eskf_airimu_rv_rmse, eskf_airimu_ro_rmse = compute_relative_metrics(outstate_eskf_airimu, "ESKF + AirIMU + GPS", use_full_mask=True)
            
            # 保存结果
            result_dic = {
                'name': data_name,
                'gps_freq': args.gps_freq,
                'gps_std': args.gps_std,
                'use_gt_rot': args.usegtrot,
                
                # Raw IMU
                'ATE(raw)': raw_ate,
                'AOE(raw)': 180./np.pi * raw_aoe,
                'AVE(raw)': raw_ave,
                'RTE(raw)': raw_rte,
                'ROE(raw)': 180./np.pi * raw_roe,
                'RVE(raw)': raw_rve,
                'RP_RMSE(raw)': raw_rp_rmse,
                'RV_RMSE(raw)': raw_rv_rmse,
                'RO_RMSE(raw)': raw_ro_rmse,
                
                # AirIMU
                'ATE(AirIMU)': airimu_ate,
                'AOE(AirIMU)': 180./np.pi * airimu_aoe,
                'AVE(AirIMU)': airimu_ave,
                'RTE(AirIMU)': airimu_rte,
                'ROE(AirIMU)': 180./np.pi * airimu_roe,
                'RVE(AirIMU)': airimu_rve,
                'RP_RMSE(AirIMU)': airimu_rp_rmse,
                'RV_RMSE(AirIMU)': airimu_rv_rmse,
                'RO_RMSE(AirIMU)': airimu_ro_rmse,
                
                # ESKF + Raw IMU + GPS
                'ATE(ESKF_Raw)': eskf_raw_ate,
                'AOE(ESKF_Raw)': 180./np.pi * eskf_raw_aoe,
                'AVE(ESKF_Raw)': eskf_raw_ave,
                'RTE(ESKF_Raw)': eskf_raw_rte,
                'ROE(ESKF_Raw)': 180./np.pi * eskf_raw_roe,
                'RVE(ESKF_Raw)': eskf_raw_rve,
                'RP_RMSE(ESKF_Raw)': eskf_raw_rp_rmse,
                'RV_RMSE(ESKF_Raw)': eskf_raw_rv_rmse,
                'RO_RMSE(ESKF_Raw)': eskf_raw_ro_rmse,
                
                # ESKF + AirIMU + GPS
                'ATE(ESKF_AirIMU)': eskf_airimu_ate,
                'AOE(ESKF_AirIMU)': 180./np.pi * eskf_airimu_aoe,
                'AVE(ESKF_AirIMU)': eskf_airimu_ave,
                'RTE(ESKF_AirIMU)': eskf_airimu_rte,
                'ROE(ESKF_AirIMU)': 180./np.pi * eskf_airimu_roe,
                'RVE(ESKF_AirIMU)': eskf_airimu_rve,
                'RP_RMSE(ESKF_AirIMU)': eskf_airimu_rp_rmse,
                'RV_RMSE(ESKF_AirIMU)': eskf_airimu_rv_rmse,
                'RO_RMSE(ESKF_AirIMU)': eskf_airimu_ro_rmse,
            }
            
            AllResults.append(result_dic)
            
            # 保存轨迹数据用于后续3D可视化
            trajectory_data = {
                'name': data_name,
                'gt_pos': outstate_eskf_airimu['poses_gt'][0].cpu().numpy().tolist(),
                'gt_rot': outstate_eskf_airimu['orientations_gt'][0].cpu().numpy().tolist(),
                'eskf_airimu_pos': outstate_eskf_airimu['poses'][0].cpu().numpy().tolist(),
                'eskf_airimu_rot': outstate_eskf_airimu['orientations'][0].cpu().numpy().tolist(),
                'eskf_raw_pos': outstate_eskf_raw['poses'][0].cpu().numpy().tolist(),
                'airimu_pos': outstate_airimu['poses'][0].cpu().numpy().tolist(),
                'raw_pos': outstate_raw['poses'][0].cpu().numpy().tolist(),
            }
            traj_file = os.path.join(folder, f"{data_name}_trajectory.json")
            with open(traj_file, 'w') as f:
                json.dump(trajectory_data, f)
            print(f"  Trajectory data saved to: {traj_file}")
            
            # 可视化
            print(f"\nGenerating visualizations...")
            visualize_state_error(
                data_name, outstate_raw, outstate_eskf_airimu,
                save_folder=folder, mask=mask
            )
            visualize_rotations(
                data_name,
                outstate_raw['orientations_gt'][0],
                outstate_raw['orientations'][0],
                outstate_eskf_airimu['orientations'][0],
                save_folder=folder
            )
            
            print(f"\nCompleted processing: {data_name}")
    
    # 保存所有结果到JSON
    file_path = os.path.join(folder, "ekf_result.json")
    with open(file_path, 'w') as f:
        json.dump(AllResults, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {file_path}")
    print(f"{'='*60}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("Summary of all sequences:")
    print("="*60)
    for result in AllResults:
        print(f"\n{result['name']}:")
        print(f"  Raw IMU - ATE: {result['ATE(raw)']:.4f}m, RTE: {result['RTE(raw)']:.4f}m")
        print(f"  AirIMU  - ATE: {result['ATE(AirIMU)']:.4f}m, RTE: {result['RTE(AirIMU)']:.4f}m")
        print(f"  ESKF+Raw - ATE: {result['ATE(ESKF_Raw)']:.4f}m, RTE: {result['RTE(ESKF_Raw)']:.4f}m")
        print(f"  ESKF+AirIMU - ATE: {result['ATE(ESKF_AirIMU)']:.4f}m, RTE: {result['RTE(ESKF_AirIMU)']:.4f}m")

#python evaluation/GPS_airIMU.py --exp experiments/EuRoC/freqnet_improved_0 --seqlen 200 --dataconf configs/datasets/BaselineEuroc/Euroc_1000.conf --savedir ./ekf_result --gps_freq 1.0 --att_freq 1.0 --gps_std 0.1 --seed 42
#python evaluation/GPS_airIMU.py --exp experiments/EuRoC/freqnet_improved_0 --seqlen 200 --dataconf configs/datasets/BaselineEuroc/Euroc_1000.conf --savedir ./ekf_result --gps_freq 0.5 --att_freq 0.1 --gps_std 0.1 --seed 42
#python evaluation/GPS_airIMU.py --exp experiments/EuRoC/codenet --seqlen 200 --dataconf configs/datasets/BaselineEuroc/Euroc_1000.conf --savedir ./ekf_result --gps_freq 0.5 --att_freq 0.1 --gps_std 0.1 --seed 42
#python evaluation/GPS_airIMU.py --exp experiments/EuRoC/freqnet1 --seqlen 200 --dataconf configs/datasets/BaselineEuroc/Euroc_1000.conf --savedir ./ekf_result --gps_freq 0.5 --att_freq 0.1 --gps_std 0.1 --seed 42

#python evaluation/GPS_airIMU.py --exp experiments/EuRoC/freqnet_improved_0 --seqlen 200 --dataconf configs/datasets/BaselineEuroc/Euroc_1000.conf --savedir ./ekf_result --gps_freq 1.0 --att_freq 1.0 --gps_std 0.1 --seed 42