"""
轨迹评估脚本 - 计算 AOE、ATE、ROE、RTE 四个指标
支持指定 RTE、ROE 的序列长度（delta_t）
输出每个指标的 rmse、std、mean、min、max
"""

import os
import sys
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import Dict, List, Tuple, Any


class TrajectoryMetricsEvaluator:
    """
    轨迹评估器，计算 AOE、ATE、ROE、RTE 指标
    """
    
    def __init__(self, delta_t: int = 200):
        """
        初始化评估器
        
        Args:
            delta_t: RTE 和 ROE 计算的时间间隔（帧数）
        """
        self.delta_t = delta_t
    
    def load_poses(self, file_path: str) -> np.ndarray:
        """
        加载位姿文件
        
        期望格式: (N, 8) [timestamp, x, y, z, qw, qx, qy, qz]
        或 (N, 7) [x, y, z, qw, qx, qy, qz]
        
        Returns:
            poses: (N, 7) [x, y, z, qw, qx, qy, qz]
        """
        data = np.load(file_path)
        
        if data.shape[1] == 8:
            # 包含时间戳，去掉第一列
            return data[:, 1:]
        elif data.shape[1] == 7:
            return data
        else:
            raise ValueError(f"Unsupported pose format: {data.shape}")
    
    def compute_ate(self, est_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
        """
        计算绝对轨迹误差 (ATE)
        ATE = sqrt(1/N * sum(||p_i - p_hat_i||^2))
        
        Args:
            est_poses: 估计位姿 (N, 7) [x, y, z, qw, qx, qy, qz]
            gt_poses: 真实位姿 (N, 7)
            
        Returns:
            ATE 指标字典
        """
        # 提取位置
        est_positions = est_poses[:, :3]
        gt_positions = gt_poses[:, :3]
        
        # 计算位置误差
        errors = est_positions - gt_positions
        point_errors = np.linalg.norm(errors, axis=1)
        
        return {
            'rmse': float(np.sqrt(np.mean(point_errors ** 2))),
            'mean': float(np.mean(point_errors)),
            'std': float(np.std(point_errors)),
            'min': float(np.min(point_errors)),
            'max': float(np.max(point_errors))
        }
    
    def compute_rte(self, est_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
        """
        计算相对轨迹误差 (RTE)
        RTE = ||(p_{i+Δt} - p_i) - R_i R_hat_i^T (p_hat_{i+Δt} - p_hat_i)||
        
        Args:
            est_poses: 估计位姿 (N, 7)
            gt_poses: 真实位姿 (N, 7)
            
        Returns:
            RTE 指标字典
        """
        n = min(len(est_poses), len(gt_poses))
        if n <= self.delta_t:
            return {'rmse': 0.0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        relative_errors = []
        
        for i in range(n - self.delta_t):
            # 真实相对位移
            gt_displacement = gt_poses[i + self.delta_t, :3] - gt_poses[i, :3]
            
            # 估计相对位移
            est_displacement = est_poses[i + self.delta_t, :3] - est_poses[i, :3]
            
            # 获取旋转矩阵 (四元数格式 [x, y, z, w])
            gt_rot_i = R.from_quat(gt_poses[i, [4, 5, 6, 3]])
            est_rot_i = R.from_quat(est_poses[i, [4, 5, 6, 3]])
            
            # 计算 R_i * R_hat_i^T
            rotation_correction = gt_rot_i.as_matrix() @ est_rot_i.as_matrix().T
            
            # 应用旋转校正
            corrected_est_displacement = rotation_correction @ est_displacement
            
            # 计算相对误差
            error_vector = gt_displacement - corrected_est_displacement
            relative_error = np.linalg.norm(error_vector)
            relative_errors.append(relative_error)
        
        relative_errors = np.array(relative_errors)
        
        return {
            'rmse': float(np.sqrt(np.mean(relative_errors ** 2))),
            'mean': float(np.mean(relative_errors)),
            'std': float(np.std(relative_errors)),
            'min': float(np.min(relative_errors)),
            'max': float(np.max(relative_errors))
        }
    
    def compute_aoe(self, est_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
        """
        计算绝对旋转误差 (AOE)
        AOE = ||log(R_n^T R_hat_n)||
        
        Args:
            est_poses: 估计位姿 (N, 7)
            gt_poses: 真实位姿 (N, 7)
            
        Returns:
            AOE 指标字典（角度制）
        """
        n = min(len(est_poses), len(gt_poses))
        rotation_errors = []
        
        for i in range(n):
            # 获取旋转矩阵
            gt_rot = R.from_quat(gt_poses[i, [4, 5, 6, 3]]).as_matrix()
            est_rot = R.from_quat(est_poses[i, [4, 5, 6, 3]]).as_matrix()
            
            # 计算 R_n^T R_hat_n
            relative_rot = gt_rot.T @ est_rot
            
            # 转换为旋转向量
            rotation_vector = R.from_matrix(relative_rot).as_rotvec()
            
            # 计算旋转角度（弧度）
            rotation_angle = np.linalg.norm(rotation_vector)
            rotation_errors.append(rotation_angle)
        
        rotation_errors = np.array(rotation_errors)
        rotation_errors_deg = np.rad2deg(rotation_errors)
        
        return {
            'rmse': float(np.sqrt(np.mean(rotation_errors_deg ** 2))),
            'mean': float(np.mean(rotation_errors_deg)),
            'std': float(np.std(rotation_errors_deg)),
            'min': float(np.min(rotation_errors_deg)),
            'max': float(np.max(rotation_errors_deg))
        }
    
    def compute_roe(self, est_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, float]:
        """
        计算相对旋转误差 (ROE)
        ROE = ||log(δR_{n,g(n)}^T δR_hat_{n,g(n)})||
        
        Args:
            est_poses: 估计位姿 (N, 7)
            gt_poses: 真实位姿 (N, 7)
            
        Returns:
            ROE 指标字典（角度制）
        """
        n = min(len(est_poses), len(gt_poses))
        if n <= self.delta_t:
            return {'rmse': 0.0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        relative_rotation_errors = []
        
        for i in range(n - self.delta_t):
            # 计算真实相对旋转
            gt_rot_i = R.from_quat(gt_poses[i, [4, 5, 6, 3]])
            gt_rot_i_delta = R.from_quat(gt_poses[i + self.delta_t, [4, 5, 6, 3]])
            gt_relative_rot = gt_rot_i_delta * gt_rot_i.inv()
            
            # 计算估计相对旋转
            est_rot_i = R.from_quat(est_poses[i, [4, 5, 6, 3]])
            est_rot_i_delta = R.from_quat(est_poses[i + self.delta_t, [4, 5, 6, 3]])
            est_relative_rot = est_rot_i_delta * est_rot_i.inv()
            
            # 计算相对旋转误差
            relative_error_rot = gt_relative_rot.inv() * est_relative_rot
            
            # 转换为旋转向量并计算角度
            error_rotation_vector = relative_error_rot.as_rotvec()
            error_angle = np.linalg.norm(error_rotation_vector)
            
            relative_rotation_errors.append(error_angle)
        
        relative_rotation_errors = np.array(relative_rotation_errors)
        relative_rotation_errors_deg = np.rad2deg(relative_rotation_errors)
        
        return {
            'rmse': float(np.sqrt(np.mean(relative_rotation_errors_deg ** 2))),
            'mean': float(np.mean(relative_rotation_errors_deg)),
            'std': float(np.std(relative_rotation_errors_deg)),
            'min': float(np.min(relative_rotation_errors_deg)),
            'max': float(np.max(relative_rotation_errors_deg))
        }
    
    def evaluate(self, est_poses: np.ndarray, gt_poses: np.ndarray) -> Dict[str, Any]:
        """
        完整评估，计算所有四个指标
        
        Args:
            est_poses: 估计位姿 (N, 7)
            gt_poses: 真实位姿 (N, 7)
            
        Returns:
            包含所有指标的字典
        """
        # 确保轨迹长度一致
        min_len = min(len(est_poses), len(gt_poses))
        est_poses = est_poses[:min_len]
        gt_poses = gt_poses[:min_len]
        
        return {
            'ate': self.compute_ate(est_poses, gt_poses),
            'rte': self.compute_rte(est_poses, gt_poses),
            'aoe': self.compute_aoe(est_poses, gt_poses),
            'roe': self.compute_roe(est_poses, gt_poses),
            'trajectory_length': min_len,
            'delta_t': self.delta_t
        }


def find_gt_file(scene_name: str, gt_root: str) -> str:
    """
    根据场景名查找对应的真值文件
    
    Args:
        scene_name: 场景名称（如 MH_01_easy）
        gt_root: 真值文件根目录
        
    Returns:
        真值文件路径
    """
    # 常见的真值文件命名模式
    gt_patterns = [
        f"{scene_name}_gt.npy",
        f"{scene_name}_ground_truth.npy",
        f"{scene_name}_poses.npy",
        f"{scene_name}.npy",
        f"gt_{scene_name}.npy",
    ]
    
    # 在 gt_root 及其子目录中查找
    for root, dirs, files in os.walk(gt_root):
        for pattern in gt_patterns:
            gt_path = os.path.join(root, pattern)
            if os.path.exists(gt_path):
                return gt_path
    
    return None


def evaluate_folder(input_folder: str, gt_folder: str, delta_t: int, output_file: str):
    """
    评估文件夹中的所有 EKF 结果文件
    
    Args:
        input_folder: 包含 _ekf_poses.npy 文件的文件夹
        gt_folder: 真值文件文件夹
        delta_t: RTE 和 ROE 的时间间隔
        output_file: 输出 JSON 文件路径
    """
    evaluator = TrajectoryMetricsEvaluator(delta_t=delta_t)
    
    # 查找所有 _ekf_poses.npy 文件
    npy_files = list(Path(input_folder).glob("*_ekf_poses.npy"))
    
    if not npy_files:
        print(f"警告: 在 {input_folder} 中未找到 *_ekf_poses.npy 文件")
        # 也尝试查找其他模式
        npy_files = list(Path(input_folder).glob("*.npy"))
        print(f"找到的其他 .npy 文件: {[f.name for f in npy_files]}")
        # 过滤出可能是位姿文件的
        npy_files = [f for f in npy_files if 'pose' in f.name.lower() or 'ekf' in f.name.lower()]
    
    print(f"\n找到 {len(npy_files)} 个待评估文件")
    print(f"RTE/ROE 时间间隔 (delta_t): {delta_t}")
    print("=" * 80)
    
    all_results = []
    
    for npy_file in sorted(npy_files):
        scene_name = npy_file.stem.replace("_ekf_poses", "").replace("_poses", "").replace("_ekf", "")
        
        print(f"\n处理场景: {scene_name}")
        print(f"  文件: {npy_file.name}")
        
        try:
            # 加载估计位姿
            est_poses = evaluator.load_poses(str(npy_file))
            print(f"  估计位姿形状: {est_poses.shape}")
            
            # 查找对应的真值文件
            gt_file = find_gt_file(scene_name, gt_folder)
            
            if gt_file is None:
                # 尝试在输入文件夹中查找
                gt_candidates = list(Path(input_folder).glob(f"{scene_name}*gt*.npy"))
                if not gt_candidates:
                    gt_candidates = list(Path(input_folder).glob(f"*{scene_name}*.npy"))
                    gt_candidates = [f for f in gt_candidates if 'ekf' not in f.name.lower()]
                
                if gt_candidates:
                    gt_file = str(gt_candidates[0])
            
            if gt_file is None:
                print(f"  警告: 未找到场景 {scene_name} 的真值文件，跳过")
                continue
            
            print(f"  真值文件: {gt_file}")
            
            # 加载真值位姿
            gt_poses = evaluator.load_poses(gt_file)
            print(f"  真值位姿形状: {gt_poses.shape}")
            
            # 计算指标
            metrics = evaluator.evaluate(est_poses, gt_poses)
            
            # 构建结果
            result = {
                'scene_name': scene_name,
                'file_name': npy_file.name,
                'gt_file': gt_file,
                'metrics': metrics
            }
            
            all_results.append(result)
            
            # 打印指标
            print(f"\n  评估结果:")
            print(f"    轨迹长度: {metrics['trajectory_length']}")
            print(f"    ATE (m)  - RMSE: {metrics['ate']['rmse']:.4f}, Mean: {metrics['ate']['mean']:.4f}, "
                  f"Std: {metrics['ate']['std']:.4f}, Min: {metrics['ate']['min']:.4f}, Max: {metrics['ate']['max']:.4f}")
            print(f"    RTE (m)  - RMSE: {metrics['rte']['rmse']:.4f}, Mean: {metrics['rte']['mean']:.4f}, "
                  f"Std: {metrics['rte']['std']:.4f}, Min: {metrics['rte']['min']:.4f}, Max: {metrics['rte']['max']:.4f}")
            print(f"    AOE (deg) - RMSE: {metrics['aoe']['rmse']:.4f}, Mean: {metrics['aoe']['mean']:.4f}, "
                  f"Std: {metrics['aoe']['std']:.4f}, Min: {metrics['aoe']['min']:.4f}, Max: {metrics['aoe']['max']:.4f}")
            print(f"    ROE (deg) - RMSE: {metrics['roe']['rmse']:.4f}, Mean: {metrics['roe']['mean']:.4f}, "
                  f"Std: {metrics['roe']['std']:.4f}, Min: {metrics['roe']['min']:.4f}, Max: {metrics['roe']['max']:.4f}")
            
        except Exception as e:
            print(f"  错误: 处理文件时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算平均指标
    if all_results:
        avg_metrics = {
            'ate': {
                'rmse': float(np.mean([r['metrics']['ate']['rmse'] for r in all_results])),
                'mean': float(np.mean([r['metrics']['ate']['mean'] for r in all_results])),
                'std': float(np.mean([r['metrics']['ate']['std'] for r in all_results])),
                'min': float(np.mean([r['metrics']['ate']['min'] for r in all_results])),
                'max': float(np.mean([r['metrics']['ate']['max'] for r in all_results]))
            },
            'rte': {
                'rmse': float(np.mean([r['metrics']['rte']['rmse'] for r in all_results])),
                'mean': float(np.mean([r['metrics']['rte']['mean'] for r in all_results])),
                'std': float(np.mean([r['metrics']['rte']['std'] for r in all_results])),
                'min': float(np.mean([r['metrics']['rte']['min'] for r in all_results])),
                'max': float(np.mean([r['metrics']['rte']['max'] for r in all_results]))
            },
            'aoe': {
                'rmse': float(np.mean([r['metrics']['aoe']['rmse'] for r in all_results])),
                'mean': float(np.mean([r['metrics']['aoe']['mean'] for r in all_results])),
                'std': float(np.mean([r['metrics']['aoe']['std'] for r in all_results])),
                'min': float(np.mean([r['metrics']['aoe']['min'] for r in all_results])),
                'max': float(np.mean([r['metrics']['aoe']['max'] for r in all_results]))
            },
            'roe': {
                'rmse': float(np.mean([r['metrics']['roe']['rmse'] for r in all_results])),
                'mean': float(np.mean([r['metrics']['roe']['mean'] for r in all_results])),
                'std': float(np.mean([r['metrics']['roe']['std'] for r in all_results])),
                'min': float(np.mean([r['metrics']['roe']['min'] for r in all_results])),
                'max': float(np.mean([r['metrics']['roe']['max'] for r in all_results]))
            }
        }
        
        print("\n" + "=" * 80)
        print("平均指标:")
        print(f"  ATE (m)  - RMSE: {avg_metrics['ate']['rmse']:.4f}, Mean: {avg_metrics['ate']['mean']:.4f}")
        print(f"  RTE (m)  - RMSE: {avg_metrics['rte']['rmse']:.4f}, Mean: {avg_metrics['rte']['mean']:.4f}")
        print(f"  AOE (deg) - RMSE: {avg_metrics['aoe']['rmse']:.4f}, Mean: {avg_metrics['aoe']['mean']:.4f}")
        print(f"  ROE (deg) - RMSE: {avg_metrics['roe']['rmse']:.4f}, Mean: {avg_metrics['roe']['mean']:.4f}")
    else:
        avg_metrics = None
    
    # 保存结果到 JSON
    output_data = {
        'config': {
            'input_folder': input_folder,
            'gt_folder': gt_folder,
            'delta_t': delta_t,
            'num_scenes': len(all_results)
        },
        'individual_results': all_results,
        'average_metrics': avg_metrics
    }
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='评估轨迹指标 (AOE, ATE, ROE, RTE)')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='包含 _ekf_poses.npy 文件的输入文件夹')
    parser.add_argument('--gt', '-g', type=str, required=True,
                        help='真值文件文件夹')
    parser.add_argument('--delta_t', '-d', type=int, default=200,
                        help='RTE 和 ROE 计算的时间间隔（帧数），默认 200')
    parser.add_argument('--output', '-o', type=str, default='trajectory_metrics.json',
                        help='输出 JSON 文件路径，默认 trajectory_metrics.json')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("轨迹指标评估工具")
    print("=" * 80)
    print(f"输入文件夹: {args.input}")
    print(f"真值文件夹: {args.gt}")
    print(f"时间间隔 delta_t: {args.delta_t}")
    print(f"输出文件: {args.output}")
    
    evaluate_folder(args.input, args.gt, args.delta_t, args.output)


if __name__ == '__main__':
    main()
