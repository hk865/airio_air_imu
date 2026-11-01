# output the trajctory in the world frame for visualization and evaluation
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import argparse
import numpy as np
import torch
import pypose as pp

from pyhocon import ConfigFactory
from datasets import SeqDataset


def calculate_rte(outstate, duration, step_size):
    poses, poses_gt = outstate['poses'], outstate['poses_gt'][1:, :]

    dp = poses[duration - 1:] - poses[:-duration + 1]
    dp_gt = poses_gt[duration - 1:] - poses_gt[:-duration + 1]

    rte = (dp - dp_gt).norm(dim=-1)
    return rte


def calculate_aoe(rot_pred, rot_gt):
    """
    Calculate Attitude Orientation Error (AOE)
    rot_pred: predicted rotation in so3 log coordinates (N, 3)
    rot_gt: ground truth rotation as SO3 objects or rotation matrices
    """
    # Convert predicted so3 to rotation matrix
    pred_rot = pp.so3(rot_pred).Exp()
    
    # Calculate rotation error: error_rot = pred_rot^-1 * gt_rot
    error_rot = pred_rot.Inv() @ rot_gt
    
    # Get the angle from so3 log (this gives the rotation angle in radians)
    error_angle = error_rot.Log().norm(dim=-1)
    
    # Convert to degrees
    error_angle_deg = error_angle * 180.0 / np.pi
    
    return error_angle_deg


def calculate_roe(rot_pred, rot_gt, duration):
    """
    Calculate Relative Orientation Error (ROE)
    Similar to RTE but for rotations
    """
    # Convert to rotation matrices
    pred_rot = pp.so3(rot_pred).Exp().matrix()
    gt_rot_matrix = rot_gt.matrix()
    
    # Calculate relative rotations over the duration window
    # dr = R_{t+duration} @ R_t^T
    dr_pred = pred_rot[duration-1:] @ pred_rot[:-duration+1].transpose(-2, -1)
    dr_gt = gt_rot_matrix[duration-1:] @ gt_rot_matrix[:-duration+1].transpose(-2, -1)
    
    # Calculate relative rotation error
    error_rot_matrix = dr_pred @ dr_gt.transpose(-2, -1)
    
    # Convert rotation matrix to SO3 then get so3 log coordinates
    error_so3 = pp.from_matrix(error_rot_matrix, ltype=pp.SO3_type).Log()
    error_angle = error_so3.norm(dim=-1)
    
    # Convert to degrees
    error_angle_deg = error_angle * 180.0 / np.pi
    
    return error_angle_deg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--seqlen", type=int, default="1000", help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_global.conf",
                        help="Path to the dataset configuration file. ")
    parser.add_argument("--savedir", type=str, default="./result/loss_result",
                        help="Directory where the results wiil be saved")
    parser.add_argument("--exp", type=str, default="./EKFresult/loss_result", help="Path for the EKF results")
    args = parser.parse_args();
    print(("\n" * 3) + str(args) + ("\n" * 3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    print(dataset_conf.keys())

    folder = args.savedir
    os.makedirs(folder, exist_ok=True)

    AllResults = []
    net_out_result = {}

    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            print("data_conf.dataroot", data_conf.data_root)
            print("data_name", data_name)
            print("data_conf.name", data_conf.name)

            motion_dataset = SeqDataset(data_conf.data_root, data_name, args.device, name=data_conf.name,
                                        duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf=dataset_conf)

            if args.exp is not None:
                if data_conf.name == "BlackBird":
                    data_name = os.path.dirname(data_name).split('/')[1]

                ekf_result = np.load(os.path.join(args.exp, data_name + "_ekf_result.npy"))
                ekf_pos = torch.tensor(ekf_result[:, 6:9])
                ekf_rot = torch.tensor(ekf_result[:, 0:3])  # so3 log coordinates

                inf_state = {}
                inf_state['poses'] = ekf_pos
                inf_state['poses_gt'] = motion_dataset.data['gt_translation']

                pos_dist = (inf_state['poses'] - inf_state['poses_gt'][1:, :]).norm(dim=-1)
                inf_rte = calculate_rte(inf_state, args.seqlen, args.seqlen)

                # Calculate orientation errors
                gt_orientation = motion_dataset.data['gt_orientation'][1:]  # Skip first frame to match ekf_result
                
                # AOE - Absolute Orientation Error
                aoe = calculate_aoe(ekf_rot, gt_orientation)
                
                # ROE - Relative Orientation Error
                roe = calculate_roe(ekf_rot, gt_orientation, args.seqlen)

                # save loss result
                result_dic = {
                    'name': data_name,
                    'ATE(EKF)': torch.sqrt((pos_dist ** 2).mean()).item(),
                    'RTE(EKF)': inf_rte.mean().item(),
                    'RP_RMSE(EKF)': np.sqrt((inf_rte ** 2).mean()).numpy().item(),
                    'AOE(EKF)': aoe.mean().item(),  # Mean Absolute Orientation Error in degrees
                    'ROE(EKF)': roe.mean().item(),  # Mean Relative Orientation Error in degrees
                    'AOE_RMSE(EKF)': np.sqrt((aoe ** 2).mean()).numpy().item(),  # RMSE of AOE
                    'ROE_RMSE(EKF)': np.sqrt((roe ** 2).mean()).numpy().item(),  # RMSE of ROE
                }

                AllResults.append(result_dic)

                print("==============EKF==============")
                print("infstate:")
                print("pos_err: ", pos_dist.mean())
                print("rte", inf_rte.mean())
                print("aoe (deg):", aoe.mean())
                print("roe (deg):", roe.mean())

        file_path = os.path.join(folder, "ekf_result.json")
        with open(file_path, 'w') as f:
            json.dump(AllResults, f, indent=4)
