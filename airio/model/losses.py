import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan
import numpy as np
import torch.nn.functional as F

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def position_loss_(pred_pos, gt_pos, loss_type='huber', delta=0.005):
    """
    计算位置损失
    pred_pos: (B, T, 3) 预测位置
    gt_pos: (B, T, 3) 真值位置
    loss_type: 'huber' 或 'l2'
    delta: Huber损失的delta参数
    """
    pos_error = pred_pos - gt_pos
    if loss_type == 'huber':
        loss = F.huber_loss(pos_error, torch.zeros_like(pos_error), delta=delta)
    else:
        loss = pos_error.pow(2).mean()
    return loss, pos_error

def get_motion_loss(inte_state, label, confs):
    ## The state loss for evaluation
    loss, cov_loss = 0, {}
    loss_fc = loss_fc_list[confs.loss]
    
    # 处理 label 可能是字典或张量的情况
    if isinstance(label, dict):
        gt_vel = label['gt_vel']
        gt_pos = label.get('gt_pos', None)
    else:
        gt_vel = label
        gt_pos = None
    
    vel_loss, vel_dist = motion_loss_(loss_fc, inte_state['net_vel'], gt_vel)

    # Apply the covariance loss
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss += confs.weight * vel_loss
    
    # Position loss (optional)
    pos_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    if getattr(confs, 'posloss', False) and inte_state.get('pos') is not None and gt_pos is not None:
        pos_weight = getattr(confs, 'pos_weight', 1.0)
        pos_loss_type = getattr(confs, 'loss', 'Huber_loss005')
        if pos_loss_type.startswith('Huber'):
            delta = float(pos_loss_type.split('loss')[-1]) / 100 if 'loss' in pos_loss_type else 0.005
            pos_loss, _ = position_loss_(inte_state['pos'], gt_pos, loss_type='huber', delta=delta)
        else:
            pos_loss, _ = position_loss_(inte_state['pos'], gt_pos, loss_type='l2')
        loss += pos_weight * pos_loss
    
    return {'loss': loss, 'cov_loss': cov_loss, 'pos_loss': pos_loss}


def get_motion_loss_detailed(inte_state, label, confs):
    """
    详细版本的损失计算，返回所有分项损失
    用于训练时记录和打印详细损失信息
    """
    loss_fc = loss_fc_list[confs.loss]
    
    # 处理 label 可能是字典或张量的情况
    if isinstance(label, dict):
        gt_vel = label['gt_vel']
    else:
        gt_vel = label
    
    vel_loss, vel_dist = motion_loss_(loss_fc, inte_state['net_vel'], gt_vel)
    vel_loss_raw = vel_loss.clone()
    
    cov_loss = torch.tensor(0.0, device=vel_loss.device, dtype=vel_loss.dtype)
    if confs.propcov:
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    
    total_loss = confs.weight * vel_loss
    
    pos_loss = torch.tensor(0.0, device=vel_loss.device, dtype=vel_loss.dtype)
    if getattr(confs, 'posloss', False) and inte_state.get('pos') is not None and label.get('gt_pos') is not None:
        pos_weight = getattr(confs, 'pos_weight', 1.0)
        pos_loss_type = getattr(confs, 'loss', 'Huber_loss005')
        if pos_loss_type.startswith('Huber'):
            delta = float(pos_loss_type.split('loss')[-1]) / 100 if 'loss' in pos_loss_type else 0.005
            pos_loss, _ = position_loss_(inte_state['pos'], label['gt_pos'], loss_type='huber', delta=delta)
        else:
            pos_loss, _ = position_loss_(inte_state['pos'], label['gt_pos'], loss_type='l2')
        total_loss += pos_weight * pos_loss
    
    return {
        'loss': total_loss,
        'vel_loss': vel_loss_raw,
        'pos_loss': pos_loss,
        'cov_loss': cov_loss,
        'vel_loss_weighted': confs.weight * vel_loss,
        'pos_loss_weighted': getattr(confs, 'pos_weight', 1.0) * pos_loss if getattr(confs, 'posloss', False) else torch.tensor(0.0)
    }


def get_motion_RMSE(inte_state, label, confs):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())
    
    # 处理 label 可能是字典或张量的情况
    if isinstance(label, dict):
        gt_vel = label['gt_vel']
        gt_pos = label.get('gt_pos', None)
    else:
        gt_vel = label
        gt_pos = None
    
    cov_loss = 0
    dist = (inte_state['net_vel'] - gt_vel)
    dist = torch.mean(dist,dim=-2)
    loss = _RMSE(dist)[None,...]
    
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()
    
    # Position RMSE (optional)
    pos_rmse = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    if getattr(confs, 'posloss', False) and inte_state.get('pos') is not None and gt_pos is not None:
        pos_error = inte_state['pos'] - gt_pos
        pos_error_mean = torch.mean(pos_error, dim=-2)
        pos_rmse = _RMSE(pos_error_mean)[None, ...]
    
    return {'loss': loss, 
            'dist': dist.norm(dim=-1).mean(),
            'cov_loss': cov_loss,
            'pos_rmse': pos_rmse}
