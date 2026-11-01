import os
import torch
import io, pickle
import numpy as np
from inspect import currentframe, getframeinfo
import wandb
import pypose as pp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def save_state(out_states:dict, in_state:dict):
    for k, v in in_state.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            save_state(out_states=out_states, in_state=v)
        elif k in out_states.keys():
            out_states[k].append(v)
        else:
            out_states[k] = [v]

def Gaussian_noise(num_nodes, sigma_x=0.05 ,sigma_y=0.05, sigma_z=0.05):
    std = torch.stack([torch.ones(num_nodes)*sigma_x, torch.ones(num_nodes)*sigma_y, torch.ones(num_nodes)*sigma_z], dim=-1)
    return torch.normal(mean = 0, std = std)

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def resolve_dtype(precision="float32"):
    try:
        return _DTYPE_BY_NAME[str(precision).lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_DTYPE_BY_NAME))
        raise ValueError(
            f"Unsupported precision {precision!r}; choose one of: {supported}"
        ) from exc


def as_so3(rotation):
    """Restore SO(3) semantics after batching/device conversion."""
    return rotation if hasattr(rotation, "Log") else pp.SO3(rotation)


def move_to(obj, device, dtype=None):
    if torch.is_tensor(obj):
        target_dtype = dtype if (
            dtype is not None and (obj.is_floating_point() or obj.is_complex())
        ) else None
        return obj.to(device=device, dtype=target_dtype)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, dtype=dtype)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, dtype=dtype))
        return res
    elif isinstance(obj, np.ndarray):
        return move_to(torch.as_tensor(obj), device, dtype=dtype)
    else:
        raise TypeError("Invalid type for move_to", type(obj))

def qinterp(qs, t, t_int):
    qs = R.from_quat(qs.numpy())
    slerp = Slerp(t, qs)
    interp_rot = slerp(t_int).as_quat()
    return torch.tensor(interp_rot)
def interp_xyz(time, opt_time, xyz):
    intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
    intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
    intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return torch.tensor(inte_xyz)
def lookAt(dir_vec, up = torch.tensor([0.,0.,1.], dtype=torch.float64), source = torch.tensor([0.,0.,0.], dtype=torch.float64)):
    '''
    dir_vec: the tensor shall be (1)
    return the rotation matrix of the 
    '''
    if not isinstance(dir_vec, torch.Tensor):
        dir_vec = torch.tensor(dir_vec)
    def normalize(x):
        length = x.norm()
        if length< 0.005:
            length = 1
            warnings.warn('Normlization error that the norm is too small')
        return x/length
            
    zaxis = normalize(dir_vec - source)
    xaxis = normalize(torch.cross(zaxis, up))
    yaxis = torch.cross(xaxis, zaxis)

    m = torch.tensor([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [zaxis[0], zaxis[1], zaxis[2]],
    ])

    return m

def cat_state(in_state:dict):
    pop_list = []
    for k, v in in_state.items():
        if len(v[0].shape) > 2:
            in_state[k] = torch.cat(v,  dim=-2)
        else:
            pop_list.append(k)
    for k in pop_list:
        in_state.pop(k)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def write_board(writer, objs, epoch_i, header = ''):
    # writer = SummaryWriter(log_dir=conf.general.exp_dir)
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                writer.add_scalar(os.path.join(header, k), v, epoch_i)
    elif isinstance(objs, float):
        writer.add_scalar(header, v, epoch_i)

def write_wandb(header, objs, epoch_i):
    if isinstance(objs, dict):
        for k, v in objs.items():
            if isinstance(v, float):
                wandb.log({os.path.join(header, k): v}, epoch_i)
    else:
        wandb.log({header: objs}, step = epoch_i)

def save_ckpt(network, optimizer, scheduler, epoch_i, test_loss, conf, save_best=False, best_eval_loss=None):
    # 构建保存的字典
    save_dict = {
        'epoch': epoch_i,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': test_loss,
    }
    
    # 如果提供了best_eval_loss，也保存它
    if best_eval_loss is not None:
        save_dict['best_eval_loss'] = best_eval_loss
    
    # 定期保存
    if epoch_i % conf.train.save_freq == conf.train.save_freq - 1:
        torch.save(save_dict, os.path.join(conf.general.exp_dir, "ckpt/%04d.ckpt" % epoch_i))

    # 保存test最佳模型
    if save_best:
        print("saving the best model (test)", test_loss)
        torch.save(save_dict, os.path.join(conf.general.exp_dir, "ckpt/best_model.ckpt"))
    
    # 总是保存最新模型
    torch.save(save_dict, os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"))


def report_hasNan(x):
    cf = currentframe().f_back
    res = torch.any(torch.isnan(x)).cpu().item()
    if res: print(f"[hasnan!] {getframeinfo(cf).filename}:{cf.f_lineno}")


def report_hasNeg(x):
    cf = currentframe().f_back
    res = torch.any(x < 0).cpu().item()
    if res: print(f"[hasneg!] {getframeinfo(cf).filename}:{cf.f_lineno}")

def mat_log(R):
    """
    对数映射，R 形状 (..., 3, 3)
    返回旋转向量 (..., 3)
    """
    # 先将矩阵转换为 SO3 对象，然后取 Log
    return pp.Log(pp.SO3(R))  # 或 pp.SO3(R).Log()
    
