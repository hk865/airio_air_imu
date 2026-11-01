import argparse
import os
import tqdm, wandb
from model import net_dict

from datasets import collate_fcs,SeqeuncesMotionDataset
import pickle

import numpy as np
import pypose as pp
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import yaml
from pyhocon import ConfigFactory
from pyhocon import HOCONConverter as conf_convert
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.losses import get_motion_loss, get_motion_RMSE, get_motion_loss_detailed
from utils import (as_so3, cat_state, move_to, resolve_dtype, save_ckpt, save_state,
                   write_wandb)
import copy


class LossLogger:
    """
    损失记录器：用于记录和打印详细的损失信息
    可通过配置启用/禁用，易于维护和扩展
    使用 TensorBoard 记录
    """
    def __init__(self, confs, writer=None):
        self.confs = confs
        self.enable_detailed = getattr(confs, 'posloss', False)
        self.writer = writer
        self.reset()
    
    def reset(self):
        """重置累积损失"""
        self.total_loss = 0.0
        self.vel_loss = 0.0
        self.pos_loss = 0.0
        self.cov_loss = 0.0
        self.count = 0
    
    def update(self, loss_dict):
        """更新累积损失"""
        self.total_loss += loss_dict['loss'].item() if isinstance(loss_dict['loss'], torch.Tensor) else loss_dict['loss']
        
        vel_loss = loss_dict.get('vel_loss', 0.0)
        self.vel_loss += vel_loss.item() if isinstance(vel_loss, torch.Tensor) else vel_loss
        
        pos_loss = loss_dict.get('pos_loss', 0.0)
        self.pos_loss += pos_loss.item() if isinstance(pos_loss, torch.Tensor) else pos_loss
        
        cov_loss = loss_dict.get('cov_loss', 0.0)
        self.cov_loss += cov_loss.item() if isinstance(cov_loss, torch.Tensor) else cov_loss
        
        self.count += 1
    
    def get_avg(self):
        """获取平均损失"""
        if self.count == 0:
            return {'total': 0.0, 'vel': 0.0, 'pos': 0.0, 'cov': 0.0}
        return {
            'total': self.total_loss / self.count,
            'vel': self.vel_loss / self.count,
            'pos': self.pos_loss / self.count,
            'cov': self.cov_loss / self.count
        }
    
    def log_to_tensorboard(self, prefix, epoch):
        """记录到 TensorBoard"""
        if not self.enable_detailed or self.writer is None:
            return
        
        avg = self.get_avg()
        self.writer.add_scalar(f'{prefix}/total_loss', avg['total'], epoch)
        self.writer.add_scalar(f'{prefix}/vel_loss', avg['vel'], epoch)
        self.writer.add_scalar(f'{prefix}/pos_loss', avg['pos'], epoch)
        self.writer.add_scalar(f'{prefix}/cov_loss', avg['cov'], epoch)
    
    def print_summary(self, prefix=""):
        """打印损失摘要"""
        if not self.enable_detailed:
            return
        
        avg = self.get_avg()
        print(f"{prefix} Loss Summary - Total: {avg['total']:.6f} | Vel: {avg['vel']:.6f} | Pos: {avg['pos']:.6f} | Cov: {avg['cov']:.6f}")


def train(network, loader, confs, epoch, optimizer, writer=None):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses
    """
    network.train()
    losses, pred_cov = 0, 0
    
    # 初始化详细损失记录器
    loss_logger = LossLogger(confs, writer=writer)

    t_range = tqdm.tqdm(loader, disable=bool(confs.get('quiet', False)))
    for i, (data, init_state, label) in enumerate(t_range):
        runtime_dtype = resolve_dtype(confs.get('precision', 'float32'))
        data, label = move_to([data, label], confs.device, dtype=runtime_dtype)
        rot = as_so3(label['gt_rot'][:, :-1, :]).Log().tensor()
        
        # 准备 init_state (用于位置积分)
        if getattr(confs, 'posloss', False) and init_state is not None:
            init_state = move_to([init_state], confs.device, dtype=runtime_dtype)[0]
            init_pos = init_state['pos']  # (B, 1, 3)
            
            # 检查是否使用滑动窗口
            window_size = getattr(confs, 'pos_window_size', None)
            if window_size is not None:
                # 从 gt_pos 提取每个窗口的初始位置
                gt_pos_full = label['gt_pos']  # (B, T+1, 3)
                seq_len = gt_pos_full.shape[1] - 1  # 减去1因为输出少一个时间步
                num_windows = (seq_len + window_size - 1) // window_size  # 向上取整
                
                # 提取每个窗口的起始位置
                window_init_pos = []
                for w in range(num_windows):
                    start_idx = w * window_size
                    if start_idx < seq_len:
                        window_init_pos.append(gt_pos_full[:, start_idx:start_idx+1, :])
                
                if window_init_pos:
                    init_pos = torch.cat(window_init_pos, dim=1)  # (B, num_windows, 3)
            else:
                # 原始方式：整个序列一个初始位置
                init_pos = init_pos.squeeze(1)  # (B, 3)
        else:
            init_pos = None
        init_state_for_model = {'pos': init_pos} if init_pos is not None else None
        inte_state = (
            network(data, rot, init_state_for_model)
            if getattr(confs, 'posloss', False)
            else network(data, rot)
        )
        gt_label = network.get_label(label['gt_vel'])
        
        # 计算损失
        if getattr(confs, 'posloss', False):
            gt_label_dict = {'gt_vel': gt_label}
            if 'gt_pos' in label:
                gt_label_dict['gt_pos'] = network.get_label(label['gt_pos'])
            # 使用详细损失函数
            loss_state = get_motion_loss_detailed(inte_state, gt_label_dict, confs)
            loss_logger.update(loss_state)
        else:
            loss_state = get_motion_loss(inte_state, gt_label, confs)

        # statistics
        losses += loss_state["loss"].item()

        if confs.propcov:
            pred_cov += loss_state["cov_loss"].mean().item()

        t_range.set_description(
            f"training epoch: %03d,losses: %.06f" % (epoch, loss_state["loss"])
        )

        t_range.refresh()
        optimizer.zero_grad()
        loss_state["loss"].backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

    # 打印详细损失摘要
    loss_logger.print_summary(f"[Train Epoch {epoch}]")
    
    return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1)), "loss_logger": loss_logger}


def test(network, loader, confs):
    network.eval() 
    with torch.no_grad():
        losses, pred_cov = 0, 0

        t_range = tqdm.tqdm(loader, disable=bool(confs.get('quiet', False)))
        for i, (data, init_state, label) in enumerate(t_range):
            runtime_dtype = resolve_dtype(confs.get('precision', 'float32'))
            data,label = move_to([data, label], confs.device, dtype=runtime_dtype)
            rot = as_so3(label['gt_rot'][:, :-1, :]).Log().tensor()
            
            # 准备 init_state (用于位置积分)
            if getattr(confs, 'posloss', False) and init_state is not None:
                init_state = move_to([init_state], confs.device, dtype=runtime_dtype)[0]
                init_pos = init_state['pos']  # (B, 1, 3)
                
                # 检查是否使用滑动窗口
                window_size = getattr(confs, 'pos_window_size', None)
                if window_size is not None:
                    # 从 gt_pos 提取每个窗口的初始位置
                    gt_pos_full = label['gt_pos']  # (B, T+1, 3)
                    seq_len = gt_pos_full.shape[1] - 1
                    num_windows = (seq_len + window_size - 1) // window_size
                    
                    window_init_pos = []
                    for w in range(num_windows):
                        start_idx = w * window_size
                        if start_idx < seq_len:
                            window_init_pos.append(gt_pos_full[:, start_idx:start_idx+1, :])
                    
                    if window_init_pos:
                        init_pos = torch.cat(window_init_pos, dim=1)  # (B, num_windows, 3)
                else:
                    init_pos = init_pos.squeeze(1)  # (B, 3)
            else:
                init_pos = None
            init_state_for_model = {'pos': init_pos} if init_pos is not None else None
            
            inte_state = (
                network(data, rot, init_state_for_model)
                if getattr(confs, 'posloss', False)
                else network(data, rot)
            )
            gt_label = network.get_label(label['gt_vel'])
            
            # 支持位置损失
            if getattr(confs, 'posloss', False):
                gt_label_dict = {'gt_vel': gt_label}
                if 'gt_pos' in label:
                    gt_label_dict['gt_pos'] = network.get_label(label['gt_pos'])
                loss_state = get_motion_RMSE(inte_state, gt_label_dict, confs)
            else:
                loss_state = get_motion_RMSE(inte_state, gt_label, confs)
                
            # statistics
            losses += loss_state["loss"].item()

            if confs.propcov:
                pred_cov += loss_state["cov_loss"].mean().item()
                cov_loss_value = torch.sqrt(loss_state['cov_loss'])
            else:   
                cov_loss_value = 0
            t_range.set_description(
                "testing loss: %.06f, cov: %.06f, error: %.06f" % (
                    losses / (i + 1), 
                    cov_loss_value, 
                    loss_state['dist']
                )
            )

            t_range.refresh()
            
    return {"loss": (losses / (i + 1)), "cov": (pred_cov / (i + 1))}


def evaluate(network, loader, confs, silent_tqdm=False):
    network.eval()
    evaluate_states, loss_states, labels = {}, {}, {}
    pred_cov = []
    skip_key = None

    with torch.no_grad():
        inte_state = None
        for i, (data, init_state, label) in enumerate(
            tqdm.tqdm(loader, disable=bool(confs.get('quiet', False)))
        ):
            
            runtime_dtype = resolve_dtype(confs.get('precision', 'float32'))
            data,label = move_to([data, label], confs.device, dtype=runtime_dtype)
            rot = as_so3(label['gt_rot'][:, :-1, :]).Log().tensor()
            
            # 准备 init_state (用于位置积分)
            if getattr(confs, 'posloss', False) and init_state is not None:
                init_state = move_to([init_state], confs.device, dtype=runtime_dtype)[0]
                init_pos = init_state['pos']  # (B, 1, 3)
                
                # 检查是否使用滑动窗口
                window_size = getattr(confs, 'pos_window_size', None)
                if window_size is not None:
                    # 从 gt_pos 提取每个窗口的初始位置
                    gt_pos_full = label['gt_pos']  # (B, T+1, 3)
                    seq_len = gt_pos_full.shape[1] - 1
                    num_windows = (seq_len + window_size - 1) // window_size
                    
                    window_init_pos = []
                    for w in range(num_windows):
                        start_idx = w * window_size
                        if start_idx < seq_len:
                            window_init_pos.append(gt_pos_full[:, start_idx:start_idx+1, :])
                    
                    if window_init_pos:
                        init_pos = torch.cat(window_init_pos, dim=1)  # (B, num_windows, 3)
                else:
                    init_pos = init_pos.squeeze(1)  # (B, 3)
            else:
                init_pos = None
            init_state_for_model = {'pos': init_pos} if init_pos is not None else None
            
            inte_state = (
                network(data, rot, init_state_for_model)
                if getattr(confs, 'posloss', False)
                else network(data, rot)
            )
            gt_label = network.get_label(label['gt_vel'])
            
            # 支持位置损失
            if getattr(confs, 'posloss', False):
                gt_label_dict = {'gt_vel': gt_label}
                if 'gt_pos' in label:
                    gt_label_dict['gt_pos'] = network.get_label(label['gt_pos'])
                loss_state = get_motion_RMSE(inte_state, gt_label_dict, confs)
            else:
                loss_state = get_motion_RMSE(inte_state, gt_label, confs)

            save_state(loss_states, loss_state)
            save_state(evaluate_states, inte_state)
            save_state(labels, label)

            if "cov" in inte_state and inte_state["cov"] is not None:
                pred_cov.append(inte_state["cov"])
        
        if confs.propcov:
            if inte_state is None:
                raise ValueError("Error: evaluate dataset is too small to compute.")
            else:
                cov = torch.cat(pred_cov, dim=-2)
        else:
            cov = torch.tensor(0.0, device=confs.device)
            skip_key = "cov_loss"


        for k, v in loss_states.items():
            if skip_key and k == skip_key:
                continue
            loss_states[k] = torch.stack(v, dim=0)

        cat_state(evaluate_states)
        cat_state(labels)

        print("evaluating: vel losses %f, evaluation cov %f" % (loss_states['loss'].mean(), cov.mean()))


    return {
        "evaluate": evaluate_states,
        "evaluate_cov": cov,
        "loss": loss_states,
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/EuRoC/motion_body.conf",
        help="config file path",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="cuda or cpu, Default is cuda:0"
    )
    parser.add_argument('--precision', choices=('float32', 'float64'), default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--cpu-threads', type=int, default=None)
    parser.add_argument('--quiet', action='store_true', help='hide per-batch progress bars')
    parser.add_argument(
        "--load_ckpt",
        default=False,
        action="store_true",
        help="If True, try to load the newest.ckpt in the \
                                                                                exp_dir specificed in our config file.",
    )
    parser.add_argument(
        "--log",
        default=True,
        action="store_false",
        help="if True, save the meta data with wandb",
    )

    args = parser.parse_args()
    print(args)
    conf = ConfigFactory.parse_file(args.config)

    conf.train.device = args.device
    conf.train['precision'] = args.precision or conf.train.get('precision', 'float32')
    conf.train['seed'] = args.seed if args.seed is not None else int(conf.train.get('seed', 42))
    conf.train['quiet'] = args.quiet or bool(conf.train.get('quiet', False))
    if args.num_workers is not None:
        conf.train['num_workers'] = args.num_workers
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    cpu_threads = args.cpu_threads or int(conf.train.get('cpu_threads', 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    np.random.seed(conf.train.seed)
    torch.manual_seed(conf.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.train.seed)
    exp_folder = os.path.split(conf.general.exp_dir)[-1]
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf["general"]["exp_dir"] = os.path.join(conf.general.exp_dir, conf_name)
    if "gravity" in conf.dataset.train:
        gravity = conf.dataset.train.gravity
        conf.train.put("gravity", conf.dataset.train.gravity)
    else:
        gravity = 9.81007

    train_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.train)
    test_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.test)
    eval_dataset = SeqeuncesMotionDataset(data_set_config=conf.dataset.eval)

    if "collate" in conf.dataset.keys():
        collate_fn_train, collate_fn_test = collate_fcs[conf.dataset.collate.type], collate_fcs[conf.dataset.collate.type]
    else:
        collate_fn_train, collate_fn_test = collate_fcs["base"], collate_fcs["base"]

    
    num_workers = int(conf.train.get('num_workers', 0))
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': args.device.startswith('cuda'),
        'persistent_workers': num_workers > 0,
    }
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=conf.train.batch_size,
        shuffle = True,
        collate_fn=collate_fn_train,
        **loader_kwargs,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        **loader_kwargs,
    )
    eval_loader = Data.DataLoader(
        dataset=eval_dataset,
        batch_size=conf.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        drop_last=True,
        **loader_kwargs,
    )


    os.makedirs(os.path.join(conf.general.exp_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(conf.general.exp_dir, "parameters.yaml"), "w") as f:
        f.write(conf_convert.to_yaml(conf))

    if not args.log:
        wandb.disabled = True
        print("wandb is disabled")
    else:
        wandb.init(
            project="AirIO" + exp_folder,
            config=conf.train,
            group=conf.train.network,
            name=conf_name,
        )

    ## optimizer and network
    network = net_dict[conf.train.network](conf.train).to(
        device=args.device, dtype=resolve_dtype(conf.train.precision)
    )
    optimizer = torch.optim.Adam(
        network.parameters(), lr=conf.train.lr, weight_decay=conf.train.weight_decay
    )  # to use with ViTs
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.train.factor,
        patience=conf.train.patience,
        min_lr=conf.train.min_lr,
    )
    best_loss = np.inf
    best_eval_loss = np.inf  # 用于保存eval最佳模型
    epoch = 0
    
    # 创建 TensorBoard writer
    tensorboard_dir = os.path.join(conf.general.exp_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    ## load the chkp if there exist
    if args.load_ckpt:
        if os.path.isfile(os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt")):
            checkpoint = torch.load(
                os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"),
                map_location=args.device,
                weights_only=True
            )
            network.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # Checkpoints store the last completed epoch. Resume from the next
            # one so interrupted full reproductions do not train an epoch twice.
            epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            best_eval_loss = checkpoint.get("best_eval_loss", np.inf)  # 兼容旧版本
            print(
                "loaded state dict %s best_loss %f best_eval_loss %f"
                % (os.path.join(conf.general.exp_dir, "ckpt/newest.ckpt"), best_loss, best_eval_loss)
            )
        else:
            print("Can't find the checkpoint")

    for epoch_i in range(epoch, conf.train.max_epoches):
        train_loss = train(network, train_loader, conf.train, epoch_i, optimizer, writer)
        test_loss = test(network, test_loader, conf.train)
        print("train loss: %f test loss: %f" % (train_loss["loss"], test_loss["loss"]))

        # save the training meta information
        if args.log:
            write_wandb("train", train_loss, epoch_i)
            write_wandb("test", test_loss, epoch_i)
            write_wandb("lr", scheduler.optimizer.param_groups[0]["lr"], epoch_i)
        
        # 记录详细损失到 TensorBoard
        if 'loss_logger' in train_loss:
            train_loss['loss_logger'].log_to_tensorboard("train_detailed", epoch_i)
        save_best_test = False
        save_best_eval = False
        
        if epoch_i % conf.train.eval_freq == conf.train.eval_freq - 1:
            eval_state = evaluate(network=network, loader=eval_loader, confs=conf.train)
            eval_loss = eval_state['loss']['loss'].mean().item()
            
            if args.log:
                write_wandb('eval/loss', eval_loss, epoch_i)
                write_wandb('eval/dist', eval_state['loss']['dist'].mean(), epoch_i)
            if "supervise_pos" in conf.train:
                print("eval pos: %f " % eval_loss)
            else:
                print("eval vel: %f " % eval_loss)
            
            # 检查是否是eval最佳模型
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_best_eval = True
                print("saving the best eval model", best_eval_loss)
                torch.save({
                    'epoch': epoch_i,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'best_eval_loss': best_eval_loss,
                }, os.path.join(conf.general.exp_dir, "ckpt/best_eval_model.ckpt"))

        scheduler.step(test_loss["loss"])
        if test_loss["loss"] < best_loss:
            best_loss = test_loss["loss"]
            save_best_test = True

        save_ckpt(
            network,
            optimizer,
            scheduler,
            epoch_i,
            best_loss,
            conf,
            save_best=save_best_test,
            best_eval_loss=best_eval_loss,
        )

    wandb.finish()
    writer.close()
