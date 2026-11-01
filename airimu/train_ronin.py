"""Train the phase-aware RoNIN horizontal-velocity adaptation."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory, HOCONConverter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.RoNINdataset import (
    RoNINWindowDataset,
    read_ronin_sequence_names,
    split_available_ronin_sequences,
)
from model.model_ronin_phase_aware import PhaseAwareVelocityModel
from utils import resolve_dtype


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def velocity_loss(prediction, target, crop=0, beta=0.1):
    if crop > 0:
        if prediction.shape[1] <= 2 * crop:
            raise ValueError('loss_crop removes the entire prediction window')
        prediction = prediction[:, crop:-crop]
        target = target[:, crop:-crop]
    return F.smooth_l1_loss(prediction, target, beta=beta)


def run_epoch(model, loader, device, dtype, crop, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    squared_error = 0.0
    absolute_error = 0.0
    value_count = 0

    progress = tqdm(loader, leave=False)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in progress:
            features = batch['features'].to(device=device, dtype=dtype)
            target = batch['velocity'].to(device=device, dtype=dtype)
            prediction = model(features)
            loss = velocity_loss(prediction, target, crop=crop, beta=beta)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            error = prediction.detach() - target
            total_loss += loss.item() * features.shape[0]
            squared_error += error.square().sum().item()
            absolute_error += error.abs().sum().item()
            value_count += error.numel()
            progress.set_description(f"{'train' if training else 'valid'} loss {loss.item():.5f}")

    return {
        'loss': total_loss / len(loader.dataset),
        'velocity_rmse': (squared_error / value_count) ** 0.5,
        'velocity_mae': absolute_error / value_count,
    }


def trajectory_metrics(model, dataset, device, dtype, batch_size=16):
    """Overlap-average window predictions and integrate horizontal velocity."""
    model.eval()
    velocity_sum = [np.zeros_like(seq.velocity) for seq in dataset.sequences]
    velocity_count = [np.zeros((len(seq.velocity), 1), dtype=np.float32)
                      for seq in dataset.sequences]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device=device, dtype=dtype)
            predictions = model(features).cpu().float().numpy()
            for prediction, sequence_id, start in zip(
                    predictions, batch['sequence_id'].tolist(), batch['start_frame'].tolist()):
                end = start + len(prediction)
                velocity_sum[sequence_id][start:end] += prediction
                velocity_count[sequence_id][start:end] += 1

    sequence_ate = {}
    for index, sequence in enumerate(dataset.sequences):
        covered = velocity_count[index][:, 0] > 0
        covered_indices = np.flatnonzero(covered)
        if len(covered_indices) < 2:
            continue
        # A capped diagnostic dataset can contain disjoint windows. Integrate
        # every contiguous covered region independently and pool its errors.
        regions = np.split(
            covered_indices, np.flatnonzero(np.diff(covered_indices) > 1) + 1
        )
        errors = []
        for region in regions:
            if len(region) < 2:
                continue
            first, last = region[0], region[-1] + 1
            prediction = (
                velocity_sum[index][first:last] / velocity_count[index][first:last]
            )
            time = sequence.time[first:last]
            gt_position = sequence.position[first:last, :2]
            predicted_position = np.empty_like(gt_position)
            predicted_position[0] = gt_position[0]
            increments = prediction[:-1] * np.diff(time)[:, None]
            predicted_position[1:] = (
                predicted_position[0] + np.cumsum(increments, axis=0)
            )
            errors.append(predicted_position - gt_position)
        if errors:
            error = np.concatenate(errors, axis=0)
            sequence_ate[dataset.sequence_names[index]] = float(
                np.sqrt(np.mean(np.sum(error * error, axis=1)))
            )

    values = list(sequence_ate.values())
    return {
        'ate_mean': float(np.mean(values)) if values else float('nan'),
        'ate_per_sequence': sequence_ate,
    }


def make_dataset(conf, sequence_names, max_sequences=None, max_windows=None):
    return RoNINWindowDataset(
        root_dir=conf.root_dir,
        sequence_names=sequence_names,
        window_size=int(conf.window_size),
        step_size=int(conf.step_size),
        interval=int(conf.get('interval', 1)),
        max_velocity_norm=float(conf.get('max_velocity_norm', 3.0)),
        max_sequences=max_sequences,
        max_windows=max_windows,
        allow_missing=bool(conf.get('allow_missing', False)),
    )


def evaluate_named_split(model, conf, list_file, device, dtype, batch_size,
                         max_sequences=None, max_windows=None):
    requested = read_ronin_sequence_names(list_file)
    root = Path(conf.root_dir)
    available = [name for name in requested if (root / name / 'data.hdf5').is_file()]
    if not available:
        if conf.get('allow_missing', False):
            return None
        raise FileNotFoundError(f'no sequences from {list_file} exist under {root}')
    if len(available) != len(requested) and not conf.get('allow_missing', False):
        raise FileNotFoundError(
            f'{len(requested) - len(available)} sequences from {list_file} are missing'
        )
    dataset = make_dataset(
        conf, available, max_sequences=max_sequences, max_windows=max_windows
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    window_metrics = run_epoch(
        model, loader, device, dtype,
        crop=0,
        beta=0.1,
    )
    trajectory = trajectory_metrics(
        model, dataset, device, dtype, batch_size=batch_size
    )
    return {
        'sequence_count': len(dataset.sequence_names),
        'window_count': len(dataset),
        'window': window_metrics,
        'trajectory': trajectory,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/exp/RoNIN/phase_aware_velocity.conf')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('float32', 'float64'))
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--load-ckpt', action='store_true')
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.config)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; use --device cpu')
    precision = args.precision or conf.train.get('precision', 'float32')
    dtype = resolve_dtype(precision)
    cpu_threads = args.cpu_threads or int(conf.train.get('cpu_threads', 0))
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    seed = int(conf.train.get('seed', 42))
    set_seed(seed)

    train_names, validation_names = split_available_ronin_sequences(
        conf.dataset.root_dir,
        conf.dataset.train_list,
        conf.dataset.validation_list,
        allow_missing=bool(conf.dataset.get('allow_missing', False)),
        validation_fraction=float(conf.dataset.get('validation_fraction', 0.2)),
    )
    train_dataset = make_dataset(
        conf.dataset,
        train_names,
        max_sequences=conf.dataset.get('max_train_sequences', None),
        max_windows=conf.dataset.get('max_train_windows', None),
    )
    validation_dataset = make_dataset(
        conf.dataset,
        validation_names,
        max_sequences=conf.dataset.get('max_validation_sequences', None),
        max_windows=conf.dataset.get('max_validation_windows', None),
    )
    print(
        f'RoNIN: {len(train_dataset.sequence_names)} train sequences / '
        f'{len(train_dataset)} windows; {len(validation_dataset.sequence_names)} '
        f'validation sequences / {len(validation_dataset)} windows'
    )

    num_workers = args.num_workers if args.num_workers is not None else int(
        conf.train.get('num_workers', 0)
    )
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': args.device.startswith('cuda'),
        'persistent_workers': num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(conf.train.batch_size),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(conf.train.batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model = PhaseAwareVelocityModel(conf.model).to(device=args.device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(conf.train.lr),
        weight_decay=float(conf.train.weight_decay),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(conf.train.get('lr_factor', 0.5)),
        patience=int(conf.train.get('lr_patience', 10)),
        min_lr=float(conf.train.get('min_lr', 1e-6)),
    )

    experiment_dir = Path(conf.general.exp_dir) / Path(args.config).stem
    checkpoint_dir = experiment_dir / 'ckpt'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / 'parameters.yaml').write_text(
        HOCONConverter.to_yaml(conf), encoding='utf-8'
    )
    best_loss = float('inf')
    start_epoch = 0
    newest = checkpoint_dir / 'newest.ckpt'
    if args.load_ckpt and newest.is_file():
        checkpoint = torch.load(newest, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    metrics_path = experiment_dir / 'metrics.json'
    history = []
    if args.load_ckpt and metrics_path.is_file():
        with open(metrics_path, encoding='utf-8') as handle:
            history = json.load(handle).get('history', [])
    for epoch in range(start_epoch, int(conf.train.max_epochs)):
        train_metrics = run_epoch(
            model, train_loader, args.device, dtype,
            crop=int(conf.train.get('loss_crop', 0)),
            beta=float(conf.train.get('huber_beta', 0.1)),
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model, validation_loader, args.device, dtype,
            crop=int(conf.train.get('loss_crop', 0)),
            beta=float(conf.train.get('huber_beta', 0.1)),
        )
        scheduler.step(validation_metrics['loss'])
        record = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': validation_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': min(best_loss, validation_metrics['loss']),
            'precision': precision,
            'seed': seed,
        }
        torch.save(state, newest)
        if validation_metrics['loss'] < best_loss:
            best_loss = validation_metrics['loss']
            torch.save(state, checkpoint_dir / 'best_model.ckpt')

    best_checkpoint = checkpoint_dir / 'best_model.ckpt'
    if best_checkpoint.is_file():
        model.load_state_dict(
            torch.load(best_checkpoint, map_location=args.device)['model_state_dict']
        )

    batch_size = int(conf.train.batch_size)
    trajectory = trajectory_metrics(
        model, validation_dataset, args.device, dtype, batch_size=batch_size
    )
    test_results = {}
    for split_name, list_key in (
        ('seen', 'test_seen_list'),
        ('unseen', 'test_unseen_list'),
    ):
        list_file = conf.dataset.get(list_key, None)
        if list_file:
            metrics = evaluate_named_split(
                model, conf.dataset, list_file, args.device, dtype, batch_size,
                max_sequences=conf.dataset.get(
                    f'max_test_{split_name}_sequences', None
                ),
                max_windows=conf.dataset.get(f'max_test_{split_name}_windows', None),
            )
            if metrics is not None:
                test_results[split_name] = metrics
    result = {
        'history': history,
        'validation_trajectory': trajectory,
        'test': test_results,
    }
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(trajectory, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
