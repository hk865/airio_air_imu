import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def _rotation_from_wxyz(values):
    values = np.asarray(values)
    return Rotation.from_quat(values[..., [1, 2, 3, 0]])


def read_ronin_sequence_names(list_file):
    with open(list_file, encoding='utf-8') as handle:
        return [
            line.strip() for line in handle
            if line.strip() and not line.lstrip().startswith('#')
        ]


class RoNINSequence:
    """Load one official RoNIN ``data.hdf5``/``info.json`` sequence.

    This follows the official ``GlobSpeedSequence`` feature definition without
    requiring ``numpy-quaternion``: calibrated gyroscope and acceleration are
    rotated into the Tango/world frame, and horizontal body velocity is derived
    from the harness-mounted Tango trajectory.
    """

    REQUIRED_HDF5 = (
        'synced/time',
        'synced/gyro_uncalib',
        'synced/acce',
        'synced/game_rv',
        'pose/tango_pos',
        'pose/tango_ori',
    )

    def __init__(self, sequence_dir, interval=1):
        self.path = Path(sequence_dir)
        if interval < 1:
            raise ValueError('interval must be positive')
        with open(self.path / 'info.json', encoding='utf-8') as handle:
            self.info = json.load(handle)

        with h5py.File(self.path / 'data.hdf5', 'r') as hdf:
            missing = [key for key in self.REQUIRED_HDF5 if key not in hdf]
            if missing:
                raise KeyError(f'{self.path}: missing HDF5 fields {missing}')
            time = np.asarray(hdf['synced/time'], dtype=np.float64)
            gyro_uncalib = np.asarray(hdf['synced/gyro_uncalib'], dtype=np.float64)
            acceleration_uncalib = np.asarray(hdf['synced/acce'], dtype=np.float64)
            game_rv = np.asarray(hdf['synced/game_rv'], dtype=np.float64)
            tango_position = np.asarray(hdf['pose/tango_pos'], dtype=np.float64)
            tango_orientation = np.asarray(hdf['pose/tango_ori'], dtype=np.float64)

        lengths = {
            len(time), len(gyro_uncalib), len(acceleration_uncalib),
            len(game_rv), len(tango_position), len(tango_orientation),
        }
        if len(lengths) != 1:
            raise ValueError(f'{self.path}: inconsistent synchronized stream lengths {lengths}')
        if len(time) <= interval:
            raise ValueError(f'{self.path}: sequence is shorter than interval={interval}')
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
            raise ValueError(f'{self.path}: timestamps must be finite and strictly increasing')

        gyro_bias = np.asarray(self.info['imu_init_gyro_bias'], dtype=np.float64)
        acceleration_bias = np.asarray(self.info['imu_acce_bias'], dtype=np.float64)
        acceleration_scale = np.asarray(self.info['imu_acce_scale'], dtype=np.float64)
        gyro = gyro_uncalib - gyro_bias
        acceleration = acceleration_scale * (acceleration_uncalib - acceleration_bias)

        # Match the official quaternion composition (stored order: w, x, y, z).
        imu_orientation = _rotation_from_wxyz(game_rv)
        initial_tango = _rotation_from_wxyz(tango_orientation[0])
        imu_to_tango = _rotation_from_wxyz(self.info['start_calibration'])
        initial_rotor = initial_tango * imu_to_tango * imu_orientation[0].inv()
        world_orientation = initial_rotor * imu_orientation

        global_gyro = world_orientation.apply(gyro)
        global_acceleration = world_orientation.apply(acceleration)
        features = np.concatenate([global_gyro, global_acceleration], axis=1)

        delta_time = time[interval:] - time[:-interval]
        velocity = (
            (tango_position[interval:] - tango_position[:-interval])
            / delta_time[:, None]
        )

        start = int(self.info.get('start_frame', 0))
        stop = len(time) - interval
        if start >= stop:
            raise ValueError(f'{self.path}: start_frame={start} leaves no usable data')
        self.time = time[start:stop]
        self.features = features[start:stop].astype(np.float32)
        self.velocity = velocity[start:, :2].astype(np.float32)
        self.position = tango_position[start:stop].astype(np.float32)
        self.orientation_wxyz = np.asarray(
            world_orientation.as_quat()[start:stop][..., [3, 0, 1, 2]],
            dtype=np.float32,
        )

        arrays = (self.features, self.velocity, self.position, self.orientation_wxyz)
        if any(len(array) != len(self.time) for array in arrays):
            raise RuntimeError(f'{self.path}: aligned arrays have inconsistent lengths')
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError(f'{self.path}: sequence contains non-finite values')


class RoNINWindowDataset(Dataset):
    """Overlapping sequence-to-sequence windows for horizontal velocity."""

    def __init__(self, root_dir, sequence_names=None, list_file=None, window_size=400,
                 step_size=100, interval=1, max_velocity_norm=3.0,
                 max_sequences=None, max_windows=None, allow_missing=False):
        self.root_dir = Path(root_dir)
        if sequence_names is None:
            if list_file is None:
                sequence_names = sorted(
                    path.name for path in self.root_dir.iterdir()
                    if path.is_dir() and (path / 'data.hdf5').is_file()
                )
            else:
                sequence_names = read_ronin_sequence_names(list_file)

        available, missing = [], []
        for name in sequence_names:
            path = self.root_dir / name
            if (path / 'data.hdf5').is_file() and (path / 'info.json').is_file():
                available.append(name)
            else:
                missing.append(name)
        if missing and not allow_missing:
            preview = ', '.join(missing[:5])
            raise FileNotFoundError(
                f'{len(missing)} RoNIN sequences are missing under {self.root_dir}: {preview}'
            )
        if max_sequences is not None:
            available = available[:int(max_sequences)]
        if not available:
            raise FileNotFoundError(f'no usable RoNIN sequences found under {self.root_dir}')

        self.sequence_names = available
        self.missing_names = missing
        self.sequences = [
            RoNINSequence(self.root_dir / name, interval=interval)
            for name in available
        ]
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.index_map = []
        for sequence_id, sequence in enumerate(self.sequences):
            bad = np.linalg.norm(sequence.velocity, axis=1) > float(max_velocity_norm)
            for end in range(self.window_size, len(sequence.features) + 1, self.step_size):
                start = end - self.window_size
                if not bad[start:end].any():
                    self.index_map.append((sequence_id, start, end))
        if not self.index_map:
            raise ValueError('no valid RoNIN windows remain after filtering')
        if max_windows is not None and len(self.index_map) > int(max_windows):
            # Deterministically cover the complete sequence list instead of
            # taking every capped window from the first (usually long) record.
            selected = np.linspace(
                0, len(self.index_map) - 1, num=int(max_windows), dtype=np.int64
            )
            self.index_map = [self.index_map[index] for index in selected]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        sequence_id, start, end = self.index_map[index]
        sequence = self.sequences[sequence_id]
        return {
            'features': sequence.features[start:end],
            'velocity': sequence.velocity[start:end],
            'sequence_id': np.int64(sequence_id),
            'start_frame': np.int64(start),
        }


def split_available_ronin_sequences(root_dir, train_list, validation_list,
                                    allow_missing=False, validation_fraction=0.2):
    """Resolve official splits; optionally derive validation from a partial archive."""
    root_dir = Path(root_dir)
    train_names = [
        name for name in read_ronin_sequence_names(train_list)
        if (root_dir / name / 'data.hdf5').is_file()
    ]
    validation_names = [
        name for name in read_ronin_sequence_names(validation_list)
        if (root_dir / name / 'data.hdf5').is_file()
    ]
    if validation_names:
        return train_names, validation_names
    if not allow_missing:
        raise FileNotFoundError('none of the official RoNIN validation sequences are available')
    if len(train_names) < 2:
        raise ValueError('at least two available training sequences are needed for a fallback split')
    validation_count = max(1, round(len(train_names) * validation_fraction))
    return train_names[:-validation_count], train_names[-validation_count:]
