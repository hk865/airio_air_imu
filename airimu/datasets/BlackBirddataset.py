"""Blackbird IMU/ground-truth loader adapted from the official AirIO loader."""

import os

import numpy as np
import pypose as pp
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from .dataset import Sequence


class BlackBird(Sequence):
    """Load one Blackbird flight and resample it to a uniform 100 Hz IMU grid."""

    def __init__(self, data_root, data_name, gravity=9.81007, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.gravity = gravity
        self.data = {}

        data_path = os.path.join(data_root, data_name)
        raw_imu = np.loadtxt(
            os.path.join(data_path, 'imu_data.csv'), dtype=float, delimiter=','
        )
        thrust = np.loadtxt(
            os.path.join(data_path, 'thrust_data.csv'), dtype=float, delimiter=','
        )
        ground_truth = np.loadtxt(
            os.path.join(data_path, 'groundTruthPoses.csv'), dtype=float, delimiter=','
        )
        self._prepare(raw_imu, thrust, ground_truth)
        print('loaded: ', data_path, 'rate: 100 Hz')

    def _prepare(self, raw_imu, thrust, ground_truth):
        # Ground truth is drone-body pose in an NED Vicon frame. Convert to a
        # z-up world frame and from drone-body axes to the IMU axes.
        world_from_ned = np.diag([1.0, -1.0, -1.0])
        body_to_imu = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        gt_time = ground_truth[:, 0] / 1e6
        gt_position = (world_from_ned @ ground_truth[:, 1:4].T).T
        body_rotation = Rotation.from_quat(
            ground_truth[:, [5, 6, 7, 4]]
        ).as_matrix()
        imu_rotation = world_from_ned[None] @ body_rotation @ body_to_imu[None]
        gt_quaternion = Rotation.from_matrix(imu_rotation).as_quat()

        raw_velocity = np.gradient(gt_position, gt_time, axis=0)
        kernel = np.ones(5) / 5
        gt_velocity = np.stack([
            np.convolve(raw_velocity[:, axis], kernel, mode='same')
            for axis in range(3)
        ], axis=1)

        # The release CSV header is misleading: columns 1:4 are angular rate
        # and columns 4:7 are acceleration, matching the official AirIO loader.
        new_time = np.arange(raw_imu[0, 0], raw_imu[-1, 0], 0.01)
        start = max(new_time[0], gt_time[0], thrust[0, 0])
        end = min(new_time[-1], gt_time[-1], thrust[-1, 0])
        time = new_time[(new_time >= start) & (new_time <= end)]
        if time.size < 2:
            raise ValueError(f'No overlapping IMU/ground-truth interval for {self.data_name}')

        gyro = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(time)
        acceleration = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(time)
        position = interp1d(gt_time, gt_position, axis=0)(time)
        velocity = interp1d(gt_time, gt_velocity, axis=0)(time)
        orientation = Slerp(gt_time, Rotation.from_quat(gt_quaternion))(time).as_quat()

        self.data['time'] = torch.from_numpy(time)
        self.data['gt_time'] = self.data['time']
        self.data['dt'] = torch.from_numpy(np.diff(time)[:, None])
        self.data['gyro'] = torch.from_numpy(gyro)
        self.data['acc'] = torch.from_numpy(acceleration)
        self.data['gt_translation'] = torch.from_numpy(position)
        self.data['velocity'] = torch.from_numpy(velocity)
        self.data['gt_orientation'] = pp.SO3(torch.from_numpy(orientation))
        self.data['mask'] = torch.ones(time.shape[0], dtype=torch.bool)

    def get_length(self):
        return self.data['time'].shape[0]
