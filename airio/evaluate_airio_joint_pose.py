#!/usr/bin/env python3
"""Joint ATE/AOE evaluation for AirIO velocity measurements fused by an ESKF.

This evaluator intentionally does not use ``EKF/IMUofflinerunner.py``. That
runner mixes pre- and post-propagation Jacobians and computes 1.5*a*dt**2 in
the position update. Here the nominal state and covariance follow the AirIO
paper's state (R, v, p, b_a, b_g), with body-frame velocity observations.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pypose as pp
import torch
from pyhocon import ConfigFactory
from scipy.spatial.transform import Rotation

from datasets import SeqInfDataset


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return CPUUnpickler(handle).load()


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use NAME=/path/to/experiment")
    name, path = value.split("=", 1)
    return name, Path(path)


def nearest_measurement_indices(
    imu_times: np.ndarray, measurement_times: np.ndarray, tolerance: float
) -> np.ndarray:
    """Map each IMU time to a measurement index, or -1 when unmatched."""
    measurement_times = measurement_times.reshape(-1)
    result = np.full(len(imu_times), -1, dtype=np.int64)
    right = np.searchsorted(measurement_times, imu_times)
    for i, candidate in enumerate(right):
        choices = []
        if candidate < len(measurement_times):
            choices.append(candidate)
        if candidate > 0:
            choices.append(candidate - 1)
        if not choices:
            continue
        best = min(choices, key=lambda j: abs(measurement_times[j] - imu_times[i]))
        if abs(measurement_times[best] - imu_times[i]) <= tolerance:
            result[i] = best
    return result


def run_eskf(
    dataset: SeqInfDataset,
    io_result: dict[str, Any],
    imu_cov_scale: float,
    obs_cov_scale: float,
    bias_random_walk: float,
    timestamp_tolerance: float,
    update_times: np.ndarray | None = None,
) -> dict[str, Any]:
    data = dataset.data
    times = as_numpy(data["time"])
    dt_all = as_numpy(data["dt"]).reshape(-1)
    accelerometer = as_numpy(data["acc"][:-1])
    gyroscope = as_numpy(data["gyro"][:-1])
    accelerometer_cov = as_numpy(data["acc_cov"])
    gyroscope_cov = as_numpy(data["gyro_cov"])

    gt_position = as_numpy(data["gt_translation"])
    gt_velocity = as_numpy(data["velocity"])
    # PyPose stores SO(3) quaternions in scipy's x,y,z,w ordering.
    gt_quaternion_xyzw = as_numpy(data["gt_orientation"])
    gt_rotation = Rotation.from_quat(gt_quaternion_xyzw).as_matrix()

    observation_velocity = as_numpy(io_result["net_vel"])
    observation_cov = as_numpy(io_result["cov"])[0]
    observation_times = as_numpy(io_result["ts"]).reshape(-1)
    if update_times is not None:
        selected = nearest_measurement_indices(
            as_numpy(update_times).reshape(-1), observation_times, timestamp_tolerance
        )
        selected = selected[selected >= 0]
        observation_velocity = observation_velocity[selected]
        observation_cov = observation_cov[selected]
        observation_times = observation_times[selected]
    measurement_at = nearest_measurement_indices(
        times[1:], observation_times, timestamp_tolerance
    )

    rotation = gt_rotation[0].copy()
    velocity = gt_velocity[0].copy()
    position = gt_position[0].copy()
    gyro_bias = np.zeros(3)
    accelerometer_bias = np.zeros(3)
    covariance = np.zeros((15, 15))
    gravity = np.array([0.0, 0.0, float(dataset.get_gravity())])

    estimated_position = np.empty((len(dt_all), 3))
    estimated_velocity = np.empty((len(dt_all), 3))
    estimated_rotation = np.empty((len(dt_all), 3, 3))
    matched_updates = 0

    identity3 = np.eye(3)
    identity15 = np.eye(15)

    for i, dt in enumerate(dt_all):
        old_rotation = rotation
        omega = gyroscope[i] - gyro_bias
        specific_force = accelerometer[i] - accelerometer_bias
        world_acceleration = old_rotation @ specific_force - gravity

        # Nominal propagation to timestamp i+1.
        position = position + velocity * dt + 0.5 * world_acceleration * dt * dt
        velocity = velocity + world_acceleration * dt
        rotation = old_rotation @ Rotation.from_rotvec(omega * dt).as_matrix()

        # Right-multiplicative error-state covariance propagation.
        transition = identity15.copy()
        transition[0:3, 0:3] -= skew(omega) * dt
        transition[0:3, 9:12] = -identity3 * dt
        transition[3:6, 0:3] = -old_rotation @ skew(specific_force) * dt
        transition[3:6, 12:15] = -old_rotation * dt
        transition[6:9, 0:3] = -0.5 * old_rotation @ skew(specific_force) * dt * dt
        transition[6:9, 3:6] = identity3 * dt
        transition[6:9, 12:15] = -0.5 * old_rotation * dt * dt

        noise_jacobian = np.zeros((15, 12))
        noise_jacobian[0:3, 0:3] = -identity3 * dt
        noise_jacobian[3:6, 3:6] = -old_rotation * dt
        noise_jacobian[6:9, 3:6] = -0.5 * old_rotation * dt * dt
        noise_jacobian[9:12, 6:9] = identity3 * dt
        noise_jacobian[12:15, 9:12] = identity3 * dt

        process_variance = np.concatenate(
            [
                np.maximum(gyroscope_cov[i], 1e-15),
                np.maximum(accelerometer_cov[i] * imu_cov_scale, 1e-15),
                np.full(6, bias_random_walk),
            ]
        )
        covariance = (
            transition @ covariance @ transition.T
            + noise_jacobian @ np.diag(process_variance) @ noise_jacobian.T
        )

        measurement_index = measurement_at[i]
        if measurement_index >= 0:
            predicted_body_velocity = rotation.T @ velocity
            observation_jacobian = np.zeros((3, 15))
            observation_jacobian[:, 0:3] = skew(predicted_body_velocity)
            observation_jacobian[:, 3:6] = rotation.T
            measurement_variance = np.maximum(
                observation_cov[measurement_index] * obs_cov_scale, 1e-12
            )
            innovation_cov = (
                observation_jacobian @ covariance @ observation_jacobian.T
                + np.diag(measurement_variance)
            )
            gain = np.linalg.solve(
                innovation_cov,
                observation_jacobian @ covariance,
            ).T
            innovation = observation_velocity[measurement_index] - predicted_body_velocity
            delta = gain @ innovation

            rotation = rotation @ Rotation.from_rotvec(delta[0:3]).as_matrix()
            velocity += delta[3:6]
            position += delta[6:9]
            gyro_bias += delta[9:12]
            accelerometer_bias += delta[12:15]

            correction = identity15 - gain @ observation_jacobian
            covariance = (
                correction @ covariance @ correction.T
                + gain @ np.diag(measurement_variance) @ gain.T
            )
            reset = identity15.copy()
            reset[0:3, 0:3] -= 0.5 * skew(delta[0:3])
            covariance = reset @ covariance @ reset.T
            matched_updates += 1

        covariance = 0.5 * (covariance + covariance.T)
        estimated_position[i] = position
        estimated_velocity[i] = velocity
        estimated_rotation[i] = rotation

    position_error = estimated_position - gt_position[1 : len(dt_all) + 1]
    velocity_error = estimated_velocity - gt_velocity[1 : len(dt_all) + 1]
    relative_rotation = np.einsum(
        "nij,njk->nik",
        np.transpose(gt_rotation[1 : len(dt_all) + 1], (0, 2, 1)),
        estimated_rotation,
    )
    orientation_error_deg = np.degrees(Rotation.from_matrix(relative_rotation).magnitude())

    metrics = {
        "frames": int(len(dt_all)),
        "duration_s": float(times[len(dt_all)] - times[0]),
        "velocity_updates": int(matched_updates),
        "velocity_measurements": int(len(observation_times)),
        "ate_rmse_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))),
        "aoe_rmse_deg": float(np.sqrt(np.mean(orientation_error_deg**2))),
        "aoe_mean_deg": float(np.mean(orientation_error_deg)),
        "aoe_median_deg": float(np.median(orientation_error_deg)),
        "final_position_error_m": float(np.linalg.norm(position_error[-1])),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=1)))),
    }
    trajectory = {
        "timestamp": times[1 : len(dt_all) + 1],
        "position": estimated_position,
        "velocity": estimated_velocity,
        "quaternion_xyzw": Rotation.from_matrix(estimated_rotation).as_quat(),
        "position_error": position_error,
        "orientation_error_deg": orientation_error_deg,
    }
    return {"metrics": metrics, "trajectory": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/EuRoC/Euroc_body.conf",
    )
    parser.add_argument(
        "--dataset-section",
        choices=("train", "test", "eval", "inference"),
        default="inference",
    )
    parser.add_argument("--airimu-exp", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, type=parse_model)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imu-cov-scale", type=float, default=100.0)
    parser.add_argument("--obs-cov-scale", type=float, default=0.1)
    parser.add_argument("--bias-random-walk", type=float, default=1e-12)
    parser.add_argument("--timestamp-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--update-times-from-model",
        default=None,
        help="use this model's timestamps as a common EKF update schedule",
    )
    args = parser.parse_args()

    config = ConfigFactory.parse_file(args.dataset_config)[args.dataset_section]
    # Even when selecting the training sequence list, this is an inference-time
    # filter evaluation. Keeping mode="train" would rotate the GT velocity into
    # the body frame inside Euroc.update_coordinate(), corrupting initialization
    # and velocity metrics.
    config["mode"] = "inference"
    airimu = load_pickle(args.airimu_exp / "net_output.pickle")
    models = {
        name: load_pickle(path / "net_output.pickle") for name, path in args.model
    }
    if args.update_times_from_model is not None and args.update_times_from_model not in models:
        parser.error("--update-times-from-model must name one of the supplied --model entries")
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "metric_definition": {
            "ATE": "sqrt(mean(||p_est - p_gt||_2^2)), no trajectory alignment",
            "AOE": "sqrt(mean(angle(R_gt^-1 R_est)^2)), degrees",
        },
        "initialization": "ground-truth pose and velocity at the first frame",
        "fusion": "right-error-state EKF; AirIMU corrected IMU and learned covariance; AirIO body-frame velocity and learned covariance",
        "parameters": {
            "imu_cov_scale": args.imu_cov_scale,
            "obs_cov_scale": args.obs_cov_scale,
            "bias_random_walk": args.bias_random_walk,
            "timestamp_tolerance_s": args.timestamp_tolerance,
            "update_times_from_model": args.update_times_from_model,
        },
        "models": {},
    }

    for model_name, model_output in models.items():
        per_sequence: dict[str, Any] = {}
        estimated_orientation: dict[str, Any] = {}
        print(f"\n[{model_name}]")
        for data_config in config.data_list:
            for sequence in data_config.data_drive:
                dataset = SeqInfDataset(
                    data_config.data_root,
                    sequence,
                    airimu[sequence],
                    device="cpu",
                    name=data_config.name,
                    duration=1,
                    step_size=1,
                    drop_last=False,
                    conf=config,
                )
                result = run_eskf(
                    dataset,
                    model_output[sequence],
                    args.imu_cov_scale,
                    args.obs_cov_scale,
                    args.bias_random_walk,
                    args.timestamp_tolerance,
                    (
                        models[args.update_times_from_model][sequence]["ts"]
                        if args.update_times_from_model is not None
                        else None
                    ),
                )
                per_sequence[sequence] = result["metrics"]
                np.savez_compressed(
                    args.output / f"{model_name}_{sequence}_trajectory.npz",
                    **result["trajectory"],
                )
                initial_quaternion = as_numpy(dataset.data["gt_orientation"][:1])
                full_quaternion = np.concatenate(
                    [initial_quaternion, result["trajectory"]["quaternion_xyzw"]], axis=0
                )
                estimated_orientation[sequence] = {
                    "airimu_rot": pp.SO3(torch.from_numpy(full_quaternion).double())
                }
                print(
                    f"  {sequence}: ATE={result['metrics']['ate_rmse_m']:.4f} m, "
                    f"AOE={result['metrics']['aoe_rmse_deg']:.4f} deg, "
                    f"updates={result['metrics']['velocity_updates']}"
                )

        numeric_keys = [
            "ate_rmse_m",
            "aoe_rmse_deg",
            "aoe_mean_deg",
            "aoe_median_deg",
            "final_position_error_m",
            "velocity_rmse_mps",
        ]
        average = {
            key: float(np.mean([metrics[key] for metrics in per_sequence.values()]))
            for key in numeric_keys
        }
        report["models"][model_name] = {
            "sequences": per_sequence,
            "sequence_average": average,
        }
        with (args.output / f"{model_name}_estimated_orientation.pickle").open("wb") as handle:
            pickle.dump(estimated_orientation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"  AVG: ATE={average['ate_rmse_m']:.4f} m, "
            f"AOE={average['aoe_rmse_deg']:.4f} deg"
        )

    output_json = args.output / "joint_ate_aoe_metrics.json"
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved {output_json}")


if __name__ == "__main__":
    main()
