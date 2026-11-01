#!/usr/bin/env python3
"""Compare the reproduced AirIMU CodeNet and wavelet/phase model.

The script uses the saved free-integration trajectory metrics and inference
outputs.  It reports both arithmetic means and sequence-balanced geometric
changes so that one long, high-error sequence cannot silently dominate the
conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data_utils
from pyhocon import ConfigFactory
from scipy.signal import welch

from datasets import SeqeuncesDataset, collate_fcs
from model import net_dict
from utils import CPU_Unpickler


UNSEEN = [
    "MH_02_easy",
    "MH_04_difficult",
    "V1_03_difficult",
    "V2_02_medium",
    "V1_01_easy",
]
TRAIN = [
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
    "MH_03_medium",
    "MH_05_difficult",
    "MH_01_easy",
]
GLOBAL_METRICS = ["ATE(AirIMU)", "AVE(AirIMU)", "O_RMSE(AirIMU)"]
RELATIVE_METRICS = [
    "RP_RMSE(AirIMU)",
    "RV_RMSE(AirIMU)",
    "RO_RMSE(AirIMU)",
]


def load_json_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as handle:
        return {row["name"]: row for row in json.load(handle)}


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mean_metrics(rows, sequences, metrics):
    return {
        metric: float(np.mean([rows[seq][metric] for seq in sequences]))
        for metric in metrics
    }


def median_metrics(rows, sequences, metrics):
    return {
        metric: float(np.median([rows[seq][metric] for seq in sequences]))
        for metric in metrics
    }


def percent_change(after, before):
    return 100.0 * (after / before - 1.0)


def geometric_percent_change(baseline, phase, sequences, metric):
    ratios = [phase[seq][metric] / baseline[seq][metric] for seq in sequences]
    return percent_change(float(np.exp(np.mean(np.log(ratios)))), 1.0)


def band_energy(signal, sample_rate, low, high):
    frequency, density = welch(
        signal, fs=sample_rate, nperseg=min(2048, len(signal))
    )
    mask = (frequency >= low) & (frequency <= high)
    return float(np.trapz(density[mask], frequency[mask]))


def band_fraction(signal, sample_rate, low, high):
    frequency, density = welch(
        signal,
        fs=sample_rate,
        nperseg=min(2048, len(signal)),
        detrend=False,
    )
    mask = (frequency >= low) & (frequency <= high)
    total = float(np.trapz(density, frequency))
    selected = float(np.trapz(density[mask], frequency[mask]))
    return selected / total if total > 0.0 else float("nan")


def correction_spectrum(pickle_path: Path):
    with pickle_path.open("rb") as handle:
        results = CPU_Unpickler(handle).load()

    sequence_rows = []
    for sequence in UNSEEN:
        state = results[sequence]
        correction_acc = tensor_to_numpy(state["correction_acc"]).squeeze(0)
        correction_gyro = tensor_to_numpy(state["correction_gyro"]).squeeze(0)
        corrected_acc = tensor_to_numpy(state["corrected_acc"])
        corrected_gyro = tensor_to_numpy(state["corrected_gyro"])
        raw_acc = corrected_acc - correction_acc
        raw_gyro = corrected_gyro - correction_gyro
        dt = tensor_to_numpy(state["dt"]).reshape(-1)
        sample_rate = float(1.0 / np.median(dt))

        row = {"sequence": sequence}
        for sensor, correction, raw, corrected in (
            ("acc", correction_acc, raw_acc, corrected_acc),
            ("gyro", correction_gyro, raw_gyro, corrected_gyro),
        ):
            row[f"{sensor}_correction_mean"] = np.mean(correction, axis=0).tolist()
            row[f"{sensor}_correction_std"] = np.std(correction, axis=0).tolist()
            row[f"{sensor}_correction_over_raw_std_percent"] = (
                100.0 * np.std(correction, axis=0) / np.std(raw, axis=0)
            ).tolist()
            row[f"{sensor}_40hz_change_db"] = [
                10.0
                * np.log10(
                    (band_energy(corrected[:, channel], sample_rate, 38.0, 42.0) + 1e-30)
                    / (band_energy(raw[:, channel], sample_rate, 38.0, 42.0) + 1e-30)
                )
                for channel in range(3)
            ]
            row[f"{sensor}_correction_below_1hz_fraction"] = [
                band_fraction(correction[:, channel], sample_rate, 0.0, 1.0)
                for channel in range(3)
            ]
            row[f"{sensor}_correction_38_to_42hz_fraction"] = [
                band_fraction(correction[:, channel], sample_rate, 38.0, 42.0)
                for channel in range(3)
            ]
        sequence_rows.append(row)

    keys = [key for key in sequence_rows[0] if key != "sequence"]
    aggregate = {
        key: np.mean([row[key] for row in sequence_rows], axis=0).tolist()
        for key in keys
    }
    return {"sequences": sequence_rows, "mean": aggregate}


def phase_activation_diagnostic(device="cpu", sample_windows=2):
    """Probe whether the trained network actually uses phase features.

    This is an activation/sensitivity diagnostic, not a replacement for a
    retrained no-phase ablation.  It samples the first windows of each unseen
    sequence and measures the output change after zeroing or shifting the
    phase-encoder activations.
    """
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    config = ConfigFactory.parse_file("configs/exp/EuRoC/freqnet_phase_aware.conf")
    config.train.device = device
    config.train.precision = "float32"
    config.train.gtrot = True
    config.train.gtinit = True
    config.train.sampling = False
    model = net_dict[config.train.network](config.train).to(device).float().eval()
    checkpoint = torch.load(
        "experiments/EuRoC/freqnet_phase_aware/ckpt/best_model.ckpt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    collate = collate_fcs[config.dataset.collate]

    results = {}
    with torch.no_grad():
        for data_config in config.dataset.eval.data_list:
            for sequence in data_config.data_drive:
                dataset = SeqeuncesDataset(
                    data_set_config=config.dataset.eval, data_path=sequence
                )
                loader = data_utils.DataLoader(
                    dataset,
                    batch_size=sample_windows,
                    shuffle=False,
                    collate_fn=collate,
                    drop_last=False,
                )
                data, _, _ = next(iter(loader))
                acc = data["acc"].to(device=device, dtype=torch.float32)
                gyro = data["gyro"].to(device=device, dtype=torch.float32)
                feature = torch.cat([acc, gyro], dim=-1)

                wavelet_feature, aux = model.wavelet_branch(feature, return_aux=True)
                deep_feature = model.deep_branch(feature)
                projected_wavelet = model.proj_w(wavelet_feature)
                projected_deep = model.proj_d(deep_feature)
                gate = model.sigmoid(
                    model.gate_linear(
                        torch.cat([projected_wavelet, projected_deep], dim=-1)
                    )
                )

                normal = model.inference({"acc": acc, "gyro": gyro})
                normal_correction = torch.cat(
                    [normal["correction_acc"], normal["correction_gyro"]], dim=-1
                )

                zero_handles = [
                    encoder.register_forward_hook(
                        lambda _module, _inputs, output: torch.zeros_like(output)
                    )
                    for encoder in model.wavelet_branch.phase_encoders
                ]
                zeroed = model.inference({"acc": acc, "gyro": gyro})
                for handle in zero_handles:
                    handle.remove()

                def roll_phase(_module, _inputs, output):
                    shift = max(1, output.shape[-1] // 4)
                    return torch.roll(output, shifts=shift, dims=-1)

                roll_handles = [
                    encoder.register_forward_hook(roll_phase)
                    for encoder in model.wavelet_branch.phase_encoders
                ]
                shifted = model.inference({"acc": acc, "gyro": gyro})
                for handle in roll_handles:
                    handle.remove()

                zeroed_correction = torch.cat(
                    [zeroed["correction_acc"], zeroed["correction_gyro"]], dim=-1
                )
                shifted_correction = torch.cat(
                    [shifted["correction_acc"], shifted["correction_gyro"]], dim=-1
                )
                correction_rms = normal_correction.square().mean((0, 1)).sqrt()
                zero_delta = (
                    (zeroed_correction - normal_correction).square().mean((0, 1)).sqrt()
                )
                shift_delta = (
                    (shifted_correction - normal_correction).square().mean((0, 1)).sqrt()
                )

                results[str(sequence)] = {
                    "scale_weights_detail1_detail2_detail3_detail4_lowpass": (
                        aux["scale_weights"].mean((0, 2)).cpu().tolist()
                    ),
                    "phase_reliability_detail1_to_detail4": (
                        aux["phase_reliability"].mean((0, 2)).cpu().tolist()
                    ),
                    "fusion_gate_mean": float(gate.mean().cpu()),
                    "fusion_gate_std": float(gate.std().cpu()),
                    "phase_zero_delta_over_correction_rms_percent": (
                        100.0 * zero_delta / (correction_rms + 1e-12)
                    ).cpu().tolist(),
                    "phase_shift_delta_over_correction_rms_percent": (
                        100.0 * shift_delta / (correction_rms + 1e-12)
                    ).cpu().tolist(),
                }

    keys = list(next(iter(results.values())))
    aggregate = {
        key: np.mean([row[key] for row in results.values()], axis=0).tolist()
        for key in keys
    }
    return {
        "note": "Sensitivity probe only; a publishable ablation must retrain without phase.",
        "sample_windows_per_sequence": sample_windows,
        "sequences": results,
        "mean": aggregate,
    }


def make_plot(trajectory, output_path: Path):
    baseline_1s = trajectory["1s"]["CodeNet"]
    phase_1s = trajectory["1s"]["WaveletPhase"]
    baseline_5s = trajectory["5s"]["CodeNet"]
    phase_5s = trajectory["5s"]["WaveletPhase"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    metric_labels = ["Full position", "Full velocity", "Full orientation"]
    x = np.arange(len(UNSEEN))
    width = 0.24
    for index, (metric, label) in enumerate(zip(GLOBAL_METRICS, metric_labels)):
        changes = [
            percent_change(phase_1s[seq][metric], baseline_1s[seq][metric])
            for seq in UNSEEN
        ]
        axes[0].bar(x + (index - 1) * width, changes, width, label=label)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, UNSEEN, rotation=15, ha="right")
    axes[0].set_ylabel("Wavelet/phase change vs CodeNet (%)")
    axes[0].set_title("Free-integration full-trajectory errors (lower is better)")
    axes[0].legend(ncols=3, fontsize=9)

    labels = ["Position", "Velocity", "Orientation"]
    one_second = [
        geometric_percent_change(baseline_1s, phase_1s, UNSEEN, metric)
        for metric in RELATIVE_METRICS
    ]
    five_second = [
        geometric_percent_change(baseline_5s, phase_5s, UNSEEN, metric)
        for metric in RELATIVE_METRICS
    ]
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, one_second, 0.36, label="1 s")
    axes[1].bar(x + 0.18, five_second, 0.36, label="5 s")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Sequence-balanced change vs CodeNet (%)")
    axes[1].set_title("Relative RMSE on five unseen sequences (lower is better)")
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis_results")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-activation", action="store_true")
    args = parser.parse_args()

    result_paths = {
        "1s": {
            "CodeNet": Path("result/euroc_airimu_codenet_200/loss_result.json"),
            "WaveletPhase": Path("result/euroc_airimu_phase_200/loss_result.json"),
        },
        "5s": {
            "CodeNet": Path("result/euroc_airimu_codenet_1000/loss_result.json"),
            "WaveletPhase": Path("result/euroc_airimu_phase_1000/loss_result.json"),
        },
    }
    trajectory = {
        horizon: {name: load_json_rows(path) for name, path in models.items()}
        for horizon, models in result_paths.items()
    }

    all_metrics = GLOBAL_METRICS + RELATIVE_METRICS
    summary = {"trajectory": {}, "spectrum": {}}
    for horizon, models in trajectory.items():
        summary["trajectory"][horizon] = {}
        for model_name, rows in models.items():
            summary["trajectory"][horizon][model_name] = {
                "unseen_mean": mean_metrics(rows, UNSEEN, all_metrics),
                "unseen_median": median_metrics(rows, UNSEEN, all_metrics),
                "train_mean": mean_metrics(rows, TRAIN, all_metrics),
            }
        summary["trajectory"][horizon]["unseen_geometric_change_percent"] = {
            metric: geometric_percent_change(
                models["CodeNet"], models["WaveletPhase"], UNSEEN, metric
            )
            for metric in all_metrics
        }
        summary["trajectory"][horizon]["unseen_wins_out_of_5"] = {
            metric: sum(
                models["WaveletPhase"][sequence][metric]
                < models["CodeNet"][sequence][metric]
                for sequence in UNSEEN
            )
            for metric in all_metrics
        }

    for model_name, path in {
        "CodeNet": Path("experiments/EuRoC/codenet/net_output.pickle"),
        "WaveletPhase": Path(
            "experiments/EuRoC/freqnet_phase_aware/net_output.pickle"
        ),
    }.items():
        summary["spectrum"][model_name] = correction_spectrum(path)
    if not args.skip_activation:
        summary["phase_activation_diagnostic"] = phase_activation_diagnostic(
            device=args.device
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "airimu_phase_trajectory_analysis.json"
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    make_plot(trajectory, args.output_dir / "airimu_phase_trajectory_comparison.png")
    print(json_path)


if __name__ == "__main__":
    main()
