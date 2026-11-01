# AirIMU / AirIO 学习型 IMU 复现交付

这里整理既有 EuRoC 复现代码、研究模型、5 个代表 CKPT 和少量推理输出，供后续校验。原始本地快照没有 Git 历史，精确上游基线 commit **未知**；可核实的上游参考版本与许可见 [UPSTREAM](provenance/UPSTREAM.md)。

## 从哪里开始

- `airimu/`：IMU 修正及不确定性、预积分；`codenet` 为基线方法，`freqnet_phase_aware` 与 `freqnet_no_phase` 为既有研究扩展。
- `airio/`：机体系速度及不确定性；`codewithrot` 为基线方法，`freqmediumrot` 为既有研究扩展。
- `artifacts/predictions/`：两次 AirIO 历史运行的原始 pickle（约 4.4 MB），以及等值、无需 CUDA 的 NPZ；仅含预测速度、协方差和时间戳。
- `reports/`：本次 CPU 验证记录、历史指标与[适用限制](reports/RESULTS_AND_LIMITATIONS.md)。历史训练/轨迹结果不是本次重跑结果。
- `provenance/source_manifest.json`：整理前源文件的相对路径与 SHA256。`SHA256SUMS` 记录最终交付文件哈希。

## 环境与轻量校验

本次使用 Python 3.11、PyTorch 2.7.0+cu128，强制 CPU 运行。依赖版本在 `requirements-repro.txt`，未在全新环境重新安装验证。建议使用独立环境，先按平台安装 PyTorch 2.7.0，再安装此文件。`airio/requirements.upstream.txt` 仅为原快照依赖记录。

在仓库根目录执行：

```bash
python scripts/check_hashes.py
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/verify_all.py
```

默认校验会运行现有测试、严格加载 5 个 CKPT、执行合成输入的 CPU 推理，并检查历史 NPZ。它不会下载数据或开始训练。结果写到 `reports/{unit_tests,checkpoint_validation,prediction_validation}.json`。

历史基线输出在 MH_02、V2_02、V1_01 各有一个末尾重复时间戳；原始输出完整保留。校验会报告警告，做轨迹插值前必须显式去重，不能把这批保存输出视为无瑕疵的严格时间序列。NPZ 用 `numpy.load(path, allow_pickle=False)` 读取；原 pickle 含 CUDA 张量，仅对可信文件使用 `scripts/export_outputs.py` 中的 CPU 加载方式。

## 权重与对应配置

| 项目 / 方法 | 配置 | 历史选择 epoch | CKPT 大小 |
| --- | --- | ---: | ---: |
| AirIMU CodeNet | `configs/exp/EuRoC/codenet.conf` | 99 | 6.28 MB |
| AirIMU phase | `configs/exp/EuRoC/freqnet_phase_aware.conf` | 94 | 6.95 MB |
| AirIMU no-phase | `configs/exp/EuRoC/freqnet_no_phase.conf` | 92 | 6.07 MB |
| AirIO CodeNet+rot | `configs/EuRoC/motion_body_rot.conf` | 81 | 4.45 MB |
| AirIO wavelet+phase | `configs/EuRoC/motion_body_freq_medium_rot.conf` | 58 | 2.97 MB |

权重保持在各项目 `experiments/.../ckpt/best_model.ckpt` 下，配置 basename 与实验目录一致；没有用五条未见序列选出的 `best_eval_model.ckpt` 替代。完整 SHA256、配置关联与加载结果见 [checkpoint_validation.json](reports/checkpoint_validation.json)。参数 YAML 为历史记录，执行入口使用 `configs/`。

## 数据、推理与评估

自行从 [EuRoC 官方网站](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) 或[上游数据发布页](https://github.com/Air-IO/Air-IO/releases/tag/datasets)获取数据并遵守其使用条款。原始数据没有打包。每条序列应包含 `mav0/imu0/data.csv`、`mav0/state_groundtruth_estimate0/data.csv`。AirIO 上游说明 V1_01 有重算 GT，本交付没有重新确认历史本机数据的具体 GT 版本，跨版本比较前需核对。

`EUROC_ROOT` 设为解压后含 `MH_02_easy/` 等序列目录的绝对路径；这两个 EuRoC 数据配置支持环境变量覆盖。

```bash
export EUROC_ROOT=/absolute/path/to/EuRoC
cd airimu
python inference.py --config configs/exp/EuRoC/codenet.conf \
  --device cpu --precision float32 --cpu-threads 2 --seqlen 1000
python evaluation/evaluate_state.py \
  --dataconf configs/datasets/BaselineEuroc/Euroc_1000.conf \
  --exp experiments/EuRoC/codenet --seqlen 200 \
  --savedir result/new_codenet_200 --device cpu
cd ../airio
python inference_motion.py --config configs/EuRoC/motion_body_rot.conf \
  --device cpu --precision float32 --cpu-threads 2 --seqlen 1000 \
  --output result/new_airio/net_output.pickle
```

AirIO 上述推理默认使用 **GT 姿态**。完整轨迹评估入口是 `evaluation/evaluate_motion.py`，其 `--exp` 目录需包含名为 `net_output.pickle` 的输出。可执行 `python evaluation/evaluate_motion.py --dataconf configs/datasets/EuRoC/Euroc_body.conf --exp result/new_airio --seqlen 1000 --savedir result/new_airio_metrics --device cpu --usegtrot`。历史指标见 `result/*_valid/result.json`。AirIMU 的 `--gtrot` / `--gtinit` 为 `store_false` 开关，传入即关闭，复验时应记录完整命令。本次没有重跑这些全数据命令。

如需重新训练，在各项目目录中显式运行（耗时任务，不属于默认校验）：

```bash
# airimu/
python train.py --config configs/exp/EuRoC/codenet.conf \
  --device cpu --precision float32 --seed 42 --num-workers 0
# airio/
python train_motion.py --config configs/EuRoC/motion_body_rot.conf \
  --device cpu --precision float32 --seed 42 --num-workers 0 --log
```

AirIO 原入口的 `--log` 禁用 wandb。训练会写对应实验目录，应先另存已有 CKPT。研究模型使用表中对应配置；其他数据集配置保留作参考，未在本次验证。

## 本次整理改动

保留已有模型/训练/推理逻辑及研究扩展；从两个快照中筛出核心代码、配置、测试与指标，排除原始数据、虚拟环境、私人文件和无关调试产物。恢复完整上游 BSD 许可证，将本机路径改为相对路径，EuRoC 增加 `EUROC_ROOT` 覆盖，新增 CPU 校验及输出转换脚本。历史报告有路径脱敏，历史结果未重新计算。

[AirIMU 许可](airimu/LICENSE)、[AirIO 许可](airio/LICENSE)及原 README 中论文引用均保留。第三方依赖和数据继续适用各自条款；不因本次整理变更原有许可。
