# 上游来源与许可

两个本地输入快照均没有 `.git`，因此**原始快照对应的精确上游 commit 未知**。下列 SHA 是已核实的上游参考版本，用于来源核对和恢复完整 LICENSE；它们不是本地快照的原始版本证明。

| 项目 | 上游仓库 | 已核实参考 commit | 许可文件 |
| --- | --- | --- | --- |
| AirIMU | [cannnnxu/AirIMU](https://github.com/cannnnxu/AirIMU)（历史链接 sleepycan/AirIMU） | `b3399cda746f69798159472fa538d37c600a9f67` | [`airimu/LICENSE`](../airimu/LICENSE) |
| AirIO | [Air-IO/Air-IO](https://github.com/Air-IO/Air-IO) | `7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73` | [`airio/LICENSE`](../airio/LICENSE) |

两份完整许可均为 BSD-3-Clause，AirIMU 版权声明为 Carnegie Mellon University 2024，AirIO 为 Carnegie Mellon University 2025。保留原许可及版权声明；数据集与第三方依赖按各自许可使用。

本地快照已包含既有研究改动，包括频率/小波/相位模型、精度和时间戳处理、评估及滤波相关工作。不能将交付源码、权重或历史结果描述为精确官方基线，也不能用参考 SHA 暗示全部文件与该版本相同。基线名称表示方法家族；严格版本归属仍需与上游逐文件比较。

## 论文引用

以下引用信息沿用项目 README 的 arXiv 条目：

- Yuheng Qiu, Chen Wang, Can Xu, Yutian Chen, Xunfei Zhou, Youjie Xia, Sebastian Scherer. **AirIMU: Learning Uncertainty Propagation for Inertial Odometry**. 2023. [arXiv:2310.04874](https://arxiv.org/abs/2310.04874).
- Yuheng Qiu, Can Xu, Yutian Chen, Shibo Zhao, Junyi Geng, Sebastian Scherer. **AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability**. 2025. [arXiv:2501.15659](https://arxiv.org/abs/2501.15659).

引用论文不表示论文作者认可本地研究改动或本交付中的实验结论。结果的验证范围与限制见 [`reports/RESULTS_AND_LIMITATIONS.md`](../reports/RESULTS_AND_LIMITATIONS.md)。
