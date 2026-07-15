# Stage 2 双帧 Progress：B0-B5 实验说明

## 1. 目标与边界

Stage 1 仍然只负责根据 `first + goal + language` 生成 53 帧视觉轨迹，并把它缓存为
`trajectory_frame_latents [53,C,1,H,W]`。本次修改不改变 Stage 1、cache schema 或已有
V1-proper checkpoint。

Stage 2 的目标是：给定同一 episode 的轨迹 memory，以及在线观测，预测当前观测在
53-slot memory 中的位置和绝对进度。B1-B5 把在线输入从单帧扩展为
`(previous_frame, current_frame, observable_time_gap)`，用于识别停滞、前进和后退，缓解
单帧外观歧义造成的 slot 跳变。

推理输出仍是当前时刻的绝对 Progress：

```text
progress = sum_k p_current(k) * k / 52
```

它不是把每一步预测的 delta 累加起来，因此单次错误不会永久积累。

## 2. 共享数据与训练口径

六组实验固定使用：

- Stage 1 配置：`vgm_bridge_v1_proper_detailed_caption_v1_53f_full_jitter.yaml`
- Stage 1 checkpoint：`checkpoint_step_30000`
- 53-slot cache：`v1_proper_detailed_53f_step30000_s50_53slot_v2`
- Stage 2 backbone：A2 的 `serial_latent + split_height + 2 x 8 tokens`
- Progress bins：53
- 训练：60 epochs、相同 seed、optimizer、LR、数据划分和全 episode query 预算
- Ranking loss：全部关闭

B1-B5 的配对策略为：

```text
moving gap: Uniform{1,...,8}
forward:   70%
stay:      15%
reverse:   15%
```

采样时先随机排列 `current`，`queries_per_episode=0` 时每个 source frame 每个 epoch 恰好
出现一次，因此 B1-B5 与 B0 具有相同的 current-frame 边缘分布。随后再按上述比例为每个
current 采 previous；位于 episode 边界时，会在可行方向上重新归一化概率。这样不会因为
双帧配对而重复遗漏 current frame，也不会把训练分布偏向轨迹中段。

两帧分别通过同一个 `SharedFrameLatentEncoder` 编码，不在像素或 VAE 输入层拼接。模型只
接收可观测的绝对时间间隔 `abs(index_current-index_previous)/8`；它不会接收进度标签或带
符号的方向，因此不存在标签泄漏。

Cache loader 会强制检查 `frame_indices` 和 Progress 标签严格递增。双帧 sampler 和顺序
评估都依赖这一时序契约；乱序或重复标签会直接报错，而不会静默生成错误的 forward/reverse
监督。

## 3. 六组受控实验

| 组别 | Query 结构 | 输出 | Loss |
|---|---|---|---|
| B0 | 单帧 A2 baseline | `p(k), k=0..52` | `L_align + L_abs` |
| B1 | 双帧 token fusion | `p(k), k=0..52` | `L_align + L_abs` |
| B2 | 双帧联合定位 | `P(j,k), j,k=0..52` | `L_joint` |
| B3 | 双帧联合定位 | `P(j,k)` | `L_joint + 52 L_abs` |
| B4 | 双帧联合定位 | `P(j,k)` | `L_joint + 32 L_delta` |
| B5 | 双帧联合定位 | `P(j,k)` | `L_joint + 52 L_abs + 32 L_delta` |

这里 `j` 是 previous slot，`k` 是 current slot。联合分布在全部 `53 x 53=2809` 个位置
上做一次全局 softmax：

```text
p_previous(j) = sum_k P(j,k)
p_current(k)  = sum_j P(j,k)
delta         = sum_j sum_k P(j,k) * (k-j)/52
```

previous 和 current 先各自构造 53-bin Gaussian soft label，再通过“同分位数单调耦合”
组成联合 soft label。该耦合严格保留两个 Gaussian 边缘分布：forward 只落在 `k>=j`，
stay 只落在 `k=j`，reverse 只落在 `k<=j`。移动样本允许保留对角线质量，因为 source
episode 的一帧变化可能小于一个 53-bin slot；例如 214 帧 episode 的 `gap=1` 只相当于
`0.244` 个 slot，目标会用约 75.6% 对角线和 24.4% 前进一步表达，而不是错误地强迫至少
前进一个完整 slot。这样既消除反方向监督，也保持真实 delta 的期望。`L_abs` 监督当前绝对
进度，`L_delta` 监督 `y_current-y_previous`。B2-B5 的最终在线 Progress 都从
`p_current` 的期望直接计算，而不是使用离散 argmax，也不是累计 delta。

三个 loss 的原始数值不能直接代表其参数更新强度。使用 5 个真实 cache episode 和 5 个
独立初始化测量 B5 的全模型梯度，范数中位数（最小值到最大值）为：

```text
L_joint: 49.61  (48.76 - 59.20)
L_abs:    0.195  (0.028 - 0.366)
L_delta:  0.297  (0.174 - 0.445)
```

逐样本计算后，`L_abs/L_joint` 和 `L_delta/L_joint` 的梯度范数比中位数分别为 `0.0037`
和 `0.0060`；joint 与两项辅助梯度的余弦中位数分别为 `0.047` 和 `0.007`。因而 B3/B5
使用 `regression_weight=52`，B4/B5 使用 `delta_weight=32`，让每个辅助项在初始化时的
中位梯度约为 joint 的 19%。这是用于建立可比较实验起点的经验校准，不是假设训练全程比例恒定；
正式训练仍需记录各分项 loss 和总梯度。B0/B1 保留历史 `L_align + L_abs` 的 1:1 权重，
保证单帧 A2 基线和 fused 输入对照不变。

Trainer 的终端日志会同时打印 `loss/align/joint/abs/delta/mae/rmse/grad`。其中四个分项
是加权前的原始 loss，`loss` 才是按配置权重求和后的反向目标，`grad` 是裁剪前的总梯度
范数；比较 B2-B5 时不能只比较 total loss 的绝对数值。

## 4. 结构差异

### B1：Two-frame fused

```text
previous -> SharedFrameEncoder -> previous tokens --+
                                                    +-> pair fusion -> 6 x cross-attention -> p(k)
current  -> SharedFrameEncoder -> current tokens  --+
gap      -> gap MLP -------------------------------+
```

pair fusion 输入为 `[previous, current, current-previous, gap_embedding]`。B1 可验证“只增加
双帧输入”是否足够；由于监督仍只定位 current，它也可能学会忽略 previous，因此评估中
必须查看 previous-shuffle 敏感性。

### B2-B5：Two-frame joint

```text
previous tokens -> shared 6 x cross-attention -> previous unary logits
current tokens  -> shared 6 x cross-attention -> current unary logits

(current-previous query, memory_k-memory_j, relative slot k-j, gap)
                         -> transition logits [53,53]

previous unary + current unary + transition logits -> joint logits [53,53]
```

该结构显式回答“上一帧位于 j、当前帧位于 k”的联合定位问题，所以可以直接监督 stay、
forward 和 reverse，而不需要 ranking loss。

## 5. 评估

离线评估保留原有指标：MAE、RMSE、Pearson/VOC、Spearman、ordering accuracy、slot
tolerance、单步跳变和 backward-step rate。

双帧模型额外评估：

- 顺序播放：固定 previous gap=4，模拟在线连续查询。
- Balanced pair：对每个 episode 构造等量 forward/stay/reverse，报告 current MAE、delta
  MAE、direction accuracy 和 wrong-direction rate。
- Previous shuffle：只在相同 observable gap 的组内，将 previous frame 循环移动约半个
  episode，并排除无法置换的单样本组。报告有效样本数、Progress 变化、alignment total
  variation 和 `>0.01` 变化比例，用于诊断模型是否真正使用上一帧。

评估输出仍包含 `metrics.json`、`episode_metrics.csv`、`episode_outputs.pt` 和逐 episode
诊断图。图中的白色虚线表示 source-time label 对应的 memory slot，不代表视觉 nearest
neighbor 的真实匹配标签。

当前 stay 样本由同一帧复制构造，reverse 样本由成功轨迹中的前后帧交换构造。它们适合验证
模型结构和方向监督，但不能替代包含真实停滞、回退和失败恢复行为的数据；正式下游结论需要
额外的真实异常轨迹评估。

## 6. 配置与启动

配置文件：

```text
configs/progress_v1_proper_b0_single_frame_a2_60ep.yaml
configs/progress_v1_proper_b1_two_frame_fused_60ep.yaml
configs/progress_v1_proper_b2_joint_only_60ep.yaml
configs/progress_v1_proper_b3_joint_abs_60ep.yaml
configs/progress_v1_proper_b4_joint_delta_60ep.yaml
configs/progress_v1_proper_b5_joint_abs_delta_60ep.yaml
```

以两卡为正式口径时，1538 个训练 episode 对应每 epoch 769 steps，60 epochs 共 46140
optimizer steps。单卡会受 `max_steps=50000` 提前截断，不能完成完整 60 epochs。

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29620 \
  train/train_progress_stage2.py \
  --config configs/progress_v1_proper_b0_single_frame_a2_60ep.yaml
```

其余组只替换 `--config`。训练默认从零初始化 Stage 2；配置中的
`source.vgm_checkpoint` 是冻结的 Stage 1 来源，不是 Stage 2 resume checkpoint。

## 7. 已完成验证

- 全仓测试：51 passed。
- 真实 214 帧 episode：current query 覆盖 `214/214`，均值保持 `0.5`；单调耦合的两个
  Gaussian 边缘误差小于 `3e-7`，反方向概率质量为 0，sub-bin delta 期望保持不变。
- B0 严格加载历史 A2 60-epoch checkpoint：无 missing/unexpected keys。
- B0-B5：真实 53-slot cache 单卡 1-step forward/backward/checkpoint smoke 全部通过。
- B5：两卡 DDP 1-step smoke 通过，未出现不同 episode 长度导致的同步挂起。
- B5：完整单 episode eval 通过，四类输出文件均成功生成。

本地开发 worktree：`/Users/n/Documents/DreamZero/worktrees/Motus-stage2`

7779 验证 worktree：`/mnt/workspace1/users/niejunnan/codebase/Motus-stage2`

开发分支：`codex/progress-stage2-two-frame`，基准提交：`946e1ad8a04d1c103474f1beaa1efe7ffd4fd984`。
