# Real-World 推理说明（无机器人环境）

这个目录提供了一个最小推理示例：只用一张初始图像和一条语言指令运行 Motus，不依赖真实机器人或仿真环境。

---

## 目录

- [概览](#概览)
- [推理模式](#推理模式)
- [运行要求](#运行要求)
- [文件说明](#文件说明)
- [使用方法](#使用方法)
  - [步骤 0：图像预处理（必需）](#步骤-0图像预处理必需)
  - [方式一：使用预编码 T5 embedding（推荐）](#方式一使用预编码-t5-embedding推荐)
  - [方式二：运行时编码 T5](#方式二运行时编码-t5)
- [输出结果](#输出结果)
- [推理调用链](#推理调用链)
- [注意事项](#注意事项)

---

## 概览

Motus 推理需要三类输入：

| 输入 | 作用 | 典型维度 |
|------|------|----------|
| 初始图像 | 作为视频生成的条件帧 | `[B=1, C=3, H, W]` |
| 机器人状态 | 作为动作预测的初始状态；无环境时用零向量占位 | `[B=1, state_dim=14]` |
| 语言指令 | 同时供 Qwen3-VL 和 WAN/T5 使用 | VLM token 与 T5 embedding 两套输入 |

推理会同时预测：

| 输出 | 说明 | 典型维度 |
|------|------|----------|
| 未来视频帧 | 条件帧之后的预测图像 | `[B, C=3, T_pred, H, W]` |
| 动作序列 | 未来 action chunk | `[B, action_chunk_size, action_dim=14]` |

---

## 推理模式

| 模式 | 显存 | 说明 |
|------|------|------|
| 预编码 T5 embedding | 大于 24 GB | 推荐部署使用。先离线编码指令，再推理。 |
| 运行时编码 T5 | 约 41 GB | 推理时加载 T5 encoder 并编码指令，需要更大显存。 |

---

## 运行要求

| 项目 | 说明 |
|------|------|
| Motus checkpoint | 包含 `mp_rank_00_model_states.pt` 的 checkpoint 目录 |
| WAN 路径 | 包含 Wan2.2 模型、VAE、T5 encoder/tokenizer 的目录 |
| 输入图像 | 三视角拼接后的单张图像 |
| 语言指令 | 描述当前任务的文本指令 |

注意：输入图像必须是三视角拼接图，通常包含 head camera、left wrist camera、right wrist camera，并按 T-shape layout 拼成一张图。若你手里是三张独立相机图，请先使用拼接脚本。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `inference_example.py` | 主推理脚本，支持命令行和 Python API 示例 |
| `encode_t5_instruction.py` | 将文本指令预编码为 T5 embedding 的工具 |
| `utils/ac_one.yaml` | AC-One checkpoint 对应的模型配置 |
| `utils/aloha_agilex_2.yaml` | Aloha-Agilex-2 checkpoint 对应的模型配置 |

相关工具：

- [`data/utils/multi_camera_concat.py`](../../../data/utils/multi_camera_concat.py)：把多相机图像拼成 Motus 需要的单张输入图。

---

## 使用方法

### 步骤 0：图像预处理（必需）

如果你有三张独立相机图，需要先拼成一张 T-shape 图像：

```bash
python data/utils/multi_camera_concat.py \
  --head_image path/to/head_camera.jpg \
  --left_image path/to/left_wrist_camera.jpg \
  --right_image path/to/right_wrist_camera.jpg \
  --output examples/first_frame.png
```

布局约定：

- 上方：head camera，保留原始尺寸
- 左下：left wrist camera，缩放到一半
- 右下：right wrist camera，缩放到一半

### 方式一：使用预编码 T5 embedding（推荐）

这一路径显存占用较低，适合部署和反复测试同一条指令。

步骤 1：先编码语言指令。

```bash
python inference/real_world/Motus/encode_t5_instruction.py \
  --instruction "Pour water from kettle to flowers" \
  --output t5_embed.pt \
  --wan_path pretrained_models
```

步骤 2：运行 Motus 推理。

```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus \
  --wan_path pretrained_models \
  --image examples/first_frame.png \
  --instruction "Pour water from kettle to flowers" \
  --t5_embeds t5_embed.pt \
  --output examples/output_ac_one.png
```

### 方式二：运行时编码 T5

这一路径会在推理时加载 T5 encoder，显存需求更高。

```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/ac_one.yaml \
  --ckpt_dir pretrained_models/Motus \
  --wan_path pretrained_models \
  --image examples/first_frame.png \
  --instruction "Pour water from kettle to flowers" \
  --use_t5 \
  --output examples/output_ac_one.png
```

---

## 输出结果

| 输出 | 说明 |
|------|------|
| `examples/output_ac_one.png` | 条件帧和预测未来帧拼成的可视化图 |
| Console | 打印预测动作维度和前几步动作值 |

示例：

```text
Predicted actions shape: (1, 48, 14)
First 3 actions:
[[ 0.012  0.003 -0.001  0.002  0.001  0.000  0.045 ...]
 [ 0.015  0.004 -0.002  0.003  0.001  0.001  0.048 ...]
 [ 0.018  0.005 -0.003  0.004  0.002  0.001  0.051 ...]]
```

---

## 推理调用链

如果你的目标是理解模型架构，建议按下面顺序看代码。

1. 入口脚本  
   `inference/real_world/Motus/inference_example.py`

   关键位置：

   - `load_image_as_tensor(...)`：输入图像变成 `first_frame [B=1, C=3, H, W]`
   - `build_vlm_inputs(...)`：构造 Qwen3-VL 图文输入，用于 understanding token
   - `language_embeddings`：T5 embedding，进入 WAN cross-attention
   - `model.inference_step(...)`：进入 Motus 模型推理主体

2. Motus 推理主体  
   `models/motus.py::Motus.inference_step`

   主要数据流：

   ```text
   first_frame [B, 3, H, W]
     -> VAE encode
   condition_frame_latent [B, 48, 1, H_lat, W_lat]

   video_latent [B, 48, T_total_lat, H_lat, W_lat]
   action_latent [B, action_chunk_size, 14]

   每个 denoising step:
     video_latent -> video_tokens [B, L_v, 3072]
     action_latent + state + registers -> action_tokens [B, L_a, 1024]
     Qwen3-VL hidden -> und_tokens [B, L_u, 512]
     三路 token 进入 MoT joint attention
     head 输出 video_velocity/action_velocity
     Euler 更新 video_latent/action_latent
   ```

3. MoT 投影入口  
   `models/motus.py::VideoModule.process_joint_attention`

   这里完成三路 token 的对齐：

   ```text
   video_tokens  [B, L_v, 3072]
   action_tokens [B, L_a, 1024] -> [B, L_a, 24, 128]
   und_tokens    [B, L_u, 512]  -> [B, L_u, 24, 128]
   ```

4. 真正拼接 Q/K/V 的地方  
   `bak/wan/modules/model.py::WanSelfAttention.forward`

   这里把三路 token 拼在同一个 attention 序列里：

   ```text
   q_cat/k_cat/v_cat: [B, L_v + L_a + L_u, 24, 128]
   ```

   attention 结束后再拆回：

   ```text
   video_out  [B, L_v, 24, 128] -> [B, L_v, 3072]
   action_out [B, L_a, 24, 128] -> [B, L_a, 1024]
   und_out    [B, L_u, 24, 128] -> [B, L_u, 512]
   ```

---

## 注意事项

- `--ckpt_dir` 应指向 Motus checkpoint 目录，通常包含 `mp_rank_00_model_states.pt`。
- `--wan_path` 是 WAN 模型根目录，用于查找 T5 权重、T5 tokenizer 和 VAE。
- 即使使用 `--t5_embeds`，`--instruction` 仍然必需，因为 Qwen3-VL 也需要文本指令来产生 understanding token。
- 如果要批量跑多条指令，建议先把 T5 embedding 离线编码好，避免反复加载 T5 encoder。
- 阅读模型代码时，以仓库根目录下的 `models/` 为准。`inference/real_world/Motus/models/` 是推理包里的拷贝，当前 `inference_example.py` 会把仓库根目录加入 `sys.path` 并导入根目录模型。
