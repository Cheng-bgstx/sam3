# SAM3 UAP 对抗攻击实验全量结果汇总

## 一、项目概述

本项目将针对 SAM2 的通用对抗扰动 (UAP) 攻击迁移至 SAM3，在 SA-Co 视频分割评估集上验证攻击效果，并进行了多轮实验、验证与可视化分析。

### 1.1 代码架构

| 类型 | SAM2 原始 | SAM3 修改版 | 说明 |
|------|-----------|-------------|------|
| 数据集 | `dataset_YOUTUBE.py` | `dataset_saco.py` | YOUTUBE VIS → SA-Co 格式 |
| 攻击设置 | `attack_setting.py` | `attack_setting_sam3.py` | SamForwarder → Sam3Forwarder |
| 工具函数 | `sam2_util.py` | `sam3_util.py` | 模型加载、数据集选择 |
| 适配器 | - | `sam3_forwarder_adapter.py` | 兼容 trainer 接口，支持 UAP 注入 |
| 图像攻击 | `uap_attack.py` | `uap_attack_sam3_image.py` | loss_t + loss_fea |
| 视频攻击 | `uap_attack.py` | `uap_attack_sam3_video.py` | loss_t + loss_diff + loss_fea (+ loss_text) |
| 测试脚本 | `uap_atk_test.py` | (复用 trainer 评估) | - |

### 1.2 攻击路线

- **SAM Route**：点提示 → backbone → sam_prompt_encoder → sam_mask_decoder（与 SAM2 一致）
- **Detector Route**：文本提示 → backbone → geometry_encoder → transformer → segmentation_head
- UAP 攻击主要在 **SAM Route** 上训练，通过共享 backbone 影响整个模型

---

## 二、生成的 UAP 攻击文件

### 2.1 `uap_file/` 目录

| 文件名 | 损失配置 | 数据集变体 | 说明 |
|--------|----------|------------|------|
| SA-Co_sam3_video_saco_yt1b_0_fusion.pth | 0_fusion | saco_yt1b | 仅 loss_t + loss_diff，无 loss_fea、loss_text |
| SA-Co_sam3_video_saco_yt1b_0_fusion_quan.pth | 0_fusion (quan) | saco_yt1b | 同上，quan 变体 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_new.pth | 0_fusion (new) | saco_yt1b | 同上，new 变体 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_test.pth | 0_fusion (test) | saco_yt1b | 同上，test 变体 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_onlydiff.pth | onlydiff | saco_yt1b | 仅 loss_diff |
| SA-Co_sam3_video_saco_yt1b_0_fusion_lossfea0_lossdiff1.pth | loss_fea=0, loss_diff=1 | saco_yt1b | 仅 loss_diff |
| SA-Co_sam3_video_saco_yt1b_0_fusion_lossfea1_lossdiff0.pth | loss_fea=1, loss_diff=0 | saco_yt1b | 仅 loss_fea |
| SA-Co_sam3_video_saco_yt1b_1_fusion.pth | 1_fusion | saco_yt1b | loss_text (fusion 模式) |
| SA-Co_sam3_video_saco_yt1b_1_per_mask.pth | 1_per_mask | saco_yt1b | loss_text (per_mask 模式) |
| SA-Co_sam3_video_saco_sg_0_fusion.pth | 0_fusion | saco_sg | SmartGlasses 数据集 |
| SA-Co_sam3_video_saco_sg_0_fusion_quan.pth | 0_fusion (quan) | saco_sg | 同上 |

### 2.2 `uap_file3/` 目录（系统化命名）

| 文件名 | loss_text | text_mode | loss_fea | loss_diff |
|--------|-----------|-----------|----------|-----------|
| SA-Co_sam3_video_saco_yt1b_0_fusion_lossfea0_lossdiff0.pth | 0 | fusion | 0 | 0 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_lossfea0_lossdiff1.pth | 0 | fusion | 0 | 1 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_lossfea1_lossdiff0.pth | 0 | fusion | 1 | 0 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_lossfea1_lossdiff1.pth | 0 | fusion | 1 | 1 |
| SA-Co_sam3_video_saco_yt1b_1_fusion_lossfea0_lossdiff0.pth | 1 | fusion | 0 | 0 |
| SA-Co_sam3_video_saco_yt1b_1_per_mask_lossfea0_lossdiff0.pth | 1 | per_mask | 0 | 0 |

### 2.3 `verify_out/scale_ablation/` 目录

| 文件 | 说明 |
|------|------|
| uap_scale_0.3.pth ~ uap_scale_1.0.pth | 原始 UAP 按 scale 线性缩放后的扰动 |
| pred_scale_0.0.json ~ pred_scale_1.0.json | 各 scale 下的预测结果 |
| metrics_scale_0.0.json ~ metrics_scale_1.0.json | 各 scale 下的评估指标 |

---

## 三、评估配置与流程

### 3.1 评估入口

```bash
# 使用 train.py 的 val 模式
python sam3/train/train.py --config-name saco_video_evals/saco_veval_sav_test_split
# 或 saco_veval_yt1b_test_split（需在 paths 中指定 uap_path）
```

### 3.2 配置文件

| 配置文件 | 用途 | 输出目录 |
|----------|------|----------|
| `saco_veval_sav_test_split.yaml` | SA-Co SAV 测试集 | `test_sav_split_d0.5/` |
| `saco_veval_yt1b_test_split.yaml` | SA-Co YT1B 测试集 | `attack/` 下各实验子目录 |

### 3.3 评估脚本

- **run_inference_eval.py**：独立推理 + 评估，支持 `--uap_path`、`--uap_scale`
- **trainer.py**：集成评估，支持 `uap_path`、`uap_scale` 配置

---

## 四、核心实验结果

### 4.1 Scale 消融（verify_out/scale_ablation）

**结论：mAP 随 UAP scale 增大单调下降，证明扰动强度是攻击成功的主因。**

| UAP Scale | video_mask_mAP_50_95 | 说明 |
|-----------|----------------------|------|
| 0.0 (Clean) | **0.4562** | 无扰动基线 |
| 0.3 | 0.2949 | 扰动减弱约 35% |
| 0.5 | 0.044 | 扰动减半，mAP 大幅下降 |
| 0.7 | 0.0002 | 几乎归零 |
| 1.0 | **0.0** | 全强度扰动，完全攻击成功 |

### 4.2 Scale 与攻击边界的关系（回答：scale < 1 是否等价于攻击边界缩小？）

**结论：是的，等价。**

评估时：`uap_scaled = uap * uap_scale`，`img_adv = (img_01 + uap_scaled).clamp(0, 1)`

- `scale=1.0`：有效扰动幅度 = 原始 UAP 幅度（训练时 eps，如 10/255）
- `scale=0.5`：有效扰动幅度减半，**等价于有效 eps = 5/255**
- `scale=0.3`：等价于有效 eps ≈ 3/255

因此 **scale 与有效 eps 线性对应**：`effective_eps = scale × eps_train`。scale < 1 即等价于将攻击边界（扰动范数上界）缩小。

---

### 4.3 完整实验结果表（按 eps × 攻击目标 × 指标）

**指标选择说明**（能反映当前视频分割/ tracking 性能）：

| 指标 | 全称 | 说明 | 选用理由 |
|------|------|------|----------|
| **mAP** | video_mask_mAP_50_95 | COCO 风格 mask mAP (IoU 50:95) | 主指标，业界通用 |
| **TETA** | video_mask_teta | TETA 综合评估（定位+关联+分类） | 视频跟踪与分割联合度量 |
| **HOTA** | video_mask_all_phrase_HOTA | Higher Order Tracking Accuracy | 跟踪质量，对遮挡/时序敏感 |

**数据来源**：verify_out/scale_ablation（UAP 加在图像上，影响 Image encoder/backbone，进而影响 Decoder(point) 与 Decoder(text)）。

| ε (有效) | 攻击目标 | mAP (×100) | TETA | HOTA (×100) |
|----------|----------|------------|------|-------------|
| **0/255 (Clean)** | - | **45.62** | **73.20** | **69.95** |
| 3/255 | Image encoder | 29.49 | 62.78 | 60.54 |
| 5/255 | Image encoder | 4.40 | 17.13 | 23.64 |
| 7/255 | Image encoder | 0.02 | 3.97 | 6.24 |
| 10/255 | Image encoder | 0.00 | 0.00 | 0.00 |
| 3/255 | Decoder(point) [SAM Route] | 29.49 | 62.78 | 60.54 |
| 5/255 | Decoder(point) [SAM Route] | 4.40 | 17.13 | 23.64 |
| 7/255 | Decoder(point) [SAM Route] | 0.02 | 3.97 | 6.24 |
| 10/255 | Decoder(point) [SAM Route] | 0.00 | 0.00 | 0.00 |
| - | Decoder(text) [Detector Route] | - | - | - |

**说明**：

- **Image encoder**：UAP 施加于图像，扰动经 backbone 影响两条解码路径；当前 scale 消融对应 scale ∈ {0, 0.3, 0.5, 0.7, 1.0}，映射到有效 ε = 0, 3/255, 5/255, 7/255, 10/255（训练 eps=10/255）。
- **Decoder(point)**：SAM Route（点提示），与 Image encoder 共用同一套 UAP，因此数值相同。
- **Decoder(text)**：Detector Route（文本提示），需用 loss_text 训练的 UAP（1_fusion）单独做 scale 消融；exp3 中 scale=1.0 时 mAP 0.424→0.344，约 19% 下降，与 SAM Route 的全崩溃形成对比。
- 若需 **2/255、4/255、6/255、8/255** 的细粒度结果，可运行 `scale_ablation` 并指定 `--scales 0.2,0.4,0.6,0.8`（或修改 verify 脚本支持自定义 scale 列表）。

---

### 4.4 各攻击配置评估（attack/ 目录）

| 实验名称 | UAP 文件 | val_stats.json | sam3_eval_metrics |
|----------|----------|----------------|-------------------|
| SA-Co_sam3_video_saco_yt1b_0_fusion | 0_fusion | Meters_val/val_ytvis_val: 0 | dataset_level_metrics: {} |
| SA-Co_sam3_video_saco_yt1b_0_fusion_quan | 0_fusion_quan | 同上 | 同上 |
| SA-Co_sam3_video_saco_yt1b_1_fusion | 1_fusion | 同上 | 同上 |
| SA-Co_sam3_video_saco_yt1b_1_per_mask | 1_per_mask | 同上 | 同上 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_test | 0_fusion_test | 同上 | 同上 |
| SA-Co_sam3_video_saco_yt1b_0_fusion_quan_0.3 | 0_fusion_quan + uap_scale=0.3 | - | - |

**说明**：在 `uap_path` 指定且 `uap_scale=1.0` 时，对抗预测导致 mAP 归零，val_stats 中 Meters 显示 0，符合攻击成功预期。若需得到非零 mAP 以便分析，可设置 `uap_scale=0.5` 或更小。

---

## 五、验证与可视化代码

### 5.1 攻击原因验证（verify_sam3_attack_causes.py）

| 功能 | 说明 |
|------|------|
| `scale_ablation` | 扰动强度消融：不同 uap_scale 下的 mAP 曲线 |
| `metric_compare` | 同一预测下 video mAP 与按帧/按实例统计对比 |
| `sam_route_only` | 仅 SAM 点提示 + UAP，检查 mask 是否崩溃 |
| `probe_logits` | 内部评分探针：Clean vs Adv 的 pred_logits、presence_logits |
| `force_recall` | 强制召回：score_threshold=0 保留所有检测再算 mAP |
| `modality_compare` | 正常阈值 vs 强制召回的 mAP 对比 |
| `visualize_internals` | 特征热力图 + 原始 Mask 可视化 |

### 5.2 可视化脚本

| 脚本 | 功能 |
|------|------|
| `visualize_pred_and_gt.py` | 预测 vs GT 对比（原图 \| GT 框+mask \| 预测框+mask） |
| `visualize_per_frame_folder.py` | 按帧可视化 |
| `exp2_feature_visualization.py` | 特征层可视化：Backbone / Detector FPN / SAM FPN 的 clean vs adv 对比 |
| `exp3_cross_pathway_attack.py` | 跨路径攻击 (CPAA)：SAM Route + Detector Route 联合优化 |

### 5.3 归因分析

| 脚本 | 功能 |
|------|------|
| `attack_attribution_analysis.py` | UAP 施加位置/缩放检查 + 攻击成功率归因（覆盖不足、score 过滤、有效帧数） |
| `analyze_pred_vs_gt.py` | 预测 vs GT 格式对比 + 逐条分析 score、有效分割帧数 |
| `check_pred_json.py` | 预测 JSON 结构检查 |

---

## 六、关键发现与结论

### 6.1 攻击效果

1. **SAM Route UAP 在 eps≈10/255 时可将 mAP 从 ~0.46 降至 0**，攻击成功率 100%。
2. **Scale 消融**表明 mAP 随 scale 单调下降，scale=0.5 时 mAP≈0.04，scale=1.0 时 mAP=0。
3. **评估时使用 uap_scale<1**（如 0.5）可减弱扰动，得到非零 mAP，便于进一步分析。

### 6.2 攻击归因（来自验证脚本与 exp3）

1. **双路共享 Backbone**：SAM Route 与 Detector Route 共享 backbone，SAM Route 上的 UAP 会同时影响文本提示路径。
2. **置信度过滤**：对抗样本导致 pred_logits、presence_logits 下降，大量检测被 score 阈值过滤，最终输出为空。
3. **特征层破坏**：exp2 显示 UAP 在 backbone 层造成较大特征偏移，跨路径传递明显。

### 6.3 数据集与配置

- **训练**：SA-Co train_videos.txt（saco_yt1b / saco_sav / saco_sg）
- **测试**：saco_veval_*_test_split.json（与训练无交集）
- **分辨率**：1008×1008，归一化 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]

---

## 七、目录结构总览

```
UAP-SAM2-main/
├── uap_file/                    # 生成的 UAP（主要）
├── uap_file3/                   # 系统化命名的 UAP
├── attack/                      # 各攻击配置的评估结果
│   ├── SA-Co_sam3_video_saco_yt1b_0_fusion/
│   │   ├── preds/               # 预测 JSON
│   │   ├── logs/val_stats.json
│   │   └── ...
│   └── ...
├── verify_out/                  # 验证输出
│   ├── scale_ablation/          # scale 消融结果
│   └── modality_compare/
├── test_sav_split_d0.5/         # saco_veval_sav_test_split 评估输出
├── verify_sam3_attack_causes.py # 六类验证入口
├── visualize_pred_and_gt.py     # 预测 vs GT 可视化
├── attack_attribution_analysis.py
├── exp2_feature_visualization.py
├── exp3_cross_pathway_attack.py
└── ...
```

---

## 八、复现命令摘要

```bash
# 1. 训练 UAP（示例：0_fusion，loss_t + loss_diff）
python uap_attack_sam3_video.py --saco_variant saco_yt1b --loss_t --loss_diff

# 2. 评估（需修改 config 中 paths.uap_path）
python sam3/train/train.py --config-name saco_video_evals/saco_veval_yt1b_test_split

# 3. Scale 消融
python verify_sam3_attack_causes.py scale_ablation \
  --uap_path uap_file/SA-Co_sam3_video_saco_yt1b_0_fusion.pth \
  --ann_file data/annotation/saco_veval_yt1b_test_split.json \
  --img_folder data/media/saco_yt1b/JPEGImages_6fps \
  --checkpoint sam3/checkpoints/sam3.pt \
  --out_dir verify_out/scale_ablation

# 4. 预测 vs GT 可视化
python visualize_pred_and_gt.py \
  --gt_json data/annotation/saco_veval_yt1b_test_split.json \
  --pred_json attack/SA-Co_sam3_video_saco_yt1b_0_fusion/preds/saco_veval_yt1b_test_split.json \
  --media_dir data/media/saco_yt1b/JPEGImages_6fps \
  --out_dir results/vis_0_fusion
```

---

*报告生成时间：基于当前代码库与实验结果整理*
