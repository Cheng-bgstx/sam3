# 当前实验结果与进度整理

## 1. 当前整体结论

目前仓库里已经形成一条比较完整的 SAM3 UAP 实验链路：`UAP 训练 -> 官方评估 -> scale 消融 -> 阈值/归因分析 -> 特征梯度分析 -> 机理文档整理`。  
从现有结果看，**主结论已经比较明确**：

- 针对 SAM Route / 图像编码路径训练的 UAP，对 SAM3 视频分割的破坏非常强。
- 扰动强度增大时，指标呈明显单调下降。
- 仅 text 路径的攻击效果明显弱于 mask / 图像主路径。
- `diff + fea` 会进一步放大崩溃，说明时序一致性项和特征项是有效增强项。
- 当前“为什么会崩”的解释基本闭环：**共享 backbone + detector score/prob 门控 + mask/feature 质量下降** 是主要原因。

## 2. 当前最可信的核心实验结果

### 2.1 Clean 基线

来自 `verify_out/scale_ablation/metrics_scale_0.0.json`：

| 条件 | mask mAP | bbox mAP | TETA | HOTA(mask) |
|---|---:|---:|---:|---:|
| Clean | 0.4562 | 0.4283 | 73.196 | 0.6995 |

这组结果可以视为目前仓库内最稳定的 SAM3 clean 基线。

### 2.2 scale 消融结果

来自 `verify_out/scale_ablation/`：

| UAP scale | 有效扰动理解 | mask mAP | TETA | HOTA(mask) |
|---|---|---:|---:|---:|
| 0.0 | clean | 0.4562 | 73.1960 | 0.6995 |
| 0.3 | 约等于 3/255 | 0.2949 | 62.7830 | 0.6054 |
| 0.5 | 约等于 5/255 | 0.0440 | 17.1330 | 0.2364 |
| 0.7 | 约等于 7/255 | 0.00024 | 3.9732 | 0.0624 |
| 1.0 | 约等于 10/255 | 0.0000 | 0.0000 | 0.0000 |

结论：

- **攻击强度与性能下降基本单调对应。**
- `scale=0.5` 时已经接近崩溃。
- `scale=1.0` 时官方指标几乎完全归零。
- 因此目前可以认为：**SAM3 在该设置下对 UAP 非常脆弱，且不是偶然失效，而是随扰动强度稳定恶化。**

## 3. 已完成的攻击配置结果

来自 `analysis/sam3_metrics_table.md` 与 `analysis/sam3_metrics_table.csv`：

| 实验 | mask mAP | bbox mAP | TETA | HOTA(mask) | 结论 |
|---|---:|---:|---:|---:|---|
| clean_test_yt1b | 0.4507 | 0.4315 | 73.031 | 0.6981 | clean 基线正常 |
| eps3, `t=1,diff=0,fea=0,text=0` | 0.1845 | 0.1901 | 51.256 | 0.4631 | 仅 mask 路径攻击已明显有效 |
| eps4, `t=1,diff=0,fea=0,text=0` | 0.0492 | 0.0569 | 25.590 | 0.2178 | 扰动增强后继续恶化 |
| eps4, `t=1,diff=1,fea=1,text=0` | 0.0000 | 0.0000 | 1.4007 | 0.0254 | 几乎完全崩溃 |
| eps4, `text=1, per_slot_losst` | 0.3956 | 0.3879 | 71.875 | 0.6758 | 仅文本攻击影响很小 |
| eps4, `darksam=1` | 0.0376 | 0.0339 | 15.592 | 0.1838 | 比普通 eps4 更差 |

阶段性判断：

- **主攻击贡献来自 mask / 图像主路径，不是 text 路径。**
- `diff` 和 `fea` 的加入会把原本已经有效的攻击进一步推向“全崩”。
- `DarkSAM` 相关配置会进一步放大脆弱性。

## 4. 机理分析当前结论

现有分析文档主要包括：

- `analysis/SAM3_UAP_mechanism_final_report.md`
- `analysis/SAM3_UAP_mechanism_resolved.md`
- `analysis/SAM3_architecture_and_scores_deep_dive.md`
- `analysis/SAM3_sensitivity_report.md`

当前比较稳的解释如下：

1. UAP 先作用在图像输入上，经共享 backbone 传到两条路径。
2. detector / grounding head 的 `pred_probs`、mask 质量和 score 受到明显破坏。
3. 大量目标在 score threshold 处被过滤，导致后续 tracker 与评估阶段拿不到有效目标。
4. 所以最终表现不是“框还在、mask 变差一点”，而是**整条输出链路接近被掐断**。

## 5. 阈值与敏感性实验结果

来自 `analysis/sensitivity_out/threshold_sweep_results.json`：

| 阈值 | clean mAP | adv mAP |
|---|---:|---:|
| 0.1 | 0.01569 | 0.00000 |
| 0.2 | 0.01101 | 0.00000 |
| 0.3 | 0.01434 | 0.00000 |
| 0.5 | 0.01101 | 0.00000 |

这部分结果要谨慎解读：

- 这是 `limit_ids` 条件下的小规模敏感性验证，不是完整官方主表。
- 但它仍说明一个重要事实：**即便显著放宽阈值，adv 结果仍然是 0，说明问题不只是阈值太高，而是 detector/mask 本身已经被打坏。**

## 6. 特征与梯度分析结果

来自 `analysis/sam3_feature_grad_stats.json`，共 4 个样本：

- clean 特征 L2 约在 `2810 ~ 3021`
- adv 特征 L2 约在 `2881 ~ 3491`
- clean/adv 特征差的 L2 约在 `1998 ~ 3696`
- 输入梯度 `grad_x` 的 L2 约在 `109 ~ 280`

结论：

- UAP 在 backbone 特征空间中造成了显著偏移，不是只在输出层做微弱扰动。
- 输入到输出的梯度响应也不小，支持“模型存在统一脆弱方向”的判断。

## 7. 当前进度状态

### 7.1 已完成

- SAM2 攻击链路向 SAM3 的基本迁移。
- SA-Co / YT1B 评估链路接通。
- 多组 UAP 文件已经生成，保存在 `uap_file/`、`uap_file3/`、`uap_file_new/`。
- 主实验表已经整理完成。
- scale ablation 已完成，并给出最关键的强度曲线。
- threshold sweep、feature/grad、score/mask 诊断已经完成一轮。
- 机理分析文档已经比较完整。

### 7.2 进行中 / 已写脚本但未形成最终结果

从 2026-04-24 的新增文件看，近期主要在补“更细粒度分析脚本”，包括：

- `analysis/run_sam3_defense_priority_ablation.py`
- `analysis/run_sam3_stage_swap.py`
- `analysis/run_sam3_component_localizer.py`
- `analysis/run_sam3_init_follow_counterfactual.py`
- `analysis/run_single_frame_text_alignment_probe.py`
- `analysis/run_single_frame_box_mask_visualization.py`

这些脚本已经在代码层面补上，但仓库里**还没有看到与之对应的成体系输出目录或最终结果文档**。  
也就是说，**最近的主要进度更偏向“分析工具完善”，不是“新一轮大规模实验结果新增”。**

### 7.3 明显还没收口的部分

- 模块级 ablation 还没有形成稳定数值结论。
- 防御 / 鲁棒训练方向目前主要停留在方案与脚本层面，尚未看到完整对比表。
- 单帧文本对齐、阶段互换、component localizer 等方向还缺最终汇总结果。
- `analysis/inject_out/pred_inject_clean_det.json` 目前几乎为空，说明该方向还没真正完成。

## 8. 当前可以下的阶段性结论

如果只看“当前实验结果是否已经支持论文级主张”，那么目前最强的结论是：

1. **SAM3 在当前 SA-Co / YT1B 设置下，对 UAP 的确非常脆弱。**
2. **这种脆弱性主要来自图像主路径 / mask 路径，而不是 text 路径。**
3. **攻击强度增加会稳定导致指标单调下降，结果具有一致性。**
4. **机理上已经基本定位到共享 backbone、grounding head、score 门控和 mask 质量崩溃。**

如果看“项目进度”，那么当前状态可以概括为：

- **主结果已经完成。**
- **主解释已经基本完成。**
- **更细的定位实验和防御实验还在补。**

## 9. 建议的下一步优先级

建议后续优先按下面顺序推进：

1. 先把 `defense_priority_ablation` 跑出第一版正式结果，确认最有效防御点。
2. 再补 `stage_swap / component_localizer / counterfactual`，把“哪一层最关键”做成更硬的证据链。
3. 最后统一把近期新增脚本的输出补成一个总表，否则现在代码进度快于结果进度。

## 10. 一句话总结

当前项目已经证明：**SAM3 在该实验设定下会被 UAP 稳定击穿，且主因基本锁定在图像主干与检测/分割输出链路；主结论已完成，剩余工作主要是更细粒度归因和防御验证。**
