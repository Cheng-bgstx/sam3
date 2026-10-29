# sam3

SAM3 通用对抗攻击与评估代码。

## 攻击入口

- `uap_attack_sam3_image.py`：图像攻击。
- `uap_attack_sam3_video.py`：视频攻击，包含 DarkSAM、AdvCLIP 适配与组合损失。
- `uap_attack_sam3_video_advclip_only.py`：AdvCLIP 独立入口。
- `attack_setting_sam3.py`：模型适配、前向及损失实现。

## 推理与分析

`sam3/run_inference_eval.py`、`scripts/eval/` 提供推理评估；
`verify_sam3_attack_causes.py` 提供扰动强度消融与诊断。

## 环境与迁移

`environment.yml` 为原环境快照，`requirements.txt` 为早期依赖列表。
尚未验证全新机器安装；运行前修改代码与 YAML 中的本机绝对路径及 GPU 配置。
模型权重、数据媒体、大型标注、扰动及预测结果不提交 Git。
权重默认路径为 `sam3/checkpoints/sam3.pt`，数据路径见 `sam3_util.py`。

历史结果见 `sum.md`、`EXPERIMENTAL_RESULTS_SUMMARY.md`，不代表统一协议下的完整对比。
原项目说明见 `README_SAM2_ORIGINAL.md`；迁移范围见 `MIGRATION.md`。
