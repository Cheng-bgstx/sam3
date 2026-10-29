#!/bin/bash
# =============================================================================
# 一个 UAP 的完整 scale 消融：完全复刻原论文 train 推理（train.py val 模式）
# =============================================================================
# 流程: 对每个 scale -> 1) train.py -c ... --use-cluster 0 推理（Trainer val）;
#       2) saco_veval_eval.py one 得到完整指标
# 输出: OUT_DIR/scale_X/preds/saco_veval_sav_test_split.json,
#       OUT_DIR/metrics/metrics_scale_X.json
#
# 用法（直接运行即可，生成一个 UAP 的全部 scale 结果）:
#   cd /root/data2/fyc/fyc/UAP-SAM2-main && bash scripts/run_scale_ablation_sav.sh
#
# 或指定 UAP 与输出目录:
#   REPO_ROOT=/path/to/UAP-SAM2-main \
#   UAP_PATH=/path/to/uap.pth \
#   OUT_DIR=/path/to/scale_ablation_sav \
#   SCALES="0 0.2 0.3 0.5 0.7 1.0" \
#   bash scripts/run_scale_ablation_sav.sh
# =============================================================================

set -e

REPO_ROOT="/root/data2/fyc/fyc/UAP-SAM2-main"
# 一个 UAP 文件，脚本会针对该 UAP 跑所有 scale
UAP_PATH="${UAP_PATH:-${REPO_ROOT}/uap_file/SA-Co_sam3_video_saco_yt1b_0_fusion_quan.pth}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scale_ablation_yt1b_quan}"
SCALES="${SCALES:-0 0.2 0.3 0.4 0.5 0.6 0.8 1.0}"

# 与 saco_veval_sav_test_split.yaml 一致（用于 eval 步骤的 GT）
GT_ANNOT="${GT_ANNOT:-${REPO_ROOT}/data/annotation/saco_veval_yt1b_test_split.json}"
# 使用 SAV test split 的 config
# train.py 的 -c 需要相对 sam3.train 的 configs 路径（含 configs/ 前缀）
CONFIG="${CONFIG:-configs/saco_video_evals/saco_veval_yt1b_test_split}"
DUMP_NAME="${DUMP_NAME:-saco_veval_yt1b_test_split}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR/metrics"

echo "========== Scale Ablation (Trainer val 推理 + saco_veval_eval) =========="
echo "  REPO_ROOT:  $REPO_ROOT"
echo "  UAP_PATH:   $UAP_PATH"
echo "  OUT_DIR:    $OUT_DIR"
echo "  CONFIG:     $CONFIG"
echo "  GT_ANNOT:   $GT_ANNOT"
echo "  SCALES:     $SCALES"
echo "===================================================="

for scale in $SCALES; do
    echo ""
    echo ">>> Scale = $scale"
    scale_dir="$OUT_DIR/scale_${scale}"
    pred_file="$scale_dir/preds/${DUMP_NAME}.json"
    eval_res_file="$OUT_DIR/metrics/metrics_scale_${scale}.json"

    # Step 1: 使用 train.py（Trainer val）推理，与原论文一致
    if [ "$scale" = "0" ] || [ "$scale" = "0.0" ]; then
        echo "  [1/2] Inference (train.py val, no UAP)..."
        # 使用不存在的路径，Trainer 会跳过加载 UAP（uap_tensor=None）
        PYTHONPATH=. python sam3/train/train.py \
            -c "$CONFIG" \
            --use-cluster 0 \
            launcher.gpus_per_node=1 \
            launcher.num_nodes=1 \
            launcher.experiment_log_dir="$scale_dir" \
            'paths.uap_path=/nonexistent_uap_scale_0'
    else
        echo "  [1/2] Inference (train.py val, UAP scale=$scale)..."
        PYTHONPATH=. python sam3/train/train.py \
            -c "$CONFIG" \
            --use-cluster 0 \
            launcher.gpus_per_node=1 \
            launcher.num_nodes=1 \
            launcher.experiment_log_dir="$scale_dir" \
            paths.uap_path="$UAP_PATH" \
            trainer.uap_scale="$scale"
    fi

    # Step 2: 完整评估（saco_veval_eval.py one）
    if [ ! -f "$pred_file" ]; then
        echo "  [WARN] Pred file not found: $pred_file"
        continue
    fi
    echo "  [2/2] Evaluation (saco_veval_eval.py one)..."
    PYTHONPATH=. python sam3/eval/saco_veval_eval.py one \
        --gt_annot_file "$GT_ANNOT" \
        --pred_file "$pred_file" \
        --eval_res_file "$eval_res_file"

    echo "  Done: pred -> $pred_file, metrics -> $eval_res_file"
done

echo ""
echo "========== Summary =========="
echo "scale\tvideo_mask_mAP_50_95"
for scale in $SCALES; do
    mf="$OUT_DIR/metrics/metrics_scale_${scale}.json"
    if [ -f "$mf" ]; then
        m=$(python -c "import json; d=json.load(open('$mf')); print(d.get('dataset_results',{}).get('video_mask_mAP_50_95','N/A'))" 2>/dev/null || echo "N/A")
        echo "$scale\t$m"
    fi
done
echo ""
echo "Preds: $OUT_DIR/scale_*/preds/"
echo "Metrics: $OUT_DIR/metrics/"
