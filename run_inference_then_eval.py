#!/usr/bin/env python3
"""
集成推理 + 评估：先跑 train.py（推理模式），再跑 saco_veval_eval.py。
不修改现有 train/eval 代码，仅串联命令并统一路径约定。

路径约定（与 YAML 一致）：
  - 推理输出: paths.experiment_log_dir/preds/<dump_file_name>.json
  - 评估结果: paths.experiment_log_dir/metrics.json（或通过 --eval_res_file 指定）

用法示例：
  PYTHONPATH=. python run_inference_then_eval.py \\
    --config configs/saco_video_evals/saco_veval_yt1b_test_split.yaml \\
    --use-cluster 0 --num-gpus 1 \\
    paths.uap_path=/path/to/uap.pth \\
    paths.experiment_log_dir=/path/to/attack/SA-Co_xxx \\
    [--dump_file_name saco_veval_yt1b_test_split] \\
    [--gt_annot_file /path/to/saco_veval_yt1b_test_split.json] \\
    [--eval_res_file /path/to/metrics.json]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="集成推理(train.py) + 评估(saco_veval_eval.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 推理：与 train.py 一致
    parser.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help="Hydra 配置名（相对 sam3.train 包根目录），如 configs/saco_video_evals/saco_veval_yt1b_test_split.yaml",
    )
    parser.add_argument("--use-cluster", type=int, default=0, help="0=本地, 1=集群")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=None)
    # 由本脚本推导的路径（可与 YAML 中 paths 一致）
    parser.add_argument(
        "--dump_file_name",
        type=str,
        default="saco_veval_yt1b_test_split",
        help="推理输出 JSON 文件名（无 .json），与 YAML 中 paths.dump_file_name 一致则 pred 路径正确",
    )
    parser.add_argument(
        "--gt_annot_file",
        type=str,
        default=None,
        help="评估用 GT 标注 JSON；不指定则用 data/annotation/<dump_file_name>.json",
    )
    parser.add_argument(
        "--eval_res_file",
        type=str,
        default=None,
        help="评估结果输出路径；不指定则用 experiment_log_dir/metrics.json",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="只跑推理，不跑评估",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不执行",
    )
    args, overrides = parser.parse_known_args()

    # 从 overrides 中解析 paths.experiment_log_dir（供推导 pred_file / eval_res_file）
    experiment_log_dir = None
    for o in overrides:
        if o.startswith("paths.experiment_log_dir="):
            experiment_log_dir = o.split("=", 1)[1].strip()
            break
    if not experiment_log_dir:
        print("请在 overrides 中指定 paths.experiment_log_dir=...", file=sys.stderr)
        sys.exit(1)
    experiment_log_dir = os.path.abspath(experiment_log_dir)

    pred_file = os.path.join(experiment_log_dir, "preds", args.dump_file_name + ".json")
    gt_file = args.gt_annot_file
    if not gt_file:
        gt_file = str(root / "data" / "annotation" / (args.dump_file_name + ".json"))
    eval_res_file = args.eval_res_file
    if not eval_res_file:
        eval_res_file = os.path.join(experiment_log_dir, "metrics.json")

    env = os.environ.copy()
    if "PYTHONPATH" not in env or not env["PYTHONPATH"]:
        env["PYTHONPATH"] = str(root)
    else:
        env["PYTHONPATH"] = str(root) + os.pathsep + env["PYTHONPATH"]

    # 1) 推理
    train_cmd = [
        sys.executable,
        str(root / "sam3" / "train" / "train.py"),
        "-c", args.config,
        "--use-cluster", str(args.use_cluster),
        "--num-gpus", str(args.num_gpus),
    ]
    if args.num_nodes is not None:
        train_cmd += ["--num-nodes", str(args.num_nodes)]
    train_cmd += overrides

    print("Step 1 — 推理 (train.py)")
    print("  预测输出将写入:", pred_file)
    print("  命令:", " ".join(train_cmd))
    if args.dry_run:
        print("[DRY-RUN] 未执行推理")
    else:
        ret = subprocess.run(train_cmd, env=env, cwd=str(root))
        if ret.returncode != 0:
            print("推理失败，退出码:", ret.returncode, file=sys.stderr)
            sys.exit(ret.returncode)

    if args.skip_eval:
        print("已跳过评估 (--skip-eval)")
        return

    if not os.path.isfile(pred_file):
        print("未找到预测文件:", pred_file, "，跳过评估", file=sys.stderr)
        sys.exit(1)

    # 2) 评估
    eval_cmd = [
        sys.executable,
        str(root / "sam3" / "eval" / "saco_veval_eval.py"),
        "one",
        "--gt_annot_file", gt_file,
        "--pred_file", pred_file,
        "--eval_res_file", eval_res_file,
    ]
    print("Step 2 — 评估 (saco_veval_eval.py)")
    print("  GT:", gt_file)
    print("  Pred:", pred_file)
    print("  结果:", eval_res_file)
    print("  命令:", " ".join(eval_cmd))
    if args.dry_run:
        print("[DRY-RUN] 未执行评估")
        return
    ret = subprocess.run(eval_cmd, env=env, cwd=str(root))
    if ret.returncode != 0:
        print("评估失败，退出码:", ret.returncode, file=sys.stderr)
        sys.exit(ret.returncode)
    print("评估结果已写入:", eval_res_file)


if __name__ == "__main__":
    main()
