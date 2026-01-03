import os
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model: Colab'daki ImprovedSignModel ile aynı
# (npy: (T, V, C) -> input: (B, C, T, V))
# -----------------------------
class ImprovedSignModel(nn.Module):
    def __init__(self, num_classes: int, num_frames: int = 64, num_joints: int = 51):
        super().__init__()

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128 * num_joints, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        self.num_joints = num_joints
        self.num_frames = num_frames

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, V)  # (B*T, C, V)
        x = self.spatial_conv(x)                        # (B*T, 128, V)
        x = x.reshape(B, T, -1).permute(0, 2, 1)        # (B, 128*V, T)
        x = self.temporal_conv(x).squeeze(-1)           # (B, 256)
        x = self.classifier(x)                          # (B, num_classes)
        return x


def resample_T(data: np.ndarray, target_T: int = 64) -> np.ndarray:
    """
    data: (T, V, C)
    T farklıysa lineer interpolasyonla 64'e getirir.
    """
    T, V, C = data.shape
    if T == target_T:
        return data

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_T)
    out = np.empty((target_T, V, C), dtype=np.float32)

    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(x_new, x_old, data[:, v, c].astype(np.float32))

    return out


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # Senin best_model.pth formatında genelde dict var: {'model_state_dict': ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        # bazen direkt state_dict kaydedilir
        state = ckpt
    model.load_state_dict(state, strict=True)
    return ckpt


def iter_npy_files(skeleton_root: str):
    """
    skeleton_root/
      abdomen/
        00333.npy
      another_class/
        00001.npy
    """
    for cls_name in os.listdir(skeleton_root):
        cls_dir = os.path.join(skeleton_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(".npy"):
                yield cls_name, os.path.join(cls_dir, fn)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_root", required=True, help=r"..\skeleton")
    ap.add_argument("--checkpoint", required=True, help=r"..\best_model.pth")
    ap.add_argument("--class_to_idx", required=True, help=r"..\class_to_idx.json")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.class_to_idx, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    device = torch.device(args.device)

    model = ImprovedSignModel(num_classes=num_classes).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    rows = []
    per_true = defaultdict(lambda: {
        "n": 0,
        "top1_correct": 0,
        "topk_correct": 0,
        "top1_conf_sum": 0.0,
        "top1_pred_counter": Counter(),
    })

    total = 0
    top1_correct = 0
    topk_correct = 0

    pbar = tqdm(list(iter_npy_files(args.skeleton_root)), desc="Evaluating")
    for true_cls_name, npy_path in pbar:
        if true_cls_name not in class_to_idx:
            # class_to_idx.json ile klasör isimleri uyuşmuyorsa bunu atlar
            continue

        true_idx = class_to_idx[true_cls_name]

        data = np.load(npy_path)
        if data.ndim != 3:
            raise ValueError(f"Beklenen (T,V,C). Hatalı shape: {data.shape} -> {npy_path}")

        data = data.astype(np.float32)

        # (T,V,C) kontrol/uyarlama
        # C en sonda ve 3 olmalı
        if data.shape[-1] != 3:
            raise ValueError(f"C=3 bekleniyor, geldi: {data.shape} -> {npy_path}")

        # T != 64 ise resample
        data = resample_T(data, target_T=64)

        # (1, C, T, V)
        x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).to(device)

        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

        # topk
        k = max(1, int(args.topk))
        topk_idx = torch.topk(probs, k=k, dim=0).indices.cpu().numpy().tolist()
        topk_prob = torch.topk(probs, k=k, dim=0).values.cpu().numpy().tolist()

        pred1 = topk_idx[0]
        conf1 = float(topk_prob[0])

        hit1 = int(pred1 == true_idx)
        hitk = int(true_idx in topk_idx)

        total += 1
        top1_correct += hit1
        topk_correct += hitk

        per_true[true_cls_name]["n"] += 1
        per_true[true_cls_name]["top1_correct"] += hit1
        per_true[true_cls_name]["topk_correct"] += hitk
        per_true[true_cls_name]["top1_conf_sum"] += conf1
        per_true[true_cls_name]["top1_pred_counter"][idx_to_class[pred1]] += 1

        rows.append({
            "true_class": true_cls_name,
            "file": os.path.basename(npy_path),
            "top1_pred": idx_to_class[pred1],
            "top1_conf": conf1,
            "top1_correct": hit1,
            "topk_correct": hitk,
            "topk_preds": ";".join([idx_to_class[i] for i in topk_idx]),
            "topk_confs": ";".join([f"{p:.4f}" for p in topk_prob]),
        })

        pbar.set_postfix({
            "Top1": f"{(top1_correct/max(1,total))*100:.2f}%",
            f"Top{k}": f"{(topk_correct/max(1,total))*100:.2f}%"
        })

    # --- Overall summary
    overall = {
        "total_samples": int(total),
        "top1_acc": float(top1_correct / max(1, total)),
        f"top{args.topk}_acc": float(topk_correct / max(1, total)),
    }

    # --- Per-class summary
    class_rows = []
    for cls, st in per_true.items():
        n = st["n"]
        if n == 0:
            continue
        top1 = st["top1_correct"] / n
        topk = st["topk_correct"] / n
        mean_conf = st["top1_conf_sum"] / n
        most_common_pred, most_common_cnt = st["top1_pred_counter"].most_common(1)[0]

        class_rows.append({
            "class": cls,
            "n": n,
            "top1_acc": top1,
            f"top{args.topk}_acc": topk,
            "mean_top1_conf": mean_conf,
            "most_common_top1_pred": most_common_pred,
            "most_common_top1_pred_ratio": most_common_cnt / n,
        })

    df_samples = pd.DataFrame(rows)
    df_classes = pd.DataFrame(class_rows).sort_values("top1_acc", ascending=False)

    # Kaydet
    df_samples.to_csv(os.path.join(args.out_dir, "per_sample_predictions.csv"), index=False, encoding="utf-8-sig")
    df_classes.to_csv(os.path.join(args.out_dir, "per_class_stats.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(args.out_dir, "overall_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2, ensure_ascii=False))

    print("\nEn iyi 20 sınıf:")
    print(df_classes.head(20)[["class", "n", "top1_acc", f"top{args.topk}_acc", "mean_top1_conf"]].to_string(index=False))

    print("\nEn zor 20 sınıf:")
    print(df_classes.tail(20)[["class", "n", "top1_acc", f"top{args.topk}_acc", "mean_top1_conf",
                              "most_common_top1_pred", "most_common_top1_pred_ratio"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "--skeleton_root", r"C:\Users\Acer\Desktop\wlasl_balanced_results\unisiggn_output\skeleton",
            "--checkpoint", r"C:\Users\Acer\Desktop\wlasl_balanced_results\best_model_balanced.pth",
            "--class_to_idx", r"C:\Users\Acer\Desktop\wlasl_balanced_results\unisiggn_output\class_to_idx.json",
            "--topk", "5",
            "--out_dir", r"C:\Users\Acer\Desktop\wlasl_balanced_results\eval_out",
        ]
    main()

