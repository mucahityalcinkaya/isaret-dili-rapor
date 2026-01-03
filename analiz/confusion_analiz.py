import os
import argparse
from collections import Counter, defaultdict

import pandas as pd


def split_topk(s: str):
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def compute_rank(true_cls: str, topk_list):
    # 1-based rank; yoksa None
    for i, c in enumerate(topk_list, start=1):
        if c == true_cls:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"C:\Users\Acer\Desktop\wlasl_balanced_results\eval_out\per_sample_predictions.csv")
    ap.add_argument("--out_dir", default=r"C:\Users\Acer\Desktop\wlasl_balanced_results\eval_out\analiz")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=10, help="çok az örneği olan sınıfları filtrelemek için")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Beklenen kolonlar: true_class, top1_pred, top1_conf, top1_correct, topk_preds
    needed = {"true_class", "top1_pred", "top1_correct", "topk_preds"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"CSV içinde eksik kolon var: {miss}. Mevcut kolonlar: {list(df.columns)}")

    df["topk_list"] = df["topk_preds"].apply(split_topk)

    # ----------------------------
    # 1) Genel: en çok karışan çiftler (true -> yanlış top1)
    # ----------------------------
    wrong = df[df["top1_correct"] == 0].copy()
    pair_counter = Counter(zip(wrong["true_class"], wrong["top1_pred"]))
    pairs = [
        {"true_class": t, "pred_class": p, "count": c}
        for (t, p), c in pair_counter.most_common()
    ]
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(os.path.join(args.out_dir, "most_confused_pairs_top1.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------
    # 2) Sınıf bazlı: top1 doğruyken 2. sırada en sık gelen kelime
    # (senin "abdomen doğru ama 2. hep stomach" istediğin analiz)
    # ----------------------------
    second_when_correct = defaultdict(Counter)
    third_when_correct = defaultdict(Counter)

    correct_rows = df[df["top1_correct"] == 1]
    for _, r in correct_rows.iterrows():
        true_cls = r["true_class"]
        tk = r["topk_list"]
        if len(tk) >= 2:
            second_when_correct[true_cls][tk[1]] += 1
        if len(tk) >= 3:
            third_when_correct[true_cls][tk[2]] += 1

    rows2 = []
    for true_cls, cnt in second_when_correct.items():
        total = cnt.total()
        if total < args.min_count:
            continue
        pred2, c2 = cnt.most_common(1)[0]
        rows2.append({
            "true_class": true_cls,
            "n_top1_correct": int(total),
            "most_common_2nd_pred": pred2,
            "count": int(c2),
            "ratio": float(c2 / total),
        })

    df_second = pd.DataFrame(rows2).sort_values(["ratio", "count"], ascending=False)
    df_second.to_csv(os.path.join(args.out_dir, "second_pred_when_top1_correct.csv"), index=False, encoding="utf-8-sig")

    rows3 = []
    for true_cls, cnt in third_when_correct.items():
        total = cnt.total()
        if total < args.min_count:
            continue
        pred3, c3 = cnt.most_common(1)[0]
        rows3.append({
            "true_class": true_cls,
            "n_top1_correct": int(total),
            "most_common_3rd_pred": pred3,
            "count": int(c3),
            "ratio": float(c3 / total),
        })

    df_third = pd.DataFrame(rows3).sort_values(["ratio", "count"], ascending=False)
    df_third.to_csv(os.path.join(args.out_dir, "third_pred_when_top1_correct.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------
    # 3) Sınıf bazlı: top1 yanlışken en çok hangi kelimeye gidiyor?
    # ----------------------------
    wrong_to = defaultdict(Counter)
    wrong_counts = df[df["top1_correct"] == 0].groupby("true_class").size().to_dict()

    for _, r in wrong.iterrows():
        wrong_to[r["true_class"]][r["top1_pred"]] += 1

    rowsw = []
    for true_cls, cnt in wrong_to.items():
        total_wrong = cnt.total()
        if total_wrong < args.min_count:
            continue
        pred, c = cnt.most_common(1)[0]
        rowsw.append({
            "true_class": true_cls,
            "n_wrong": int(total_wrong),
            "most_common_wrong_top1_pred": pred,
            "count": int(c),
            "ratio": float(c / total_wrong),
        })

    df_wrong_to = pd.DataFrame(rowsw).sort_values(["ratio", "count"], ascending=False)
    df_wrong_to.to_csv(os.path.join(args.out_dir, "most_common_wrong_top1_per_class.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------
    # 4) Rank analizi: doğru cevap top-k içinde en çok kaçıncı sırada geliyor?
    # ----------------------------
    ranks = []
    for _, r in df.iterrows():
        rk = compute_rank(r["true_class"], r["topk_list"][:args.topk])
        ranks.append(rk if rk is not None else 0)  # 0 = topk içinde yok
    df["true_rank_in_topk"] = ranks

    # sınıf bazlı rank dağılımı
    rank_rows = []
    for true_cls, g in df.groupby("true_class"):
        n = len(g)
        if n < args.min_count:
            continue
        dist = Counter(g["true_rank_in_topk"])
        hitk = 1.0 - (dist.get(0, 0) / n)
        # en çok görülen rank (0 dahil)
        most_rank, most_cnt = dist.most_common(1)[0]
        rank_rows.append({
            "true_class": true_cls,
            "n": n,
            f"top{args.topk}_hit_rate": hitk,
            "most_common_rank(0=miss)": int(most_rank),
            "most_common_rank_ratio": float(most_cnt / n),
            "rank1": int(dist.get(1, 0)),
            "rank2": int(dist.get(2, 0)),
            "rank3": int(dist.get(3, 0)),
            "rank4": int(dist.get(4, 0)),
            "rank5": int(dist.get(5, 0)),
            "miss(0)": int(dist.get(0, 0)),
        })

    df_rank = pd.DataFrame(rank_rows).sort_values([f"top{args.topk}_hit_rate", "n"], ascending=False)
    df_rank.to_csv(os.path.join(args.out_dir, f"rank_stats_top{args.topk}.csv"), index=False, encoding="utf-8-sig")

    # ----------------------------
    # 5) Confusion matrix (top1) CSV
    # ----------------------------
    cm = pd.crosstab(df["true_class"], df["top1_pred"])
    cm.to_csv(os.path.join(args.out_dir, "confusion_matrix_top1.csv"), encoding="utf-8-sig")

    # ----------------------------
    # Özet yazdır
    # ----------------------------
    print("Bitti.")
    print("Çıktılar klasörü:", args.out_dir)
    print("- most_confused_pairs_top1.csv")
    print("- second_pred_when_top1_correct.csv  (senin 'abdomen doğru ama 2. stomach' gibi)")
    print("- third_pred_when_top1_correct.csv")
    print("- most_common_wrong_top1_per_class.csv")
    print(f"- rank_stats_top{args.topk}.csv")
    print("- confusion_matrix_top1.csv")


if __name__ == "__main__":
    main()
