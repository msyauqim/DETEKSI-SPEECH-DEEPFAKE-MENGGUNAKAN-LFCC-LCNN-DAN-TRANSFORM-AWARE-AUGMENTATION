# src/aggregate_results.py
# Meringkas hasil eksperimen menjadi bentuk yang dilaporkan pada Bab 5, sesuai
# Subbab 3.8.3: "Setiap metrik dilaporkan sebagai rerata dan simpangan baku atas
# tiga seed."
#
# Masukan : results/eval_summary.csv  (ditulis oleh evaluate.py)
# Keluaran: results/tabel_hasil.csv   + tabel siap salin ke naskah
#
#   python src/aggregate_results.py
#   python src/aggregate_results.py --metric eer auc min_tdcf
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd


def skema(tag):
    """Turunkan nama skema pelatihan dari nama berkas checkpoint."""
    t = str(tag).replace(".pt", "")
    for k in ("baseline", "augmented"):
        if t.startswith(k):
            return k
    return t.rsplit("_seed", 1)[0]          # konfigurasi ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/eval_summary.csv")
    ap.add_argument("--metric", nargs="*", default=["eer", "auc", "min_tdcf"])
    ap.add_argument("--out", default="results/tabel_hasil.csv")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f"{args.results} belum ada. Jalankan src/evaluate.py lebih dahulu.")
    df = pd.read_csv(args.results)
    df["skema"] = df.checkpoint.map(skema)

    metrics = [m for m in args.metric if m in df.columns]
    kunci = ["skema", "components", "scenario", "condition", "rm"]
    kunci = [k for k in kunci if k in df.columns]

    g = df.groupby(kunci, dropna=False)
    agg = g[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agg.to_csv(args.out, index=False)

    # ---------------- tampilan ----------------
    print(f"sumber: {args.results}  ({len(df)} baris, "
          f"{df.checkpoint.nunique()} checkpoint)\n")
    head = f"{'skema':<22}{'fitur':<9}{'skenario':<26}{'RM':<7}{'n':>3}  "
    head += "  ".join(f"{m.upper():>18}" for m in metrics)
    print(head)
    print("-" * len(head))
    for r in agg.itertuples():
        n = int(getattr(r, f"{metrics[0]}_count"))
        baris = (f"{r.skema:<22}{str(getattr(r,'components','-')):<9}"
                 f"{str(getattr(r,'condition','-')):<26}"
                 f"{str(getattr(r,'rm','-')):<7}{n:>3}  ")
        sel = []
        for m in metrics:
            mu = getattr(r, f"{m}_mean"); sd = getattr(r, f"{m}_std")
            if np.isnan(mu):
                sel.append(f"{'-':>18}"); continue
            skala = 100.0 if m == "eer" else 1.0
            satuan = "%" if m == "eer" else ""
            sd_txt = "" if (np.isnan(sd) or n < 2) else f"±{sd*skala:.2f}"
            sel.append(f"{mu*skala:>10.2f}{satuan}{sd_txt:>7}")
        print(baris + "  ".join(sel))

    kurang = agg[agg[f"{metrics[0]}_count"] < 3]
    if len(kurang):
        print(f"\ncatatan: {len(kurang)} baris belum lengkap 3 seed "
              "(simpangan baku belum bermakna)")
    print(f"\ntabel -> {args.out}")


if __name__ == "__main__":
    main()
