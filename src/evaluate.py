# src/evaluate.py
# Evaluasi checkpoint pada skenario S1-S4 (Tabel 3.6 proposal).
#
#   S1 Seen attack                        : evaluasi LA, serangan A16 dan A19     (RM 1)
#   S2 Unseen attack dalam dataset sama   : evaluasi LA, A07-A15, A17, A18        (RM 1, 2)
#   S3 Stress-test transformasi           : evaluasi LA + transformasi terkendali (RM 3)
#   S4 Unseen generator lintas dataset    : WaveFake / In-the-Wild                (RM 1, 2)
#
# Contoh:
#   python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S1
#   python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S2 --per-attack
#   python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S3
#   python src/evaluate.py --checkpoint checkpoints/augmented_seed2026.pt \
#          --scenario S4 --manifest manifests/wavefake.csv --condition wavefake
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from device import DEVICE
from dataset import ASVDataset, collate
from features_lfcc import LFCCFeature
from model_lcnn import LCNN
from metrics import compute_metrics, min_tdcf_from_manifest
from augment import STRESS_LEVELS
from train import build_lfcc, cm_score


def score(model, lfcc, manifest, sr, seconds, bs, workers, attacks=None, stress=None):
    ds = ASVDataset(manifest, training=False, augment=False, sr=sr, seconds=seconds,
                    attacks=attacks, stress=stress)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate)
    rows = []
    model.eval()
    with torch.no_grad():
        for wav, label, attacks_b, file_ids in ld:
            logits, _ = model(lfcc(wav.to(DEVICE)))
            llr = cm_score(logits)                 # log p(bona) - log p(spoof)
            prob = logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
            for fid, y, a, sc, pr in zip(file_ids, label.numpy(), attacks_b, llr, prob):
                rows.append({"file_id": fid, "source_file_id": fid, "label": int(y),
                             "attack_id": a,
                             "cm_score": float(sc),      # Subbab 2.4.2: log p(bona) - log p(spoof)
                             "fake_score": float(pr)})   # Panduan: softmax kelas spoof
    return pd.DataFrame(rows)


def report(df, name, asv_file=None, threshold=None):
    m = compute_metrics(df.label.values, df.fake_score.values,
                        threshold=0.0 if threshold is None else threshold)
    t = min_tdcf_from_manifest(df.label.values, df.fake_score.values, asv_file)
    line = (f"{name:<22} EER {m['eer']*100:6.2f}%  AUC {m['auc']:.4f}  F1 {m['f1']:.3f}  "
            f"balAcc {m['balanced_accuracy']:.3f}")
    if not np.isnan(t):
        line += f"  min t-DCF {t:.4f}"
    print(line + f"   (n={len(df):,})")
    return {**m, "min_tdcf": t, "n": len(df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scenario", choices=["S1", "S2", "S3", "S4"], default="S2")
    ap.add_argument("--manifest", default=None, help="wajib untuk S4")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)   # 0 tercepat di Apple Silicon
    ap.add_argument("--per-attack", action="store_true")
    ap.add_argument("--stress-kind", nargs="*", default=None,
                    help="batasi jenis transformasi pada S3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    sr, seconds = cfg["audio"]["target_sr"], cfg["audio"]["seconds"]
    comp = ck.get("components", cfg["lfcc"]["components"])

    lfcc = build_lfcc(cfg, comp).to(DEVICE)
    model = LCNN(n_feat=ck.get("n_feat", lfcc.out_dim),
                 embedding_dim=cfg["model"]["embedding_dim"],
                 dropout=cfg["model"].get("dropout", 0.7)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    seed = ck.get("seed", "?")
    thr = ck.get("val_threshold")
    asv_file = cfg["paths"].get("asv_scores_eval")

    sc = cfg.get("scenarios", {}).get(args.scenario, {})
    rm = sc.get("rm", "-")
    print(f"== {args.scenario} {sc.get('nama','')} (RM {rm}) | {os.path.basename(args.checkpoint)} "
          f"(seed {seed}, val_EER {ck.get('val_eer', float('nan'))*100:.2f}%, fitur {comp}) "
          f"| device={DEVICE} ==")

    eval_manifest = cfg["data"].get("eval_manifest", "manifests/eval_protocol.csv")
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    rows_summary = []

    # ------------------------------------------------ S1 / S2
    if args.scenario in ("S1", "S2"):
        attacks = cfg["attacks"]["known_eval"] if args.scenario == "S1" else cfg["attacks"]["unseen"]
        cond = args.condition or sc.get("kondisi", args.scenario.lower())
        print(f"serangan: {attacks}")
        df = score(model, lfcc, eval_manifest, sr, seconds, args.batch_size, args.workers,
                   attacks=attacks)
        m = report(df, cond, asv_file, thr)
        rows_summary.append({"scenario": args.scenario, "condition": cond, **m})
        out = args.out or os.path.join(results_dir, f"scores_{cond}_seed{seed}.csv")
        df.assign(condition=cond, seed=seed).to_csv(out, index=False)
        print(f"skor -> {out}")

        if args.per_attack:
            bona = df[df.label == 0]
            print("\nEER per serangan (tiap serangan melawan seluruh bona fide):")
            for a in sorted(df[df.label == 1].attack_id.unique()):
                sub = pd.concat([bona, df[(df.label == 1) & (df.attack_id == a)]])
                if sub.label.nunique() < 2:
                    continue
                ma = compute_metrics(sub.label.values, sub.fake_score.values)
                print(f"  {a}: EER {ma['eer']*100:6.2f}%  (n_spoof={int((df.attack_id==a).sum()):,})")

    # ------------------------------------------------ S3
    elif args.scenario == "S3":
        attacks = cfg["attacks"]["unseen"]
        kinds = args.stress_kind or list(STRESS_LEVELS.keys())
        print(f"serangan: unseen | transformasi: {kinds}\n")
        df0 = score(model, lfcc, eval_manifest, sr, seconds, args.batch_size,
                    args.workers, attacks=attacks)
        m0 = report(df0, "tanpa transformasi", asv_file, thr)
        rows_summary.append({"scenario": "S3", "condition": "clean", "kind": "-",
                             "level": -1, "param": "-", **m0})
        for kind in kinds:
            for lv, val in enumerate(STRESS_LEVELS[kind]):
                d = score(model, lfcc, eval_manifest, sr, seconds, args.batch_size,
                          args.workers, attacks=attacks, stress=(kind, lv))
                m = report(d, f"{kind} L{lv+1} ({val})", asv_file, thr)
                rows_summary.append({"scenario": "S3", "condition": f"{kind}_L{lv+1}",
                                     "kind": kind, "level": lv + 1, "param": val, **m})

    # ------------------------------------------------ S4
    else:
        if not args.manifest:
            ap.error("skenario S4 memerlukan --manifest")
        cond = args.condition or os.path.splitext(os.path.basename(args.manifest))[0]
        df = score(model, lfcc, args.manifest, sr, seconds, args.batch_size, args.workers)
        m = report(df, cond, None, thr)           # min t-DCF tidak berlaku di luar ASVspoof
        rows_summary.append({"scenario": "S4", "condition": cond, **m})
        out = args.out or os.path.join(results_dir, f"scores_{cond}_seed{seed}.csv")
        df.assign(condition=cond, seed=seed).to_csv(out, index=False)
        print(f"skor -> {out}")

    # ------------------------------------------------ ringkasan
    # Kolom tetap (superset S1/S2/S3/S4) agar skema tidak berubah antar-pemanggilan;
    # menyisipkan header baru di tengah berkas akan merusak parsing baris lama.
    COLS = ["checkpoint", "seed", "components", "rm", "scenario", "condition",
            "kind", "level", "param", "eer", "eer_threshold", "auc", "f1",
            "balanced_accuracy", "min_tdcf", "n"]
    summary = os.path.join(results_dir, "eval_summary.csv")
    s = pd.DataFrame(rows_summary)
    s.insert(0, "checkpoint", os.path.basename(args.checkpoint))
    s.insert(1, "seed", seed)
    s.insert(2, "components", comp)
    s.insert(3, "rm", rm)
    for c, default in (("kind", "-"), ("level", -1), ("param", "-")):
        if c not in s.columns:
            s[c] = default
    s = s.reindex(columns=COLS)
    s.to_csv(summary, mode="a", header=not os.path.exists(summary), index=False)
    print(f"\nringkasan -> {summary}")


if __name__ == "__main__":
    main()
