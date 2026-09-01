# src/compute_norm_stats.py
# Hitung rerata dan simpangan baku fitur LFCC dari partisi LATIH saja (Tabel 3.2).
# Statistik disimpan ke manifests/norm_stats.json dan dipakai train.py serta evaluate.py.
# Menghitung dari partisi latih saja adalah syarat agar tidak terjadi kebocoran informasi.
#
#   python src/compute_norm_stats.py --config configs/baseline.yaml --max-files 4000
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import yaml
from torch.utils.data import DataLoader

from device import DEVICE
from dataset import ASVDataset, collate
from features_lfcc import LFCCFeature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--max-files", type=int, default=4000,
                    help="jumlah berkas latih yang dicuplik untuk menaksir statistik")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)   # 0 tercepat di Apple Silicon
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    lf = cfg["lfcc"]
    sr, seconds = cfg["audio"]["target_sr"], cfg["audio"]["seconds"]
    out = args.out or lf.get("norm_stats") or "manifests/norm_stats.json"

    ds = ASVDataset(cfg["data"]["train_manifest"], training=False, augment=False,
                    sr=sr, seconds=seconds, subset=args.max_files)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=collate)

    lfcc = LFCCFeature(sample_rate=sr, n_filter=lf["n_filter"], n_lfcc=lf["n_lfcc"],
                   n_fft=lf["n_fft"], win_length=lf["win_length"],
                   hop_length=lf["hop_length"], log_lf=lf["log_lf"],
                   window=lf.get("window", "hamming"),
                   components=lf["components"], norm_stats=None).to(DEVICE)

    D = lfcc.out_dim
    # akumulasi di CPU dengan float64: MPS tidak mendukung presisi ganda
    total = torch.zeros(D, dtype=torch.float64)
    total_sq = torch.zeros(D, dtype=torch.float64)
    n = 0
    print(f"menghitung statistik dari {len(ds)} berkas latih | dim={D} | device={DEVICE}")
    with torch.no_grad():
        for bi, (wav, _, _, _) in enumerate(ld, 1):
            feat = lfcc(wav.to(DEVICE), normalize=False).squeeze(1)   # [B, D, F]
            x = feat.permute(0, 2, 1).reshape(-1, D).cpu().double()    # [B*F, D]
            total += x.sum(dim=0)
            total_sq += (x * x).sum(dim=0)
            n += x.shape[0]
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(ld)}  frame terkumpul {n:,}")

    mean = total / n
    var = (total_sq / n) - mean * mean
    std = var.clamp_min(1e-10).sqrt()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"mean": mean.tolist(), "std": std.tolist(),
               "n_frames": int(n), "n_files": len(ds),
               "components": lf["components"], "dim": D}, open(out, "w"))
    print(f"tersimpan -> {out}")
    print(f"  rerata: min {mean.min():.3f}  maks {mean.max():.3f}")
    print(f"  simpangan baku: min {std.min():.3f}  maks {std.max():.3f}")


if __name__ == "__main__":
    main()
