# src/benchmark.py
# Mengukur kecepatan nyata pelatihan pada perangkat ini, untuk menaksir
# kebutuhan waktu eksperimen penuh. Bukan bagian dari alur skripsi.
import os, sys, time, argparse, yaml
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from device import DEVICE
from dataset import ASVDataset, collate
from features_lfcc import LFCCFeature
from model_lcnn import LCNN

N_TRAIN, N_VAL = 25380, 24844


def bench(cfg, augment, workers, n, bs):
    lf = cfg["lfcc"]
    ds = ASVDataset(cfg["data"]["train_manifest"], training=True, augment=augment,
                    augment_cfg=cfg["augment"], subset=n, seed=2026)
    ld = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                    collate_fn=collate, drop_last=True)
    fx = LFCCFeature(n_filter=lf["n_filter"], n_lfcc=lf["n_lfcc"], n_fft=lf["n_fft"],
                     win_length=lf["win_length"], hop_length=lf["hop_length"],
                     window=lf.get("window", "hamming"),
                     components=lf["components"]).to(DEVICE)
    m = LCNN(n_feat=fx.out_dim, embedding_dim=80).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    it = iter(ld); next(it)                      # pemanasan
    t0, seen = time.time(), 0
    for wav, lab, _, _ in ld:
        wav, lab = wav.to(DEVICE), lab.to(DEVICE)
        loss = cross_entropy(m(fx(wav))[0], lab)
        opt.zero_grad(); loss.backward(); opt.step()
        seen += wav.shape[0]
    return seen / max(time.time() - t0, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=960)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    print(f"device={DEVICE} | workers={args.workers} | batch={args.batch_size} "
          f"| cuplikan {args.n} berkas\n")
    print(f"{'skema':<12}{'berkas/detik':>14}{'1 epoch':>12}{'50 epoch':>12}{'100 epoch':>12}")
    print("-" * 62)
    for augment in (False, True):
        r = bench(cfg, augment, args.workers, args.n, args.batch_size)
        ep = (N_TRAIN + N_VAL) / r / 60                       # menit
        print(f"{'augmented' if augment else 'clean-only':<12}{r:>14.1f}"
              f"{ep:>10.1f} m{ep*50/60:>10.1f} j{ep*100/60:>10.1f} j")


if __name__ == "__main__":
    main()
