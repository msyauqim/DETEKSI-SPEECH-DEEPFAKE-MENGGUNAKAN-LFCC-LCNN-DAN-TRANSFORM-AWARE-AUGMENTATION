# src/evaluate_multijendela.py
# Evaluasi lintas dataset dengan skor MULTI-JENDELA (analisis tambahan, bukan pengganti S4).
#
# METODE DITETAPKAN SEBELUM HASIL DILIHAT (21 Okt 2026), dan diterapkan sama pada semua model:
#   - rekaman dipotong menjadi jendela 4 detik dengan geser 2 detik (fungsi jendela() di
#     nilai_rekaman.py, sama dengan web demo); tiap jendela dinormalisasi puncak 1,0;
#   - rekaman <= 4 detik: satu jendela ditambal nol; 4-6 detik: satu jendela di tengah
#     (identik dengan evaluate.py), sehingga hanya rekaman > 6 detik yang skornya berubah;
#   - skor berkas = rerata fake_score (softmax kelas spoof) seluruh jendela;
#   - fitur, CMVN, dan bobot dari checkpoint (fungsi muat()), tanpa penyetelan apa pun.
# Data uji eksternal tidak dipakai untuk memilih metode, ambang, maupun model.
#
#   python src/evaluate_multijendela.py --checkpoint checkpoints/v2_validasi_terpisah/baseline_seed2026.pt \
#          --manifest manifests/in_the_wild.csv --condition in_the_wild --out-dir results/v2_validasi_terpisah/multijendela
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from device import DEVICE
from metrics import compute_eer
from nilai_rekaman import jendela, muat


class Berkas(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        potong, _, _ = jendela(self.df.file_path[i])
        return torch.stack([w for _, w in potong]), i      # [n_jendela, 1, T]


def kumpul(batch):
    x = torch.cat([b[0] for b in batch])
    n = [b[0].shape[0] for b in batch]
    idx = [b[1] for b in batch]
    return x, n, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="uji cepat pada sebagian berkas")
    args = ap.parse_args()

    tag = os.path.basename(args.checkpoint).replace(".pt", "")
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = pd.concat([g.sample(min(len(g), args.limit // 2), random_state=0)
                        for _, g in df.groupby("label")])
    model, lfcc, _ = muat(args.checkpoint)
    ld = DataLoader(Berkas(df), batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=kumpul)

    # Jumlah jendela per batch berubah-ubah; MPS menyusun ulang grafik untuk tiap ukuran baru
    # sehingga sangat lambat. Jendela karena itu dinilai dalam potongan berukuran TETAP;
    # potongan terakhir diisi salinan jendela terakhir lalu baris isian dibuang.
    CH = 64

    def nilai_jendela(x):
        hasil = []
        for i in range(0, len(x), CH):
            b = x[i:i + CH]
            k = len(b)
            if k < CH:
                b = torch.cat([b, b[-1:].expand(CH - k, *b.shape[1:])])
            hasil.append(model(lfcc(b.to(DEVICE)))[0].softmax(dim=1)[:k, 1].cpu())
        return torch.cat(hasil).numpy()

    skor = np.zeros(len(df))
    njen = np.zeros(len(df), dtype=int)
    with torch.no_grad():
        for bi, (x, n, idx) in enumerate(ld, 1):
            p = nilai_jendela(x)
            pos = 0
            for k, i in zip(n, idx):
                skor[i] = p[pos:pos + k].mean()
                njen[i] = k
                pos += k
            if bi % 200 == 0:
                print(f"  {bi * args.batch_size:,}/{len(df):,} berkas", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = df[["source_file_id", "label", "attack_id"]].reset_index(drop=True).copy()
    out["fake_score"] = skor
    out["n_jendela"] = njen
    out.to_csv(os.path.join(args.out_dir, f"scores_{args.condition}_{tag}.csv"), index=False)

    y = out.label.values
    eer = compute_eer(y, skor)[0]
    auc = roc_auc_score(y, skor)
    banyak = (njen > 1).mean()
    print(f"{tag} | {args.condition} | multi-jendela EER {eer * 100:.2f}% | AUC {auc:.3f} | "
          f"berkas > 1 jendela {banyak * 100:.0f}%")

    ringkas = os.path.join(args.out_dir, "ringkasan_multijendela.csv")
    baris = pd.DataFrame([{"checkpoint": tag, "condition": args.condition, "eer": eer, "auc": auc,
                           "frac_multi_jendela": banyak, "n": len(out),
                           "limit": args.limit or ""}])
    baris.to_csv(ringkas, mode="a", header=not os.path.exists(ringkas), index=False)


if __name__ == "__main__":
    main()
