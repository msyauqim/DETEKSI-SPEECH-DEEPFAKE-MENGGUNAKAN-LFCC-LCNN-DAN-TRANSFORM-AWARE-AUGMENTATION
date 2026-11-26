# src/audit_augment.py
# Audit transform-aware augmentation sesuai kriteria selesai Tahap 5 Panduan
# Pelaksanaan Skripsi:
#   - "Distribusi augmentation tercatat"
#   - "Minimal 100 file diperiksa secara visual dan auditori"
#   - "Tidak ada clipping berat"
#
# Bukan bagian alur training (dataset.py sengaja membuang label transformasi
# demi kecepatan) -- skrip ini yang mencatatnya, plus mengekspor pasangan
# audio asli/teraugmentasi + plot bentuk gelombang untuk diperiksa manual.
#
#   python src/audit_augment.py                    # 200 file untuk distribusi,
#                                                    # 100 pertama diekspor utk didengar/dilihat
#   python src/audit_augment.py --n 500 --n-samples 150
#   python src/audit_augment.py --include-worst   # + 10 kasus clipping terburuk
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audio_io import load_audio_4s
from augment import TransformAwareAugment


def audit(df, aug, n_files):
    rows = []
    waves = {}
    for i, row in df.head(n_files).iterrows():
        wav = load_audio_4s(row.file_path)
        wav_aug, applied = aug(wav)
        clipped_frac = float((wav_aug.abs() >= 1.0 - 1e-6).float().mean())
        rows.append({
            "source_file_id": row.source_file_id,
            "label": row.label,
            "attack_id": row.get("attack_id", "-"),
            "applied": ";".join(applied),
            "n_transforms": len(applied) if applied != ["clean"] else 0,
            "clipped_frac": clipped_frac,
        })
        waves[row.source_file_id] = (wav, wav_aug, applied)
    return pd.DataFrame(rows), waves


def print_summary(rep, cfg_augment):
    n = len(rep)
    print(f"\n=== distribusi augmentation ({n} file) ===")
    targets = {"gain": cfg_augment.get("p_gain"), "noise": cfg_augment.get("p_noise"),
               "resample": cfg_augment.get("p_resample"), "lowpass": cfg_augment.get("p_lowpass"),
               "reverb": cfg_augment.get("p_reverb")}
    for name, p_target in targets.items():
        cnt = rep.applied.str.contains(name).sum()
        frac = cnt / n
        print(f"  {name:10s} {cnt:4d}/{n}  = {frac:.3f}  (target p={p_target})")
    n_clean = (rep.n_transforms == 0).sum()
    print(f"  {'clean':10s} {n_clean:4d}/{n}  = {n_clean/n:.3f}  (tanpa transformasi sama sekali)")

    print(f"\n=== indikasi clipping ===")
    n_any = (rep.clipped_frac > 0).sum()
    n_heavy = (rep.clipped_frac > 0.01).sum()
    print(f"  file dgn >=1 sampel terpotong (clamp -1..1) : {n_any}/{n}")
    print(f"  file dgn >1% durasi terpotong (BERAT)        : {n_heavy}/{n}")
    if n_heavy:
        print("  PERINGATAN: ada file dengan clipping berat, periksa manifest & rentang parameter.")
    else:
        print("  tidak ada clipping berat -- kriteria selesai Panduan terpenuhi.")
    worst = rep.nlargest(5, "clipped_frac")[["source_file_id", "applied", "clipped_frac"]]
    print("\n  5 file dengan proporsi terpotong terbesar:")
    for _, r in worst.iterrows():
        print(f"    {r.clipped_frac*100:6.3f}%  {r.applied:40s} {r.source_file_id}")


def export_samples(df, waves, out_dir, n_samples, rep=None, n_worst=0, sr=16000):
    os.makedirs(out_dir, exist_ok=True)
    ids = df.head(n_samples).source_file_id.tolist()

    # Sampel acak di atas bisa melewatkan kasus clipping terburuk, padahal itu
    # yang paling perlu dinilai untuk kriteria "tidak ada clipping berat".
    # Tambahkan kasus ekstrem secara sengaja (purposive), di LUAR sampel acak.
    ids_worst = []
    if n_worst and rep is not None:
        acak = set(ids)
        kandidat = rep.nlargest(n_worst + len(acak), "clipped_frac")
        for fid in kandidat.source_file_id:
            if fid not in acak and fid not in ids_worst:
                ids_worst.append(fid)
            if len(ids_worst) >= n_worst:
                break
        ids = ids + ids_worst

    for fid in ids:
        wav, wav_aug, applied = waves[fid]
        sf.write(os.path.join(out_dir, f"{fid}_asli.wav"), wav.squeeze(0).numpy(), sr)
        sf.write(os.path.join(out_dir, f"{fid}_augmented.wav"), wav_aug.squeeze(0).numpy(), sr)

        fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True, sharey=True)
        t = np.arange(wav.shape[-1]) / sr
        axes[0].plot(t, wav.squeeze(0).numpy(), linewidth=0.5)
        axes[0].set_title(f"{fid} -- asli")
        axes[1].plot(t, wav_aug.squeeze(0).numpy(), linewidth=0.5, color="darkorange")
        axes[1].set_title(f"{fid} -- augmented ({', '.join(applied)})")
        axes[1].set_xlabel("detik")
        for ax in axes:
            ax.set_ylabel("amplitudo")
            ax.axhline(1.0, color="red", linewidth=0.5, linestyle="--")
            ax.axhline(-1.0, color="red", linewidth=0.5, linestyle="--")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{fid}_waveform.png"), dpi=120)
        plt.close(fig)
    print(f"\n{len(ids)} pasang audio (asli+augmented) & plot bentuk gelombang -> {out_dir}")
    if ids_worst:
        print(f"  {len(ids) - len(ids_worst)} sampel acak + {len(ids_worst)} kasus clipping "
              f"terburuk (purposive, di luar sampel acak):")
        for fid in ids_worst:
            frac = float(rep.loc[rep.source_file_id == fid, "clipped_frac"].iloc[0])
            print(f"    {frac*100:6.3f}%  {fid}")
    print("Periksa manual: dengarkan pasangan .wav, lihat plot -- pastikan artefak wajar")
    print("(bukan derau/dengung berlebihan yang menghapus karakter kelas).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_validasi_terpisah.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n", type=int, default=200, help="jumlah file untuk statistik distribusi")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="jumlah file diekspor audio+plot (Tahap 5 minta minimal 100)")
    ap.add_argument("--include-worst", type=int, nargs="?", const=10, default=0,
                    metavar="N",
                    help="ikut ekspor N file dgn clipping terbesar di luar sampel acak "
                         "(tanpa angka = 10); sampel jadi acak + purposive")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.n_samples > args.n:
        raise ValueError(f"--n-samples {args.n_samples} melebihi --n {args.n}; "
                         "ekspor hanya bisa dari file yang diaudit")

    cfg = yaml.safe_load(open(args.config))
    manifest = args.manifest or cfg["data"]["train_manifest"]
    results_dir = cfg["paths"]["results_dir"]
    out_dir = args.out_dir or os.path.join(results_dir, "augment_samples")

    df = pd.read_csv(manifest).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if len(df) < args.n:
        raise ValueError(f"manifest hanya {len(df)} baris, kurang dari --n {args.n}")

    import torch
    g = torch.Generator().manual_seed(args.seed)
    aug = TransformAwareAugment(cfg=cfg["augment"], generator=g)

    rep, waves = audit(df, aug, args.n)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "audit_augment.csv")
    rep.to_csv(out_csv, index=False)
    print(f"log distribusi per file -> {out_csv}")

    print_summary(rep, cfg["augment"])
    export_samples(df, waves, out_dir, args.n_samples,
                   rep=rep, n_worst=args.include_worst)


if __name__ == "__main__":
    main()
