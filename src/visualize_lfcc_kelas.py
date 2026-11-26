# src/visualize_lfcc_kelas.py
# Visualisasi LFCC yang benar-benar menunjukkan perbedaan KELAS.
#
# MASALAH pada results/figures/lfcc_bonafide_vs_fake.png (dibuat visualize_lfcc.py):
# gambar itu membandingkan SATU berkas bona fide (3,46 s) dengan SATU berkas fake
# (1,92 s). Perbedaan paling mencolok di sana -- blok datar yang muncul di frame
# ~345 lawan ~190 -- semata akibat zero-padding karena durasi keduanya berbeda,
# BUKAN artefak sintesis. Durasi bukan pembeda kelas: pada partisi latih, rerata
# bona fide 3,39 s dan spoof 3,43 s (median 3,24 s lawan 3,20 s). Menafsirkan
# gambar itu sebagai "beda kelas terlihat jelas" tidak dapat dipertahankan.
#
# PERBAIKANNYA di skrip ini:
#   1. hanya memakai berkas berdurasi >= 4 detik, sehingga TIDAK ADA tambalan nol
#      sama sekali dan setiap frame berisi sinyal nyata
#   2. merata-ratakan BANYAK berkas per kelas, bukan mengandalkan satu berkas
#   3. menampilkan peta SELISIH antar-kelas memakai skala diverging bertitik
#      tengah netral, sehingga arah perbedaan terbaca
#   4. menampilkan profil per koefisien dengan pita simpangan baku, sehingga
#      terlihat koefisien mana yang benar-benar memisahkan kelas dan mana yang
#      sebarannya bertumpang tindih
#
#   python src/visualize_lfcc_kelas.py
#   python src/visualize_lfcc_kelas.py --n 500
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from audio_io import load_audio_4s
from features_lfcc import LFCCFeature

# Pasangan kategorikal aman-buta-warna (Okabe-Ito). Divalidasi:
# CVD dE 21,9 (protan) / 30,9 (tritan), normal 31,2 -- seluruh cek lolos.
WARNA = {"bona fide": "#0072B2", "spoof": "#D55E00"}
SEQ = "magma"        # sequential: satu hue, terang -> gelap (magnitudo)
DIV = "RdBu_r"       # diverging: dua kutub + titik tengah netral (polaritas)


def muat_berkas(cfg, df, n, seed=0):
    """Rata-rata LFCC atas n berkas. Hanya berkas >= 4 detik supaya tidak ada
    satu pun frame tambalan yang mencemari rerata."""
    lf = cfg["lfcc"]
    stats = json.load(open(lf["norm_stats"]))
    fitur = LFCCFeature(sample_rate=cfg["audio"]["target_sr"], n_filter=lf["n_filter"],
                        n_lfcc=lf["n_lfcc"], n_fft=lf["n_fft"],
                        win_length=lf["win_length"], hop_length=lf["hop_length"],
                        components="delta2", norm_stats=stats)
    sub = df.sample(min(n, len(df)), random_state=seed)
    tumpuk = []
    for p in sub.file_path:
        try:
            wav = load_audio_4s(p)
        except Exception:
            continue
        with torch.no_grad():
            out = fitur(wav.unsqueeze(0), normalize=True)
        tumpuk.append(out.squeeze().numpy())      # [180, frame]
    return np.stack(tumpuk), len(tumpuk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_validasi_terpisah.yaml")
    ap.add_argument("--n", type=int, default=300, help="berkas per kelas")
    ap.add_argument("--out", default="results/figures/lfcc_kelas_rerata.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    man = pd.read_csv(cfg["data"]["train_manifest"])
    audit = pd.read_csv("results/audit/audit_train_protocol.csv")[["source_file_id", "duration_s"]]
    man = man.merge(audit, on="source_file_id")

    panjang = man[man.duration_s >= 4.0]
    print(f"berkas latih >= 4 detik (tanpa tambalan): {len(panjang):,} dari {len(man):,}")

    data, n_riil = {}, {}
    for lab, nama in [(0, "bona fide"), (1, "spoof")]:
        arr, n = muat_berkas(cfg, panjang[panjang.label == lab], args.n)
        data[nama], n_riil[nama] = arr, n
        print(f"  {nama:<10} {n} berkas diproses, bentuk {arr.shape}")

    b, s = data["bona fide"], data["spoof"]
    mb, ms = b.mean(0), s.mean(0)                  # [180, frame]
    selisih = ms - mb

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
    fig.suptitle("LFCC rerata per kelas -- hanya berkas >= 4 detik, tanpa zero-padding",
                 fontsize=13, fontweight="bold", y=0.975)

    # ---------- baris 1: peta rerata + selisih ----------
    vmin = min(mb.min(), ms.min())
    vmax = max(mb.max(), ms.max())
    for j, (arr, nama) in enumerate([(mb, "bona fide"), (ms, "spoof")]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap=SEQ, vmin=vmin, vmax=vmax)
        ax.set_title(f"Rerata {nama}  (n={n_riil[nama]})", fontsize=10)
        ax.set_xlabel("frame (10 ms)", fontsize=9)
        if j == 0:
            ax.set_ylabel("koefisien (0-59 statis, 60-119 delta,\n120-179 delta-delta)",
                          fontsize=9)
        for y in (60, 120):
            ax.axhline(y, color="white", lw=0.8, alpha=0.55)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[0, 2])
    lim = float(np.abs(selisih).max())
    im = ax.imshow(selisih, aspect="auto", origin="lower", cmap=DIV, vmin=-lim, vmax=lim)
    ax.set_title("Selisih (spoof - bona fide)", fontsize=10)
    ax.set_xlabel("frame (10 ms)", fontsize=9)
    for y in (60, 120):
        ax.axhline(y, color="#333333", lw=0.8, alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # ---------- baris 2: profil per koefisien, tiap komponen ----------
    nama_komp = ["Statis (0-59)", "Delta (60-119)", "Delta-delta (120-179)"]
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        sl = slice(k * 60, (k + 1) * 60)
        x = np.arange(60)
        for arr, nama in [(b, "bona fide"), (s, "spoof")]:
            per_berkas = arr[:, sl, :].mean(2)       # rerata atas waktu -> [berkas, 60]
            m, sd = per_berkas.mean(0), per_berkas.std(0, ddof=1)
            ax.plot(x, m, color=WARNA[nama], lw=2, label=nama, zorder=3)
            ax.fill_between(x, m - sd, m + sd, color=WARNA[nama], alpha=0.16, lw=0, zorder=2)
        ax.axhline(0, color="#999999", lw=0.8, zorder=1)
        ax.set_title(nama_komp[k], fontsize=10)
        ax.set_xlabel("nomor koefisien", fontsize=9)
        ax.grid(alpha=0.18, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k == 0:
            ax.set_ylabel("nilai rerata (setelah CMVN)", fontsize=9)
            ax.legend(frameon=False, fontsize=9, loc="best")
        # Label langsung supaya identitas tidak bergantung warna saja. Pada
        # komponen delta dan delta-delta kedua garis nyaris berimpit, jadi
        # labelnya direnggangkan dulu agar tidak saling menimpa.
        ujung = {nama: arr[:, sl, :].mean(2).mean(0)[-1]
                 for arr, nama in [(b, "bona fide"), (s, "spoof")]}
        lo, hi = ax.get_ylim()
        minimal = (hi - lo) * 0.07               # jarak tegak minimum antar-label
        atas, bawah = max(ujung, key=ujung.get), min(ujung, key=ujung.get)
        ya, yb = ujung[atas], ujung[bawah]
        if ya - yb < minimal:
            tengah = (ya + yb) / 2
            ya, yb = tengah + minimal / 2, tengah - minimal / 2
        for nama, y in ((atas, ya), (bawah, yb)):
            ax.annotate(nama, xy=(59, y), xytext=(4, 0), textcoords="offset points",
                        color=WARNA[nama], fontsize=8, va="center",
                        annotation_clip=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\ngambar -> {args.out}")

    # ---------- ringkasan angka, supaya klaim tidak bersandar pada mata ----------
    print("\nkoefisien dengan selisih rerata terbesar (|spoof - bona fide|):")
    per_koef = np.abs(selisih).mean(1)
    for i in np.argsort(per_koef)[::-1][:8]:
        komp = "statis" if i < 60 else ("delta" if i < 120 else "delta-delta")
        pb = b[:, i, :].mean(1)
        ps = s[:, i, :].mean(1)
        tumpang = not (pb.max() < ps.min() or pb.min() > ps.max())
        print(f"  koef {i:>3} ({komp:<11}) selisih {per_koef[i]:.3f}  "
              f"sebaran antar-berkas: {'TUMPANG TINDIH' if tumpang else 'TERPISAH'}")


if __name__ == "__main__":
    main()
