# src/grafik_skripsi_v2.py
# Grafik Bab 6 naskah skripsi dari hasil eksperimen v2 (validasi terpisah: latih A01, A02, A04, A06;
# validasi A03, A05). Membaca results/v2_validasi_terpisah dan menulis ke results/figures/v2/.
# Tidak ada pelatihan/evaluasi ulang: semua angka dibaca dari results/.
#
#   python src/grafik_skripsi_v2.py         -> results/figures/v2/*.png
#
# Warna: biru = clean-only, oranye = augmented (lolos validasi CVD, sama dengan web demo).
# Seed dibedakan dengan bentuk penanda agar tetap terbaca bila dicetak hitam-putih.
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_curve

R = "results"                       # folder bersama: figures, audit, augment_samples
V1 = "results/v2_validasi_terpisah"   # hasil eksperimen v2 (nama variabel dipertahankan)
OUT = f"{R}/figures/v2"
MIN_EPOCH = 5
BIRU, ORANYE, ABU, TINTA, TINTA2 = "#0072B2", "#D55E00", "#B8B8B4", "#1F1F1F", "#5C5C58"
WARNA = {"baseline": BIRU, "augmented": ORANYE}
NAMA = {"baseline": "Clean-only", "augmented": "Augmented"}
SEEDS = (2026, 2027, 2028)
PENANDA = {2026: "o", 2027: "s", 2028: "^"}
GARIS = {2026: "-", 2027: "--", 2028: ":"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": "#8A8A86", "axes.linewidth": 0.8, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E3",
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 200,
})

e = pd.read_csv(f"{V1}/eval_summary.csv")
e["skema"] = e.checkpoint.str.replace(r"_seed\d+\.pt$", "", regex=True)


def simpan(fig, nama):
    fig.savefig(f"{OUT}/{nama}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", f"{OUT}/{nama}")


def per_seed(skema, cond, scen):
    x = e[(e.skema == skema) & (e.condition == cond) & (e.scenario == scen)].sort_values("seed")
    return x.eer.values * 100


# ---------------------------------------------------------------- 1. kurva pelatihan
def baca_log(tag):
    rows = []
    pola = re.compile(r"^epoch\s+(\d+) \| loss ([\d.]+) \| val_EER\s+([\d.]+)%")
    for baris in open(f"{V1}/logs/train_{tag}.log", encoding="utf8"):
        m = pola.match(baris)
        if m:
            rows.append((int(m[1]), float(m[2]), float(m[3])))
    d = pd.DataFrame(rows, columns=["epoch", "loss", "val_eer"])
    # log yang pernah dilanjutkan (resume) dapat memuat epoch yang sama dua kali; ambil yang terakhir
    return d.drop_duplicates("epoch", keep="last").sort_values("epoch")


def kurva_pelatihan():
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for sk in ("baseline", "augmented"):
        for s in SEEDS:
            d = baca_log(f"{sk}_seed{s}")
            for a, kol in zip(ax, ("loss", "val_eer")):
                a.plot(d.epoch, d[kol], color=WARNA[sk], ls=GARIS[s], lw=1.6,
                       label=f"{NAMA[sk]} seed {s}")
    ax[0].set_title("Loss pelatihan per epoch", loc="left", color=TINTA)
    ax[1].set_title("EER validasi per epoch (%)", loc="left", color=TINTA)
    for a in ax:
        a.set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("EER validasi (%)")
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    for a in ax:
        a.axvspan(0.5, MIN_EPOCH - 0.5, color=ABU, alpha=0.22, lw=0, zorder=0)
    ax[1].text(0.7, ax[1].get_ylim()[1] * 0.97, "epoch < 5:\ntidak dipilih", fontsize=7,
               va="top", color=TINTA2)
    for sk in ("baseline", "augmented"):
        for s in SEEDS:
            d = baca_log(f"{sk}_seed{s}")
            b = d.loc[d[d.epoch >= MIN_EPOCH].val_eer.idxmin()]
            ax[1].plot(b.epoch, b.val_eer, "*", ms=9, color=WARNA[sk], mec="white", mew=0.6, zorder=4)
    ax[1].plot([], [], "*", ms=9, color=TINTA2, label="checkpoint terpilih")
    h, l = ax[0].get_legend_handles_labels()
    h2, l2 = ax[1].get_legend_handles_labels()
    h, l = h + h2[-1:], l + l2[-1:]
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13), fontsize=8)
    fig.tight_layout()
    simpan(fig, "kurva_pelatihan.png")


# ---------------------------------------------------------------- 2. EER per skenario
def eer_skenario():
    kond = [("seen", "S1", "S1 seen"), ("unseen", "S2", "S2 unseen"),
            ("wavefake_ljspeech", "S4", "WaveFake"), ("in_the_wild", "S4", "In-the-Wild")]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    w = 0.36
    for j, sk in enumerate(("baseline", "augmented")):
        for i, (c, sc, _) in enumerate(kond):
            v = per_seed(sk, c, sc)
            x = i + (j - 0.5) * w * 1.08
            ax.bar(x, v.mean(), w, color=WARNA[sk], alpha=0.85, zorder=2,
                   label=NAMA[sk] if i == 0 else None)
            for s, y in zip(SEEDS, v):
                ax.plot(x, y, PENANDA[s], ms=4.5, mfc="white", mec=TINTA, mew=0.9, zorder=3)
            ax.text(x, v.max() + 1.6, f"{v.mean():.2f}".replace(".", ","), ha="center",
                    va="bottom", fontsize=8, color=TINTA)
    ax.axhline(50, color=TINTA2, lw=0.8, ls="--", zorder=1)
    ax.text(-0.42, 51, "EER 50% = tebakan acak", ha="left", va="bottom", fontsize=7.5, color=TINTA2)
    ax.set_xticks(range(len(kond)), [k[2] for k in kond])
    ax.set_ylabel("EER (%)")
    ax.set_ylim(0, 96)
    ax.grid(axis="x", visible=False)
    for s in SEEDS:
        ax.plot([], [], PENANDA[s], ms=4.5, mfc="white", mec=TINTA, label=f"seed {s}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=8)
    simpan(fig, "eer_per_skenario.png")


# ---------------------------------------------------------------- 3. stress test
def stress_test():
    jenis = [("noise", "Additive noise", ["bersih", "20 dB", "12,5 dB", "5 dB"]),
             ("resample", "Resampling", ["bersih", "12 kHz", "11 kHz", "8 kHz"]),
             ("lowpass", "Low-pass filter", ["bersih", "7 kHz", "5,2 kHz", "3,4 kHz"]),
             ("reverb", "Reverberation", ["bersih", "0,15 s", "0,4 s", "0,7 s"]),
             ("gain", "Atenuasi gain", ["bersih", "−2 dB", "−4 dB", "−6 dB"])]
    fig, ax = plt.subplots(2, 5, figsize=(10.5, 4.8), sharey="row")
    for r, (pfx, judul_baris) in enumerate((("", "unseen"), ("seen_", "seen"))):
        for c, (kd, judul, lbl) in enumerate(jenis):
            a = ax[r, c]
            for sk in ("baseline", "augmented"):
                v = [per_seed(sk, pfx + "clean", "S3")]
                v += [per_seed(sk, f"{pfx}{kd}_L{lv}", "S3") for lv in (1, 2, 3)]
                m = np.array([x.mean() for x in v])
                lo = np.array([x.min() for x in v]); hi = np.array([x.max() for x in v])
                a.fill_between(range(4), lo, hi, color=WARNA[sk], alpha=0.15, lw=0)
                a.plot(range(4), m, color=WARNA[sk], lw=2, marker="o", ms=4,
                       mfc="white", mew=1.4, label=NAMA[sk])
            a.set_xticks(range(4), lbl, fontsize=7)
            a.set_title(f"{judul} ({judul_baris})", fontsize=8.5, loc="left", color=TINTA)
            a.set_ylim(bottom=0)
        ax[r, 0].set_ylabel("EER (%)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    simpan(fig, "stress_test.png")


# ---------------------------------------------------------------- 4. ablation augmentasi
def ablation_aug():
    konf = [("augmented", "Augmentasi penuh"), ("abl_noaug_noise", "Tanpa additive noise"),
            ("abl_noaug_reverb", "Tanpa reverberation"), ("abl_noaug_gain", "Tanpa gain"),
            ("abl_noaug_lowpass", "Tanpa low-pass filter"),
            ("abl_noaug_resample", "Tanpa resampling"), ("baseline", "Clean-only (pembanding)")]
    full = per_seed("augmented", "unseen", "S2")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axvspan(full.min(), full.max(), color=ORANYE, alpha=0.10, lw=0, zorder=0)
    for i, (sk, nm) in enumerate(konf):
        v = per_seed(sk, "unseen", "S2")
        y = len(konf) - 1 - i
        col = BIRU if sk == "baseline" else ORANYE
        ax.plot([v.min(), v.max()], [y, y], color=col, lw=1.4, alpha=0.6, zorder=1)
        for s, x in zip(SEEDS, v):
            ax.plot(x, y, PENANDA[s], ms=6, color=col, mec="white", mew=1, zorder=3)
        ax.plot(v.mean(), y, "|", ms=13, mew=2, color=TINTA, zorder=4)
        ax.text(v.max() + 0.35, y, f"{v.mean():.2f}".replace(".", ","), va="center",
                fontsize=8, color=TINTA)
    ax.set_yticks(range(len(konf)), [n for _, n in konf][::-1])
    ax.set_xlabel("EER pada serangan unseen (%)")
    ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    for s in SEEDS:
        ax.plot([], [], PENANDA[s], ms=6, color=TINTA2, label=f"seed {s}")
    ax.plot([], [], "|", ms=11, mew=2, color=TINTA, label="rerata")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8)
    simpan(fig, "ablation_augmentasi.png")


# ---------------------------------------------------------------- 5. ROC serangan unseen
def roc_unseen():
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for j, (cond, judul) in enumerate((("unseen", "S2 serangan unseen"), ("in_the_wild", "In-the-Wild"))):
        a = ax[j]
        for sk in ("baseline", "augmented"):
            for s in SEEDS:
                d = pd.read_csv(f"{V1}/scores_{cond}_{sk}_seed{s}.csv")
                fpr, tpr, _ = roc_curve(d.label, d.fake_score)
                a.plot(fpr * 100, tpr * 100, color=WARNA[sk], ls=GARIS[s], lw=1.5,
                       label=f"{NAMA[sk]} seed {s}")
        a.plot([0, 100], [0, 100], color=ABU, lw=0.8, ls="--")
        a.set_title(f"Kurva ROC, {judul}", loc="left", color=TINTA)
        a.set_xlabel("audio asli dinilai palsu (%)")
        a.set_ylabel("audio palsu terdeteksi (%)")
        a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_aspect("equal")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.tight_layout()
    simpan(fig, "roc_unseen_itw.png")


# ---------------------------------------------------------------- 6. spektrogram augmentasi (lampiran)
def spektrogram_augmentasi(n=3):
    a = pd.read_csv(f"{V1}/audit_augment.csv")
    ada = sorted(f[:-9] for f in os.listdir(f"{V1}/augment_samples") if f.endswith("_asli.wav"))
    a = a[a.source_file_id.isin(ada)]
    # pilih contoh yang memuat transformasi berbeda-beda, utamakan additive noise dan reverberation
    pilih, dipakai = [], set()
    for kunci in ("noise", "reverb", "lowpass", "resample"):
        cand = a[a.applied.fillna("").str.contains(kunci) & ~a.source_file_id.isin(dipakai)]
        if len(cand) and len(pilih) < n:
            r = cand.sort_values("n_transforms").iloc[0]
            pilih.append(r); dipakai.add(r.source_file_id)
    nama_t = {"gain": "gain", "noise": "additive noise", "resample": "resampling",
              "lowpass": "low-pass filter", "reverb": "reverberation"}
    fig, ax = plt.subplots(len(pilih), 2, figsize=(8.2, 2.2 * len(pilih)), sharex=True, sharey=True)
    for i, r in enumerate(pilih):
        for j, akhir in enumerate(("asli", "augmented")):
            x, sr = sf.read(f"{V1}/augment_samples/{r.source_file_id}_{akhir}.wav", dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            g = ax[i, j]
            g.specgram(x + 1e-9 * np.random.default_rng(0).standard_normal(len(x)), NFFT=512, Fs=sr, noverlap=352, cmap="magma", vmin=-130, vmax=-20)
            g.grid(False)
            lbl = "bona fide" if r.label == 0 else f"spoof {r.attack_id}"
            if j == 0:
                g.set_title(f"{r.source_file_id} ({lbl}), asli", loc="left", fontsize=8.5, color=TINTA)
                g.set_ylabel("frekuensi (Hz)")
            else:
                tr = ", ".join(nama_t.get(t, t) for t in str(r.applied).split(";"))
                g.set_title(f"setelah augmentasi: {tr}", loc="left", fontsize=8.5, color=TINTA)
    for g in ax[-1]:
        g.set_xlabel("waktu (detik)")
    fig.tight_layout()
    simpan(fig, "spektrogram_augmentasi.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    kurva_pelatihan()
    eer_skenario()
    stress_test()
    ablation_aug()
    roc_unseen()
    spektrogram_augmentasi()
