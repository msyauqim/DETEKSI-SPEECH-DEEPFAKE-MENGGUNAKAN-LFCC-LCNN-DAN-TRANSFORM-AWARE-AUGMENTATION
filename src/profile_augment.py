# src/profile_augment.py
# Mengukur biaya tiap transformasi augmentasi. Bukan bagian alur skripsi.
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from augment import TransformAwareAugment, simulate_rir

SR, N = 16000, 16000 * 4
wav = torch.randn(1, N).clamp(-1, 1)
g = torch.Generator().manual_seed(0)
aug = TransformAwareAugment(SR, generator=g)
REP = 30

print(f"berkas 4 detik @ {SR} Hz, {REP} ulangan\n")
print(f"{'transformasi':<16}{'ms/berkas':>12}{'berkas/detik':>15}")
print("-" * 43)
rows = []
for name, fn in [("gain", aug._gain), ("noise", aug._noise), ("resample", aug._resample),
                 ("lowpass", aug._lowpass), ("reverb", aug._reverb)]:
    fn(wav)                                   # pemanasan
    t0 = time.time()
    for _ in range(REP):
        fn(wav)
    ms = (time.time() - t0) / REP * 1000
    rows.append((name, ms))
    print(f"{name:<16}{ms:>12.2f}{1000/ms:>15.1f}")

print("\nbiaya baca berkas + LFCC sebagai pembanding:")
from audio_io import load_audio_4s
import pandas as pd
df = pd.read_csv("manifests/train_protocol.csv").head(REP)
t0 = time.time()
for p in df.file_path:
    load_audio_4s(p)
ms_io = (time.time() - t0) / REP * 1000
print(f"{'baca audio':<16}{ms_io:>12.2f}{1000/ms_io:>15.1f}")

total = sum(m for _, m in rows)
print(f"\ntotal augmentasi bila semua aktif: {total:.1f} ms  "
      f"({total/max(ms_io,1e-9):.1f}x biaya baca audio)")
worst = max(rows, key=lambda r: r[1])
print(f"penyumbang terbesar: {worst[0]} = {worst[1]:.1f} ms "
      f"({worst[1]/total*100:.0f}% dari total)")
