# src/build_wavefake_ljspeech.py
# Manifest WaveFake yang SEPADAN dengan kelas bona fide-nya.
#
# Masalah pada manifests/wavefake.csv: ia mencampur tiga sumber spoof yang
# tidak sebanding dengan bona fide (LJSpeech-1.1, Inggris, pembicara tunggal):
#   - jsut_*  (2 subset, 10.000 berkas)  bahasa JEPANG, 24 kHz
#   - common_voices_prompts_... (16.283) teks CommonVoice + pipeline TTS penuh
# Model bisa "menang" hanya dengan mengenali bahasa atau laju cuplik, bukan
# artefak sintesis -- jadi EER-nya tidak mengukur apa yang kita maksud.
#
# Manifest ini menyisakan perbandingan yang benar-benar terkendali:
# bona fide LJSpeech vs 7 vokoder yang meresintesis LJSpeech yang SAMA
# (pembicara sama, teks sama, korpus sama) -- satu-satunya yang berbeda
# adalah vokodernya.
#
#   python src/build_wavefake_ljspeech.py
import hashlib
import json
import os
import stat

import pandas as pd

SUB_LJ = [
    "ljspeech_full_band_melgan",
    "ljspeech_hifiGAN",
    "ljspeech_melgan",
    "ljspeech_melgan_large",
    "ljspeech_multi_band_melgan",
    "ljspeech_parallel_wavegan",
    "ljspeech_waveglow",
]
SUMBER = "manifests/wavefake.csv"
KELUARAN = "manifests/wavefake_ljspeech.csv"
FREEZE = "manifests/external_freeze.json"


def main():
    d = pd.read_csv(SUMBER)
    bona = d[d.label == 0]
    spoof = d[(d.label == 1) & (d.attack_id.isin(SUB_LJ))]
    dibuang = d[(d.label == 1) & (~d.attack_id.isin(SUB_LJ))]

    print(f"sumber {SUMBER}: {len(d):,} baris")
    print(f"\n  dipertahankan:")
    print(f"    bona fide (LJSpeech-1.1)          {len(bona):>7,}")
    print(f"    spoof 7 vokoder turunan LJSpeech  {len(spoof):>7,}")
    print(f"\n  dibuang (tidak sepadan dgn bona fide):")
    for a, n in dibuang.attack_id.value_counts().items():
        print(f"    {a:<55} {n:>7,}")
    print(f"    {'TOTAL DIBUANG':<55} {len(dibuang):>7,}")

    out = pd.concat([bona, spoof]).sort_values("source_file_id").reset_index(drop=True)
    if os.path.exists(KELUARAN):            # berkas lama read-only
        os.chmod(KELUARAN, stat.S_IWUSR | stat.S_IRUSR)
    out.to_csv(KELUARAN, index=False)

    h = hashlib.sha256(open(KELUARAN, "rb").read()).hexdigest()
    beku = json.load(open(FREEZE)) if os.path.exists(FREEZE) else {}
    if os.path.exists(FREEZE):          # berkas freeze juga dibekukan read-only
        os.chmod(FREEZE, stat.S_IWUSR | stat.S_IRUSR)
    beku[os.path.basename(KELUARAN)] = {
        "sha256": h,
        "baris": len(out),
        "catatan": ("WaveFake terbatas 7 vokoder turunan LJSpeech + LJSpeech-1.1 "
                    "(bona fide); subset jsut_* dan common_voices_prompts_* dibuang "
                    "karena tidak sepadan dengan kelas bona fide"),
    }
    json.dump(beku, open(FREEZE, "w"), indent=1)
    os.chmod(FREEZE, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)     # bekukan kembali
    os.chmod(KELUARAN, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)   # bekukan

    print(f"\n-> {KELUARAN}: {len(out):,} baris "
          f"({(out.label==0).sum():,} bona fide / {(out.label==1).sum():,} spoof)")
    print(f"   SHA-256 {h[:16]}...  dicatat di {FREEZE}, berkas dijadikan read-only")


if __name__ == "__main__":
    main()
