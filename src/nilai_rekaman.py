# src/nilai_rekaman.py
# Menilai rekaman audio sembarang (misalnya suara sendiri dari mic) dengan
# checkpoint terlatih: asli (bona fide) atau deepfake (spoof)?
#
# Jalur pemrosesan sengaja SAMA PERSIS dengan evaluate.py:
#   mono -> 16 kHz -> potongan 4 detik -> normalisasi puncak 1,0
#   -> LFCC + CMVN (build_lfcc) -> LCNN -> fake_score = softmax kelas spoof
#   ambang = val_threshold checkpoint, dipetakan lewat sigmoid (sama dgn report())
#
# Rekaman lebih panjang dari 4 detik dipotong menjadi beberapa jendela 4 detik
# (bergeser 2 detik) dan tiap jendela dinilai; keputusan akhir dari rerata.
#
# Berkas non-WAV (m4a dari Voice Memos, mov dari QuickTime, mp3, ...) diubah
# otomatis memakai afconvert bawaan macOS -- tidak perlu ffmpeg.
#
# PERINGATAN TAFSIR: rekaman mic di ruangan biasa termasuk kondisi dunia nyata
# seperti In-the-Wild. Pada In-the-Wild model ini memberi rerata fake_score
# 0,97 kepada audio ASLI (model clean-only v2, AUC 0,18). Jadi suara asli dari mic sangat mungkin
# dinilai "deepfake" -- itu keterbatasan generalisasi lintas domain yang sudah
# terukur, bukan berarti suaramu palsu.
#
#   python src/nilai_rekaman.py ~/Desktop/suaraku.m4a
#   python src/nilai_rekaman.py rekaman1.wav rekaman2.m4a --semua
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import soundfile as sf
import torch
import torchaudio

from device import DEVICE
from model_lcnn import LCNN
from train import build_lfcc

VERSI = os.environ.get("VERSI_MODEL", "v2_validasi_terpisah")   # model v2 (validasi terpisah)
UTAMA = [f"checkpoints/{VERSI}/{s}_seed{d}.pt" for s in ("baseline", "augmented")
         for d in (2026, 2027, 2028)]


def ke_wav(path, tmpdir):
    """Kembalikan path WAV yang bisa dibaca soundfile; ubah via afconvert bila perlu."""
    try:
        sf.info(path)
        return path
    except Exception:
        out = os.path.join(tmpdir, os.path.splitext(os.path.basename(path))[0] + ".wav")
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", path, out],
                       check=True, capture_output=True)
        return out


def jendela(path, sr=16000, seconds=4, hop=2):
    data, asal_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).t().mean(0, keepdim=True)        # mono [1, T]
    if asal_sr != sr:
        wav = torchaudio.functional.resample(wav, asal_sr, sr)
    durasi = wav.shape[-1] / sr
    L, H = sr * seconds, sr * hop
    if wav.shape[-1] <= L:
        starts = [0]
    elif wav.shape[-1] < L + H:
        starts = [(wav.shape[-1] - L) // 2]       # satu jendela: ambil tengah, sama dgn evaluate.py
    else:
        starts = list(range(0, wav.shape[-1] - L + 1, H))
    potong = []
    for s in starts:
        w = wav[:, s:s + L]
        if w.shape[-1] < L:
            w = torch.nn.functional.pad(w, (0, L - w.shape[-1]))
        w = w / w.abs().max().clamp_min(1e-6)                      # sama dgn load_audio_4s
        potong.append((s / sr, w))
    return potong, durasi, asal_sr


def muat(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg = ck["config"]
    comp = ck.get("components", cfg["lfcc"]["components"])
    lfcc = build_lfcc(cfg, comp).to(DEVICE)
    model = LCNN(n_feat=ck.get("n_feat", lfcc.out_dim),
                 embedding_dim=cfg["model"]["embedding_dim"],
                 dropout=cfg["model"].get("dropout", 0.7)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    t = ck.get("val_threshold")
    thr = 0.5 if t is None else float(1.0 / (1.0 + np.exp(-float(t))))
    return model, lfcc, thr


def nilai(model, lfcc, potong):
    x = torch.stack([w for _, w in potong]).to(DEVICE)               # [N, 1, T]
    with torch.no_grad():
        logits, _ = model(lfcc(x))
        return logits.softmax(dim=1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("berkas", nargs="+")
    ap.add_argument("--checkpoint", default="checkpoints/v2_validasi_terpisah/augmented_seed2026.pt",
                    help="bawaan: augmented_seed2026 (dipilih dari EER validasi terkecil)")
    ap.add_argument("--semua", action="store_true",
                    help="nilai dengan ke-6 model utama (baseline & augmented x 3 seed)")
    args = ap.parse_args()

    ckpts = UTAMA if args.semua else [args.checkpoint]
    model_siap = {c: muat(c) for c in ckpts}

    with tempfile.TemporaryDirectory() as tmp:
        for p in args.berkas:
            p = os.path.expanduser(p)
            wavp = ke_wav(p, tmp)
            potong, durasi, asal_sr = jendela(wavp)
            catatan_sr = (f"{asal_sr} Hz" if wavp == p
                          else "diubah ke 16000 Hz oleh afconvert")
            print(f"\n{os.path.basename(p)}  |  {durasi:.2f} detik, {catatan_sr}"
                  f"  |  {len(potong)} jendela 4 detik")
            if durasi < 1.0:
                print("  ! rekaman < 1 detik, terlalu pendek untuk dinilai bermakna")

            for c, (model, lfcc, thr) in model_siap.items():
                fs = nilai(model, lfcc, potong)
                rerata = float(fs.mean())
                vonis = "DEEPFAKE" if rerata >= thr else "ASLI"
                nama = os.path.basename(c).replace(".pt", "")
                print(f"  {nama:<20} fake_score rerata {rerata:.3f}  "
                      f"(ambang {thr:.3f})  ->  {vonis}")
                if len(potong) > 1 and not args.semua:
                    for (t0, _), f in zip(potong, fs):
                        print(f"      detik {t0:5.1f}-{t0+4:5.1f}  fake_score {f:.3f}")

    print("\nCatatan tafsir: fake_score mendekati 1 = model yakin palsu, mendekati 0 = asli.")
    print("Rekaman mic termasuk kondisi dunia nyata (seperti In-the-Wild), tempat model ini")
    print("sudah terukur kolaps: audio asli In-the-Wild rerata fake_score 0,97. Vonis")
    print("DEEPFAKE pada suaramu sendiri adalah keterbatasan lintas domain, bukan bukti palsu.")


if __name__ == "__main__":
    main()
