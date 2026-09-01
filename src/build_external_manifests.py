# src/build_external_manifests.py
# Menyusun manifest himpunan uji eksternal untuk skenario S4 (Subbab 3.3.3):
#   - In-the-Wild : meta.csv menyediakan label bona-fide / spoof
#   - WaveFake    : hanya memuat ujaran spoof; ujaran bona fide diambil dari
#                   korpus asalnya (LJSpeech) yang harus diunduh terpisah
#
# Manifest DIBEKUKAN: setelah dibuat, berkas ditandai hanya-baca dan sidik jari
# SHA-256-nya dicatat pada manifests/external_freeze.json. Skrip pelatihan dan
# pemilihan model tidak boleh menyentuh berkas ini.
#
#   python src/build_external_manifests.py --itw --wavefake
import os, sys, json, glob, hashlib, argparse
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

RAW = "data/raw"
MAN = "manifests"
FREEZE = os.path.join(MAN, "external_freeze.json")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def freeze(path, catatan):
    """Tandai manifest hanya-baca dan catat sidik jarinya."""
    digest = sha256_file(path)
    os.chmod(path, 0o444)
    rec = {}
    if os.path.exists(FREEZE):
        rec = json.load(open(FREEZE))
    rec[os.path.basename(path)] = {
        "sha256": digest,
        "baris": int(sum(1 for _ in open(path)) - 1),
        "catatan": catatan,
    }
    old = os.path.exists(FREEZE) and not os.access(FREEZE, os.W_OK)
    if old:
        os.chmod(FREEZE, 0o644)
    json.dump(rec, open(FREEZE, "w"), indent=1, ensure_ascii=False)
    os.chmod(FREEZE, 0o444)
    print(f"  dibekukan: sha256 {digest[:16]}...  (hanya-baca)")


# ---------------------------------------------------------------- In-the-Wild
def build_itw():
    root = os.path.join(RAW, "in_the_wild", "release_in_the_wild")
    meta = os.path.join(root, "meta.csv")
    if not os.path.exists(meta):
        print("  ! meta.csv In-the-Wild tidak ditemukan"); return None
    m = pd.read_csv(meta)
    rows = []
    hilang = 0
    for r in m.itertuples():
        p = os.path.join(root, r.file)
        if not os.path.exists(p):
            hilang += 1; continue
        rows.append({
            "source_file_id": f"ITW_{os.path.splitext(r.file)[0]}",
            "file_path": p,
            "speaker_id": str(r.speaker).replace(" ", "_"),
            "label": 0 if str(r.label).lower().startswith("bona") else 1,
            "attack_id": "-" if str(r.label).lower().startswith("bona") else "ITW_spoof",
            "dataset": "In-the-Wild",
            "split": "eval_eksternal",
        })
    df = pd.DataFrame(rows)
    out = os.path.join(MAN, "in_the_wild.csv")
    if os.path.exists(out):
        os.chmod(out, 0o644)
    df.to_csv(out, index=False)
    print(f"  In-the-Wild: {len(df):,} berkas "
          f"(bona fide {int((df.label==0).sum()):,} | spoof {int((df.label==1).sum()):,}) "
          f"| {df.speaker_id.nunique()} pembicara" + (f" | {hilang} berkas hilang" if hilang else ""))
    freeze(out, "In-the-Wild; label dari meta.csv resmi")
    return out


# ---------------------------------------------------------------- WaveFake
def find_ljspeech():
    for pat in [os.path.join(RAW, "LJSpeech-1.1", "wavs"),
                os.path.join(RAW, "**", "LJSpeech-1.1", "wavs")]:
        hit = glob.glob(pat, recursive=True)
        if hit:
            return hit[0]
    return None


def build_wavefake():
    gen = os.path.join(RAW, "wavefake", "generated_audio")
    if not os.path.isdir(gen):
        print("  ! folder WaveFake tidak ditemukan"); return None
    lj = find_ljspeech()
    rows = []

    # --- kelas spoof: seluruh subset WaveFake ---
    for sub in sorted(os.listdir(gen)):
        d = os.path.join(gen, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".wav"):
                continue
            rows.append({
                "source_file_id": f"WF_{sub}_{os.path.splitext(f)[0]}",
                "file_path": os.path.join(d, f),
                "speaker_id": "LJ" if sub.startswith("ljspeech") else (
                    "JSUT" if sub.startswith("jsut") else "CV"),
                "label": 1, "attack_id": sub,
                "dataset": "WaveFake", "split": "eval_eksternal",
            })
    n_spoof = len(rows)

    # --- kelas bona fide: LJSpeech asli ---
    if lj:
        for f in sorted(os.listdir(lj)):
            if f.lower().endswith(".wav"):
                rows.append({
                    "source_file_id": f"LJ_{os.path.splitext(f)[0]}",
                    "file_path": os.path.join(lj, f),
                    "speaker_id": "LJ", "label": 0, "attack_id": "-",
                    "dataset": "LJSpeech", "split": "eval_eksternal",
                })
    df = pd.DataFrame(rows)
    n_bona = int((df.label == 0).sum())
    out = os.path.join(MAN, "wavefake.csv")
    if os.path.exists(out):
        os.chmod(out, 0o644)
    df.to_csv(out, index=False)
    print(f"  WaveFake: {len(df):,} berkas (bona fide {n_bona:,} | spoof {n_spoof:,}) "
          f"| {df[df.label==1].attack_id.nunique()} subset generator")
    if n_bona == 0:
        print("  !! LJSpeech asli belum ada -> manifest ini BELUM bisa dipakai menghitung EER.")
        print("     Unduh LJSpeech-1.1 ke data/raw/ lalu jalankan ulang skrip ini.")
    else:
        freeze(out, "WaveFake (spoof) + LJSpeech-1.1 (bona fide)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itw", action="store_true", help="bangun manifest In-the-Wild")
    ap.add_argument("--wavefake", action="store_true", help="bangun manifest WaveFake")
    args = ap.parse_args()
    if not (args.itw or args.wavefake):
        ap.error("pilih --itw dan/atau --wavefake")
    os.makedirs(MAN, exist_ok=True)
    if args.itw:
        print("== In-the-Wild ==");  build_itw()
    if args.wavefake:
        print("== WaveFake ==");     build_wavefake()
    if os.path.exists(FREEZE):
        print(f"\ncatatan pembekuan -> {FREEZE}")
        for k, v in json.load(open(FREEZE)).items():
            print(f"  {k:20s} {v['baris']:>8,} baris  sha256 {v['sha256'][:16]}...")


if __name__ == "__main__":
    main()
