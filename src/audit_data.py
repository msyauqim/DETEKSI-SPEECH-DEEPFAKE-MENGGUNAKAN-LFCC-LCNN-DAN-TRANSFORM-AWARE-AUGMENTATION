# src/audit_data.py
# Audit integritas data sesuai Subbab 3.3.1 proposal.
#
# Memeriksa untuk setiap berkas pada sebuah manifest:
#   1. keterbacaan berkas
#   2. kesesuaian laju cuplik
#   3. jumlah kanal
#   4. durasi minimum
#   5. berkas rusak atau kosong (senyap total / mengandung NaN)
#
# Berkas yang tidak lolos dicatat pada results/audit_<nama>.csv dan dapat
# dikeluarkan dari manifest memakai --write-clean.
#
#   python src/audit_data.py --manifest manifests/train_protocol.csv
#   python src/audit_data.py --all
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import soundfile as sf
import yaml


def audit_file(path, target_sr, min_sec):
    """Kembalikan (status, keterangan, sr, kanal, durasi)."""
    try:
        info = sf.info(str(path))
    except Exception as e:
        return "gagal_baca", str(e)[:60], None, None, None
    dur = info.frames / info.samplerate if info.samplerate else 0.0
    if info.frames == 0:
        return "kosong", "0 cuplikan", info.samplerate, info.channels, dur
    if dur < min_sec:
        return "terlalu_pendek", f"{dur:.3f} s < {min_sec} s", info.samplerate, info.channels, dur
    notes = []
    if info.samplerate != target_sr:
        notes.append(f"laju {info.samplerate} != {target_sr}")
    if info.channels != 1:
        notes.append(f"{info.channels} kanal")
    try:
        data, _ = sf.read(str(path), dtype="float32")
    except Exception as e:
        return "gagal_baca", str(e)[:60], info.samplerate, info.channels, dur
    if not np.isfinite(data).all():
        return "rusak", "mengandung NaN/Inf", info.samplerate, info.channels, dur
    if float(np.abs(data).max()) == 0.0:
        return "senyap", "amplitudo nol", info.samplerate, info.channels, dur
    status = "lolos_dengan_catatan" if notes else "lolos"
    return status, "; ".join(notes), info.samplerate, info.channels, dur


def audit_manifest(path, target_sr, min_sec, limit=None, quiet=False):
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        st, note, sr, ch, dur = audit_file(r.file_path, target_sr, min_sec)
        rows.append({"source_file_id": getattr(r, "source_file_id", ""),
                     "file_path": r.file_path, "status": st, "keterangan": note,
                     "sample_rate": sr, "channels": ch, "duration_s": dur})
        if not quiet and i % 5000 == 0:
            print(f"    {i:,}/{len(df):,} diperiksa")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--all", action="store_true", help="audit train, dev, dan eval")
    ap.add_argument("--min-sec", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="periksa n berkas pertama saja")
    ap.add_argument("--write-clean", action="store_true",
                    help="tulis ulang manifest tanpa berkas yang gagal")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    target_sr = cfg["audio"]["target_sr"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    targets = ([cfg["data"]["train_manifest"], cfg["data"]["dev_manifest"],
                cfg["data"]["eval_manifest"]] if args.all else [args.manifest])
    if not targets or targets[0] is None:
        ap.error("berikan --manifest atau --all")

    ringkasan = []
    for mpath in targets:
        name = os.path.splitext(os.path.basename(mpath))[0]
        print(f"\n== AUDIT {name} ==")
        rep = audit_manifest(mpath, target_sr, args.min_sec, args.limit)
        counts = rep.status.value_counts().to_dict()
        gagal = rep[~rep.status.isin(["lolos", "lolos_dengan_catatan"])]

        out = os.path.join(results_dir, f"audit_{name}.csv")
        rep.to_csv(out, index=False)
        print(f"  diperiksa {len(rep):,} berkas -> {out}")
        for k, v in sorted(counts.items()):
            print(f"    {k:22s}: {v:,}")
        if len(rep):
            print(f"    durasi  min {rep.duration_s.min():.2f} s | "
                  f"rerata {rep.duration_s.mean():.2f} s | maks {rep.duration_s.max():.2f} s")
            print(f"    laju cuplik unik: {sorted(rep.sample_rate.dropna().unique().tolist())}")
            print(f"    jumlah kanal unik: {sorted(rep.channels.dropna().unique().tolist())}")

        if args.write_clean and len(gagal):
            df = pd.read_csv(mpath)
            df = df[~df.file_path.isin(gagal.file_path)]
            df.to_csv(mpath, index=False)
            print(f"  {len(gagal)} berkas dikeluarkan; manifest ditulis ulang")
        ringkasan.append({"manifest": name, "diperiksa": len(rep),
                          "gagal": len(gagal), **counts})

    s = pd.DataFrame(ringkasan)
    out = os.path.join(results_dir, "audit_ringkasan.csv")
    s.to_csv(out, index=False)
    print(f"\nringkasan -> {out}")
    total_gagal = int(s.gagal.sum())
    print("HASIL: seluruh berkas lolos audit" if total_gagal == 0
          else f"HASIL: {total_gagal} berkas TIDAK lolos, lihat berkas audit di atas")


if __name__ == "__main__":
    main()
