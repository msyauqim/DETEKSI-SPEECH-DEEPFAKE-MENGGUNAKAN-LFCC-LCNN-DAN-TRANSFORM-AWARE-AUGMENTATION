# src/dedup_sha256.py
# Menutup kriteria selesai Tahap 1 Panduan: "SHA-256 dihitung, duplikasi exact
# dihapus".
#
# Kenapa perlu meski pelatihan sudah selesai: ini VERIFIKASI, bukan pembersihan.
# Pemeriksaan source_file_id hanya membuktikan ID-nya unik, dan ID unik TIDAK
# berarti isinya berbeda. Risiko yang diperiksa adalah audio dengan isi sama
# persis tetapi ID berbeda yang muncul di partisi latih DAN partisi uji -- bila
# itu terjadi, model sebagian "mengenali" berkas yang pernah dilihatnya dan
# angka EER unseen menjadi terlalu bagus. Hanya hash isi yang bisa
# membuktikannya.
#
# Skrip ini TIDAK mengubah manifest apa pun dan TIDAK menyentuh checkpoint.
# Keluarannya laporan; penghapusan baris hanya dilakukan bila --write-clean
# diberikan DAN memang ditemukan duplikat.
#
#   python src/dedup_sha256.py                 # seluruh partisi ASVspoof
#   python src/dedup_sha256.py --limit 500     # uji cepat
import argparse
import hashlib
import os
import sys
import time

import pandas as pd

MANIFESTS = [
    ("train", "manifests/train_protocol.csv"),
    ("dev", "manifests/dev_protocol.csv"),
    ("eval", "manifests/eval_protocol.csv"),
]


def sha256_file(path, blok=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for potong in iter(lambda: f.read(blok), b""):
                h.update(potong)
    except Exception:
        return None
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="hanya N berkas pertama tiap partisi (untuk uji cepat)")
    ap.add_argument("--out", default="results/audit/audit_sha256.csv")
    ap.add_argument("--write-clean", action="store_true",
                    help="tulis manifest tanpa duplikat (HANYA bila ada duplikat)")
    args = ap.parse_args()

    bagian = []
    for nama, p in MANIFESTS:
        if not os.path.exists(p):
            sys.exit(f"{p} tidak ditemukan (jalankan dari root proyek).")
        d = pd.read_csv(p)
        if args.limit:
            d = d.head(args.limit)
        d["partisi"] = nama
        bagian.append(d)
    df = pd.concat(bagian, ignore_index=True)

    print(f"menghitung SHA-256 untuk {len(df):,} berkas ...")
    t0 = time.time()
    hashes, gagal = [], 0
    for i, p in enumerate(df.file_path, 1):
        h = sha256_file(p)
        if h is None:
            gagal += 1
        hashes.append(h)
        if i % 10000 == 0:
            laju = i / (time.time() - t0)
            print(f"  {i:,}/{len(df):,}  ({laju:.0f} berkas/s, "
                  f"sisa ~{(len(df)-i)/laju/60:.1f} menit)")
    df["sha256"] = hashes
    detik = time.time() - t0
    print(f"selesai dalam {detik/60:.1f} menit"
          + (f"  |  {gagal} berkas gagal dibaca" if gagal else ""))

    ada = df[df.sha256.notna()]
    n_unik = ada.sha256.nunique()
    dup_mask = ada.duplicated(subset=["sha256"], keep=False)
    dup = ada[dup_mask]

    print("\n=== HASIL ===")
    print(f"  berkas diperiksa      : {len(ada):,}")
    print(f"  hash unik             : {n_unik:,}")
    print(f"  berkas ber-hash kembar: {len(dup):,}")

    lintas = 0
    if len(dup):
        g = dup.groupby("sha256").partisi.nunique()
        lintas = int((g > 1).sum())
        print(f"  kelompok duplikat     : {dup.sha256.nunique():,}")
        print(f"  kelompok LINTAS partisi: {lintas}   <-- ini yang berbahaya")
        print("\n  10 kelompok duplikat pertama:")
        for h, sub in list(dup.groupby("sha256"))[:10]:
            part = ",".join(sorted(sub.partisi.unique()))
            print(f"    {h[:16]}...  n={len(sub)}  partisi={part}")
            for _, r in sub.head(3).iterrows():
                print(f"      {r.partisi:<6} {r.source_file_id}  label={r.label} attack={r.attack_id}")
    else:
        print("  kelompok duplikat     : 0")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kolom = ["source_file_id", "partisi", "label", "attack_id", "sha256", "file_path"]
    df.reindex(columns=kolom).to_csv(args.out, index=False)
    print(f"\nhash per berkas -> {args.out}")

    print("\n=== KESIMPULAN UNTUK NASKAH ===")
    if len(dup) == 0:
        print("  Tidak ada duplikasi isi. Kriteria Tahap 1 'SHA-256 dihitung,")
        print("  duplikasi exact dihapus' TERPENUHI: tidak ada yang perlu dihapus.")
        print("  Hasil eksperimen bebas dari kebocoran berbasis duplikasi isi.")
    elif lintas == 0:
        print(f"  Ada {dup.sha256.nunique()} kelompok duplikat, tetapi SEMUANYA di dalam")
        print("  partisi yang sama. Tidak ada kebocoran antar-partisi, jadi hasil")
        print("  eksperimen tidak terpengaruh.")
    else:
        print(f"  PERINGATAN: {lintas} kelompok duplikat MELINTASI partisi.")
        print("  Ini kebocoran nyata -- angka EER berpotensi terlalu bagus.")
        print("  Periksa daftar di atas sebelum memakai hasil di naskah.")

    if args.write_clean and len(dup):
        for nama, p in MANIFESTS:
            sub = df[df.partisi == nama].drop_duplicates(subset=["sha256"], keep="first")
            keluar = p.replace(".csv", "_dedup.csv")
            sub.drop(columns=["partisi", "sha256"]).to_csv(keluar, index=False)
            print(f"  manifest bersih -> {keluar} ({len(sub):,} baris)")


if __name__ == "__main__":
    main()
