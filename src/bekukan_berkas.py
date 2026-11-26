# src/bekukan_berkas.py
# Membekukan berkas penentu hasil dengan SHA-256 dan menjadikannya read-only,
# lalu mencatatnya di manifests/external_freeze.json.
#
# Kriteria Tahap 2 Panduan meminta daftar serangan dibekukan sebelum pelatihan.
# Prinsip yang sama berlaku untuk setiap berkas yang, bila berubah, membuat
# seluruh angka hasil ikut berubah tanpa jejak:
#
#   manifests/split_attack.csv  daftar seen/unseen -- menentukan klaim generalisasi
#   manifests/norm_stats.json   statistik CMVN -- dipakai train.py DAN evaluate.py,
#                               jadi setiap skor bergantung padanya
#
# Menjalankan ulang skrip ini pada berkas yang sudah beku akan MEMBANDINGKAN
# hash-nya dan memperingatkan bila berubah, bukan diam-diam menimpa catatan.
#
#   python src/bekukan_berkas.py --cek            # periksa saja, jangan ubah
#   python src/bekukan_berkas.py
import argparse
import hashlib
import json
import os
import stat

FREEZE = "manifests/external_freeze.json"
TARGET = {
    "manifests/split_attack.csv":
        "split seen/unseen attack; menentukan klaim generalisasi (Tahap 2)",
    "manifests/norm_stats.json":
        "statistik CMVN dari partisi latih; dipakai train.py dan evaluate.py "
        "sehingga setiap skor bergantung padanya (Tahap 3)",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def baris_berkas(path):
    if path.endswith(".csv"):
        with open(path) as f:
            return sum(1 for _ in f) - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cek", action="store_true", help="periksa saja, jangan ubah apa pun")
    args = ap.parse_args()

    beku = json.load(open(FREEZE)) if os.path.exists(FREEZE) else {}
    berubah, baru, tetap = [], [], []

    for path, catatan in TARGET.items():
        if not os.path.exists(path):
            print(f"  ! {path} tidak ada, dilewat")
            continue
        nama = os.path.basename(path)
        h = sha256(path)
        lama = beku.get(nama, {}).get("sha256")
        if lama is None:
            baru.append((nama, h, path, catatan))
        elif lama != h:
            berubah.append((nama, lama, h))
        else:
            tetap.append(nama)

    for n in tetap:
        print(f"  tetap   {n}  (hash cocok dengan catatan)")
    for n, h, _, _ in baru:
        print(f"  BARU    {n}  {h[:16]}...")
    for n, lama, h in berubah:
        print(f"  BERUBAH {n}  {lama[:16]}... -> {h[:16]}...")
        print(f"          berkas ini seharusnya beku; periksa apa yang mengubahnya")

    if args.cek:
        print("\n(--cek: tidak ada yang diubah)")
        return
    if not baru:
        print("\ntidak ada berkas baru untuk dibekukan")
        return

    if os.path.exists(FREEZE):
        os.chmod(FREEZE, stat.S_IWUSR | stat.S_IRUSR)
    for nama, h, path, catatan in baru:
        beku[nama] = {"sha256": h, "catatan": catatan}
        n = baris_berkas(path)
        if n is not None:
            beku[nama]["baris"] = n
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    json.dump(beku, open(FREEZE, "w"), indent=1)
    os.chmod(FREEZE, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"\n{len(baru)} berkas dibekukan dan dicatat di {FREEZE}")


if __name__ == "__main__":
    main()
