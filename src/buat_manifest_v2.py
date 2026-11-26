# src/buat_manifest_v2.py
# Membentuk ulang manifest pembagian berbasis serangan (latih A01, A02, A04, A06 dari partisi train;
# validasi A03, A05 dari partisi dev) serta statistik CMVN 60-D dan 120-D sebagai irisan statistik 180-D.
# Berkas yang dipakai eksperimen sudah dibekukan di manifests/v2_validasi_terpisah/ (freeze.json);
# skrip ini menulis ke --out dan membandingkan hash SHA-256-nya dengan freeze.json.
#
#   python src/buat_manifest_v2.py --out /tmp/cek_manifest
import argparse, hashlib, json, os
import pandas as pd

LATIH, VALIDASI = ["A01", "A02", "A04", "A06"], ["A03", "A05"]
BEKU = "manifests/v2_validasi_terpisah"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=BEKU)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = pd.read_csv("manifests/train_protocol.csv")
    dv = pd.read_csv("manifests/dev_protocol.csv")
    tr[(tr.label == 0) | tr.attack_id.isin(LATIH)].to_csv(f"{a.out}/latih_A01-A02-A04-A06.csv", index=False)
    dv[(dv.label == 0) | dv.attack_id.isin(VALIDASI)].to_csv(f"{a.out}/validasi_A03-A05.csv", index=False)
    st = json.load(open(f"{BEKU}/norm_stats.json"))            # CMVN 180-D dari data latih
    for nama, d in (("static", 60), ("delta", 120)):
        sub = {"mean": st["mean"][:d], "std": st["std"][:d], "n_frames": st["n_frames"],
               "n_files": st["n_files"], "components": nama, "dim": d,
               "sumber": "irisan norm_stats.json (delta2, 180-D); fitur 180-D = [statis | delta | delta-delta], "
                         "jadi statistik per dimensi identik untuk berkas latih yang sama"}
        json.dump(sub, open(f"{a.out}/norm_stats_{nama}.json", "w"))
    beku = json.load(open(f"{BEKU}/freeze.json"))["berkas"]
    for f in ("latih_A01-A02-A04-A06.csv", "validasi_A03-A05.csv", "norm_stats_static.json", "norm_stats_delta.json"):
        p = f"{a.out}/{f}"
        print(f"{f:28s} {'COCOK' if sha(p) == beku[f]['sha256'] else 'BERBEDA'} dengan freeze.json")


if __name__ == "__main__":
    main()
