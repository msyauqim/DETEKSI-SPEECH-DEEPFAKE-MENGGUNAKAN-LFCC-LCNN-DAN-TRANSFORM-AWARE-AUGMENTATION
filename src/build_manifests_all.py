# src/build_manifests_all.py
# Bangun manifest dari protocol resmi ASVspoof2019 LA untuk train/dev/eval.
# Cepat: hanya membaca protocol (tanpa SHA-256/durasi). Audit berat -> build_manifest.py
# Jalankan dari root proyek: python src/build_manifests_all.py
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/LA")
PROTO = RAW / "ASVspoof2019_LA_cm_protocols"
OUT = Path("manifests")
OUT.mkdir(exist_ok=True)

SPLITS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev":   ("ASVspoof2019.LA.cm.dev.trl.txt",   "ASVspoof2019_LA_dev"),
    "eval":  ("ASVspoof2019.LA.cm.eval.trl.txt",  "ASVspoof2019_LA_eval"),
}

all_rows = []
for split, (proto_name, flac_dir) in SPLITS.items():
    rows = []
    for line in (PROTO / proto_name).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        speaker_id, file_id, _, attack_id, key = parts
        rows.append({
            "source_file_id": file_id,
            "file_path": f"data/raw/LA/{flac_dir}/flac/{file_id}.flac",
            "speaker_id": speaker_id,
            "label": 0 if key == "bonafide" else 1,   # bona fide=0, fake=1
            "attack_id": attack_id,                    # '-' untuk bona fide
            "dataset": "ASVspoof2019_LA",
            "split": split,
        })
    df = pd.DataFrame(rows)
    # verifikasi semua file ada
    missing = [p for p in df.file_path if not Path(p).exists()]
    df.to_csv(OUT / f"{split}_protocol.csv", index=False)
    n_bona = int((df.label == 0).sum())
    n_fake = int((df.label == 1).sum())
    print(f"[{split:5}] {len(df):6d} file | bona {n_bona:5d} | fake {n_fake:5d} "
          f"| attacks {sorted(a for a in df.attack_id.unique() if a != '-')} "
          f"| missing {len(missing)}")
    all_rows.extend(rows)

full = pd.DataFrame(all_rows)
full.to_csv(OUT / "source_manifest.csv", index=False)
print(f"\nTOTAL source_manifest.csv: {len(full)} baris")
print("Speaker overlap train∩eval:",
      len(set(full[full.split=='train'].speaker_id) & set(full[full.split=='eval'].speaker_id)))
