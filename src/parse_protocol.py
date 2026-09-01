# src/parse_protocol.py
# Contoh parser protocol, sesuaikan urutan kolom dengan file resmi
import pandas as pd
from pathlib import Path

protocol = Path("data/raw/ASVspoof2019.LA.cm.train.trn.txt")
rows = []
for line in protocol.read_text().splitlines():
    parts = line.split()
    speaker_id, file_id, _, attack_id, key = parts
    rows.append({
        "source_file_id": file_id,
        "speaker_id": speaker_id,
        "attack_id": attack_id,
        "label": 0 if key == "bonafide" else 1,
        "file_path": f"data/raw/LA/ASVspoof2019_LA_train/flac/{file_id}.flac",
        "dataset": "ASVspoof2019_LA",
        "split": "train",
    })

pd.DataFrame(rows).to_csv("manifests/train_protocol.csv", index=False)
