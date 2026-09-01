# src/build_manifest.py
from pathlib import Path
import hashlib, soundfile as sf, pandas as pd


def sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def inspect_audio(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate, info.channels


rows = []
for path in Path("data/raw").rglob("*.flac"):
    duration, sr, channels = inspect_audio(path)
    if duration < 1.0:
        continue
    rows.append({
        "source_file_id": path.stem,
        "file_path": str(path),
        "speaker_id": "FILL_FROM_PROTOCOL",
        "attack_id": "FILL_FROM_PROTOCOL",
        "label": "FILL_FROM_PROTOCOL",
        "dataset": "ASVspoof2019_LA",
        "split": "FILL_FROM_PROTOCOL",
        "duration_s": duration,
        "sha256": sha256(path),
    })

pd.DataFrame(rows).to_csv("manifests/source_manifest.csv", index=False)
print("Rows:", len(rows))
