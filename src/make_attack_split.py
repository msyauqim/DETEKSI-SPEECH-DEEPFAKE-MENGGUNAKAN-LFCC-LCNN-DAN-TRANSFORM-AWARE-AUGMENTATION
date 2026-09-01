# src/make_attack_split.py
# Membekukan peran attack sesuai protokol RESMI ASVspoof 2019 LA:
#   seen   = A01-A06  (muncul di train & dev)
#   unseen = A07-A19  (hanya muncul di eval)
#   bonafide = label 0
import pandas as pd

SEEN_ATTACKS = {"A01", "A02", "A03", "A04", "A05", "A06"}

df = pd.read_csv("manifests/source_manifest.csv")

df["attack_role"] = "unseen"                       # default untuk semua spoof
df.loc[df.attack_id.isin(SEEN_ATTACKS), "attack_role"] = "seen"
df.loc[df.label == 0, "attack_role"] = "bonafide"  # bona fide bukan seen/unseen

df.to_csv("manifests/split_attack.csv", index=False)
print(df.groupby(["split", "attack_role", "label"]).size())
