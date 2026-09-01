# src/dataset.py
# Dataset membaca manifest CSV -> audio 4 detik (+ augmentasi opsional).
# LFCC diekstrak di train/eval loop (di device), bukan di sini.
import torch
import pandas as pd
from torch.utils.data import Dataset
from audio_io import load_audio_4s
from augment import TransformAwareAugment, apply_stress


class ASVDataset(Dataset):
    def __init__(self, manifest_csv, training=False, augment=False, augment_cfg=None,
                 sr=16000, seconds=4, subset=None, attack_role=None, attacks=None,
                 seed=2026, stress=None):
        """attacks : daftar kode serangan yang disertakan (bona fide selalu ikut).
        stress    : (jenis, tingkat) untuk skenario S3, mis. ("noise", 2)."""
        df = pd.read_csv(manifest_csv)
        if attack_role is not None and "attack_role" in df.columns:
            df = df[df.attack_role == attack_role].reset_index(drop=True)
        if attacks is not None:
            keep = df.attack_id.isin(list(attacks)) | (df.label == 0)
            df = df[keep].reset_index(drop=True)
        if subset is not None and subset < len(df):
            per = max(1, subset // 2)
            parts = [g.sample(min(len(g), per), random_state=0)
                     for _, g in df.groupby("label")]
            df = pd.concat(parts).sample(frac=1, random_state=0).reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.training = training
        self.sr = sr
        self.seconds = seconds
        self.seed = seed
        self.stress = stress
        self.augment_cfg = augment_cfg or {}
        self.use_augment = augment
        self._aug = None            # dibuat malas, agar aman pada worker DataLoader

    def __len__(self):
        return len(self.df)

    def _augmenter(self):
        if self._aug is None:
            g = torch.Generator()
            g.manual_seed(self.seed + torch.utils.data.get_worker_info().id
                          if torch.utils.data.get_worker_info() else self.seed)
            self._aug = TransformAwareAugment(self.sr, self.augment_cfg, generator=g)
        return self._aug

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = load_audio_4s(row["file_path"], training=self.training,
                            target_sr=self.sr, seconds=self.seconds)
        if self.use_augment:
            wav, _ = self._augmenter()(wav)
        if self.stress is not None:
            kind, level = self.stress
            wav = apply_stress(wav, kind, level, sr=self.sr)
        label = int(row["label"])
        attack_id = str(row["attack_id"]) if "attack_id" in row else "-"
        file_id = str(row["source_file_id"])
        return wav, label, attack_id, file_id


def collate(batch):
    wavs = torch.stack([b[0] for b in batch])          # [B,1,T]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    attacks = [b[2] for b in batch]
    file_ids = [b[3] for b in batch]
    return wavs, labels, attacks, file_ids
