# src/model_lcnn.py
# LCNN dengan Max-Feature-Map, sesuai Tabel 3.3 proposal.
#
#   Tahap  Operasi                                    masuk -> keluar
#   -----  ----------------------------------------   ---------------
#     1    Conv 5x5, MFM 2/1                              1 -> 32
#     2    Max-pool 2x2                                  32 -> 32
#     3    Conv 1x1 & 3x3, MFM 2/1, BatchNorm            32 -> 48
#     4    Max-pool 2x2                                  48 -> 48
#     5    Conv 1x1 & 3x3, MFM 2/1, BatchNorm            48 -> 64
#     6    Max-pool 2x2                                  64 -> 64
#     7    Conv 1x1 & 3x3, MFM 2/1, BatchNorm            64 -> 32
#     8    Max-pool 2x2 + Dropout p = 0,7                32 -> 32
#     9    Average pooling sepanjang sumbu waktu         32 -> 32
#    10    Fully connected 512, MFM 2/1                     -> 256
#    11    Fully connected 2                            256 -> 2
#
# embedding_dim=256 sesuai Panduan Pelaksanaan Skripsi Tahap 4
# ("Output minimum ... embedding 256 dimensi").
import torch
import torch.nn as nn


class MFM(nn.Module):
    """Max-Feature-Map 2/1 (Persamaan 2.8). Bekerja pada tensor 4D (kanal)
    maupun 2D (neuron), keduanya membelah dim=1."""
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.maximum(a, b)


class MFMBlock(nn.Module):
    """Conv 1x1 -> MFM -> BN -> Conv 3x3 -> MFM -> BN."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_in * 2, kernel_size=1), MFM(), nn.BatchNorm2d(c_in),
            nn.Conv2d(c_in, c_out * 2, kernel_size=3, padding=1), MFM(), nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        return self.body(x)


class LCNN(nn.Module):
    def __init__(self, n_feat=180, embedding_dim=256, dropout=0.7):
        """n_feat = tinggi peta masukan (180 statis+Δ+ΔΔ, 120 statis+Δ, 60 statis)."""
        super().__init__()
        self.n_feat = n_feat
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2), MFM(),   # tahap 1 -> 32
            nn.MaxPool2d(2),                                     # tahap 2
            MFMBlock(32, 48),                                    # tahap 3
            nn.MaxPool2d(2),                                     # tahap 4
            MFMBlock(48, 64),                                    # tahap 5
            nn.MaxPool2d(2),                                     # tahap 6
            MFMBlock(64, 32),                                    # tahap 7
            nn.MaxPool2d(2), nn.Dropout(dropout),                # tahap 8
        )
        # empat kali max-pool 2x2 -> tinggi peta menyusut 16 kali
        f_out = n_feat
        for _ in range(4):
            f_out //= 2
        if f_out < 1:
            raise ValueError(f"n_feat={n_feat} terlalu kecil untuk 4 kali max-pool")
        self.flat_dim = 32 * f_out
        self.fc1 = nn.Linear(self.flat_dim, embedding_dim * 2)   # tahap 10: FC 512
        self.mfm_fc = MFM()                                      #           MFM -> 256
        self.classifier = nn.Linear(embedding_dim, 2)            # tahap 11

    def forward(self, x):                       # x: [B, 1, n_feat, T]
        h = self.features(x)                    # [B, 32, F', T']
        h = h.mean(dim=3)                       # tahap 9: rerata sumbu waktu -> [B, 32, F']
        h = torch.flatten(h, 1)                 # [B, 32*F']
        z = self.mfm_fc(self.fc1(h))            # embedding [B, 256]
        return self.classifier(z), z


if __name__ == "__main__":
    for n_feat in (180, 120, 60):
        m = LCNN(n_feat=n_feat)
        logits, z = m(torch.randn(2, 1, n_feat, 397))
        n_par = sum(p.numel() for p in m.parameters())
        print(f"n_feat={n_feat:3d} | flat={m.flat_dim:4d} | logits={tuple(logits.shape)} "
              f"| embed={tuple(z.shape)} | parameter={n_par:,}")
