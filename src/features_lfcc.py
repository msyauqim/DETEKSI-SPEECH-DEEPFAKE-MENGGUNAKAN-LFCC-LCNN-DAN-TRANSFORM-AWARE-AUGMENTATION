# src/features_lfcc.py
# Ekstraksi LFCC sesuai Tabel 3.2 proposal dan Tahap 3 Panduan Pelaksanaan Skripsi.
#
#   Laju cuplik            16 kHz
#   Panjang frame          25 ms (400 cuplikan)
#   Pergeseran frame       10 ms (160 cuplikan)
#   Jendela                Hann
#   Panjang DFT            512 titik
#   Kanal filterbank       128       (harus >= jumlah koefisien; lihat Subbab 2.3.3)
#   Koefisien statis       60
#   Turunan                delta + delta-delta  ->  180 dimensi per frame
#   Normalisasi            rerata & simpangan baku partisi LATIH (CMVN)
import torch
import torchaudio

# jumlah salinan komponen fitur, dipakai untuk ablation (Tabel 3.7, RM 4)
COMPONENTS = {
    "static": 1,        # 60 dimensi
    "delta": 2,         # 120 dimensi
    "delta2": 3,        # 180 dimensi (konfigurasi penuh)
}

WINDOWS = {
    "hamming": torch.hamming_window,
    "hann": torch.hann_window,
}


class LFCCFeature(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_filter=128, n_lfcc=60, n_fft=512,
                 win_length=400, hop_length=160, log_lf=True, window="hann",
                 components="delta2", norm_stats=None):
        super().__init__()
        if n_lfcc > n_filter:
            raise ValueError(
                f"n_lfcc ({n_lfcc}) tidak boleh melampaui n_filter ({n_filter}); "
                "DCT atas M kanal filterbank hanya menghasilkan paling banyak M koefisien.")
        if components not in COMPONENTS:
            raise ValueError(f"components harus salah satu dari {list(COMPONENTS)}")
        if window not in WINDOWS:
            raise ValueError(f"window harus salah satu dari {list(WINDOWS)}")
        self.components = components
        self.n_lfcc = n_lfcc
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            log_lf=log_lf,
            speckwargs={"n_fft": n_fft, "win_length": win_length,
                        "hop_length": hop_length, "center": False,
                        "window_fn": WINDOWS[window]},
        )
        self.set_norm_stats(norm_stats)

    # ---------- normalisasi ----------
    def set_norm_stats(self, stats):
        """stats: dict {'mean': [D], 'std': [D]} dari partisi latih, atau None."""
        if stats is None:
            self.register_buffer("mean", None, persistent=False)
            self.register_buffer("std", None, persistent=False)
            return
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).view(1, -1, 1).clamp_min(1e-6)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @property
    def out_dim(self):
        return self.n_lfcc * COMPONENTS[self.components]

    # ---------- alur ----------
    def forward(self, wav, normalize=True):     # wav: [B, 1, T]
        x = self.lfcc(wav.squeeze(1))           # [B, n_lfcc, frames]
        if self.components == "static":
            feat = x
        elif self.components == "delta":
            feat = torch.cat([x, torchaudio.functional.compute_deltas(x)], dim=1)
        else:
            d1 = torchaudio.functional.compute_deltas(x)          # Persamaan 2.7
            d2 = torchaudio.functional.compute_deltas(d1)
            feat = torch.cat([x, d1, d2], dim=1)                  # [B, 180, frames]
        if normalize and getattr(self, "mean", None) is not None:
            feat = (feat - self.mean) / self.std
        return feat.unsqueeze(1)                # [B, 1, D, frames]


if __name__ == "__main__":
    wav = torch.randn(2, 1, 16000 * 4).clamp(-1, 1)
    for comp in ("static", "delta", "delta2"):
        f = LFCCFeature(components=comp)
        print(f"{comp:7s} -> {tuple(f(wav).shape)}  out_dim={f.out_dim}")
    try:
        LFCCFeature(n_filter=20, n_lfcc=60)
    except ValueError as e:
        print("penjagaan i<=M bekerja:", str(e)[:62], "...")
