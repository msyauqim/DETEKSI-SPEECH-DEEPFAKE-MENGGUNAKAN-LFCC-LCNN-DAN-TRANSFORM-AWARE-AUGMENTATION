# src/augment.py
# Transform-aware augmentation sesuai Tabel 3.4 proposal.
#
#   Transformasi          Rentang parameter                       Peluang
#   -------------------   -------------------------------------   -------
#   Perubahan gain        -6 dB .. +6 dB                          0,50
#   Derau aditif          SNR 5 dB .. 25 dB                       0,40
#   Cuplik ulang          8 / 11,025 / 12 kHz -> kembali 16 kHz   0,30
#   Tapis lolos rendah    frekuensi potong 3,4 .. 7 kHz           0,30
#   Simulasi dengung      RT60 0,15 .. 0,7 detik                  0,25
#
# Tiga aturan yang dipegang (Subbab 3.6):
#   1. Peluang & rentang sama untuk kelas bona fide maupun spoof.
#   2. Hanya aktif pada partisi latih (dikendalikan pemanggil).
#   3. Seluruh bilangan acak berasal dari generator ber-seed yang dicatat.
import math
import torch
import torchaudio


def fft_convolve(x, h):
    """Konvolusi x [C, T] dengan tanggapan impuls h [L] melalui FFT.
    Konvolusi langsung berbiaya O(T*L); lewat FFT menjadi O(n log n).
    Untuk T = 64.000 dan L = 11.200 selisihnya lebih dari seratus kali."""
    T = x.shape[-1]
    n = T + h.shape[-1] - 1
    n_fft = 1 << (n - 1).bit_length()          # pangkat dua terdekat
    y = torch.fft.irfft(torch.fft.rfft(x, n_fft) * torch.fft.rfft(h, n_fft), n_fft)
    return y[..., :T]                           # Persamaan 2.13


def simulate_rir(rt60, sr=16000, generator=None, direct_delay=0.002):
    """Tanggapan impuls ruangan sintetis: derau putih dengan selubung
    peluruhan eksponensial. RT60 = waktu bagi energi meluruh 60 dB."""
    length = max(int(sr * rt60), 8)
    t = torch.arange(length, dtype=torch.float32) / sr
    decay = torch.pow(10.0, -3.0 * t / rt60)             # -60 dB pada t = RT60
    noise = torch.randn(length, generator=generator)
    h = noise * decay
    d = int(sr * direct_delay)
    h[:d] = 0.0
    h[d] = 1.0                                            # jalur langsung
    return h / h.abs().max().clamp_min(1e-8)


class TransformAwareAugment:
    def __init__(self, sr=16000, cfg=None, generator=None):
        cfg = cfg or {}
        self.sr = sr
        self.g = generator                                # torch.Generator, boleh None
        self.p_gain = cfg.get("p_gain", 0.50)
        self.p_noise = cfg.get("p_noise", 0.40)
        self.p_resample = cfg.get("p_resample", 0.30)
        self.p_lowpass = cfg.get("p_lowpass", 0.30)
        self.p_reverb = cfg.get("p_reverb", 0.25)
        self.gain_db = tuple(cfg.get("gain_db", (-6.0, 6.0)))
        self.snr_db = tuple(cfg.get("snr_db", (5.0, 25.0)))
        self.resample_rates = list(cfg.get("resample_rates", (8000, 11025, 12000)))
        self.lowpass_hz = tuple(cfg.get("lowpass_hz", (3400.0, 7000.0)))
        self.rt60_s = tuple(cfg.get("rt60_s", (0.15, 0.70)))
        # transformasi yang dimatikan untuk keperluan ablation
        self.disabled = set(cfg.get("disable", []))

    # ---------- util acak ber-seed ----------
    def _u(self, lo=0.0, hi=1.0):
        return float(torch.empty(1).uniform_(lo, hi, generator=self.g).item())

    def _choice(self, seq):
        i = int(torch.randint(len(seq), (1,), generator=self.g).item())
        return seq[i]

    def _hit(self, p, name):
        return name not in self.disabled and self._u() < p

    # ---------- transformasi ----------
    def _gain(self, wav):
        db = self._u(*self.gain_db)
        return wav * (10.0 ** (db / 20.0))

    def _noise(self, wav):
        snr = self._u(*self.snr_db)
        noise = torch.randn(wav.shape, generator=self.g)
        p_x = wav.pow(2).mean().clamp_min(1e-12)
        p_d = noise.pow(2).mean().clamp_min(1e-12)
        g = torch.sqrt(p_x / (p_d * (10.0 ** (snr / 10.0))))   # Persamaan 2.11
        return wav + g * noise                                  # Persamaan 2.12

    def _resample(self, wav):
        mid = self._choice(self.resample_rates)
        wav = torchaudio.functional.resample(wav, self.sr, mid)
        return torchaudio.functional.resample(wav, mid, self.sr)

    def _lowpass(self, wav):
        cutoff = self._u(*self.lowpass_hz)
        return torchaudio.functional.lowpass_biquad(wav, self.sr, cutoff)

    def _reverb(self, wav):
        rt60 = self._u(*self.rt60_s)
        h = simulate_rir(rt60, self.sr, generator=self.g).to(wav.dtype)
        return fft_convolve(wav, h)                             # Persamaan 2.13

    # ---------- alur ----------
    def __call__(self, wav):
        """wav: [C, T] ternormalisasi. Kembalikan (wav, daftar transformasi)."""
        applied = []
        if self._hit(self.p_gain, "gain"):
            wav = self._gain(wav); applied.append("gain")
        if self._hit(self.p_noise, "noise"):
            wav = self._noise(wav); applied.append("noise")
        if self._hit(self.p_resample, "resample"):
            wav = self._resample(wav); applied.append("resample")
        if self._hit(self.p_lowpass, "lowpass"):
            wav = self._lowpass(wav); applied.append("lowpass")
        if self._hit(self.p_reverb, "reverb"):
            wav = self._reverb(wav); applied.append("reverb")
        peak = wav.abs().max().clamp_min(1e-6)
        if peak > 1.0:
            wav = wav / peak
        return wav.clamp(-1.0, 1.0), (applied or ["clean"])


# ---------- versi deterministik untuk skenario stress-test (S3) ----------
STRESS_LEVELS = {
    "noise":    [20.0, 12.5, 5.0],          # SNR dB, makin kecil makin berat
    "resample": [12000, 11025, 8000],       # laju antara
    "lowpass":  [7000.0, 5200.0, 3400.0],   # frekuensi potong Hz
    "reverb":   [0.15, 0.40, 0.70],         # RT60 detik
    "gain":     [2.0, 4.0, 6.0],            # |dB|
}


def apply_stress(wav, kind, level, sr=16000, generator=None):
    """Terapkan SATU transformasi dengan parameter TETAP (bukan diacak).
    level: indeks 0..2 pada STRESS_LEVELS[kind]."""
    v = STRESS_LEVELS[kind][level]
    aug = TransformAwareAugment(sr=sr, generator=generator)
    if kind == "gain":
        wav = wav * (10.0 ** (v / 20.0))
    elif kind == "noise":
        noise = torch.randn(wav.shape, generator=generator)
        p_x = wav.pow(2).mean().clamp_min(1e-12)
        p_d = noise.pow(2).mean().clamp_min(1e-12)
        wav = wav + torch.sqrt(p_x / (p_d * (10.0 ** (v / 10.0)))) * noise
    elif kind == "resample":
        wav = torchaudio.functional.resample(wav, sr, int(v))
        wav = torchaudio.functional.resample(wav, int(v), sr)
    elif kind == "lowpass":
        wav = torchaudio.functional.lowpass_biquad(wav, sr, v)
    elif kind == "reverb":
        h = simulate_rir(v, sr, generator=generator).to(wav.dtype)
        wav = fft_convolve(wav, h)
    else:
        raise ValueError(f"transformasi tidak dikenal: {kind}")
    peak = wav.abs().max().clamp_min(1e-6)
    if peak > 1.0:
        wav = wav / peak
    return wav.clamp(-1.0, 1.0)


if __name__ == "__main__":
    g = torch.Generator().manual_seed(2026)
    aug = TransformAwareAugment(generator=g)
    x = torch.randn(1, 16000 * 4).clamp(-1, 1)
    for _ in range(5):
        y, names = aug(x)
        print(f"{str(names):<45} shape={tuple(y.shape)} peak={y.abs().max():.3f}")
    for k in STRESS_LEVELS:
        y = apply_stress(x, k, 2)
        print(f"stress {k:9s} -> shape={tuple(y.shape)} peak={y.abs().max():.3f}")
