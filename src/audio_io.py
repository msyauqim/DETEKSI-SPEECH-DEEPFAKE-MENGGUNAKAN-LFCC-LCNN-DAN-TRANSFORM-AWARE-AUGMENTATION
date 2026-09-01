# src/audio_io.py
# Loader audio 4 detik. Memakai soundfile untuk membaca (FLAC/WAV) agar tidak
# bergantung pada torchcodec/ffmpeg yang diwajibkan torchaudio.load() versi baru.
import torch, torchaudio
import torch.nn.functional as F
import soundfile as sf


def load_audio_4s(path, training=False, target_sr=16000, seconds=4):
    # soundfile mengembalikan [frames, channels]; ubah ke [channels, frames]
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).t().contiguous()   # [C, T]
    wav = wav.mean(dim=0, keepdim=True)              # mono [1, T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    target = target_sr * seconds
    n = wav.shape[-1]
    if n < target:
        wav = F.pad(wav, (0, target - n))
    elif n > target:
        if training:
            start = torch.randint(0, n - target + 1, (1,)).item()
        else:
            start = (n - target) // 2
        wav = wav[:, start:start + target]
    peak = wav.abs().max().clamp_min(1e-6)
    return (wav / peak).float()
