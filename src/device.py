# src/device.py
# Helper device-agnostic. Import ini SEBELUM membuat tensor/model.
# Set fallback MPS sebagai jaring pengaman (nol overhead bila tak terpakai).
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch


def get_device():
    """Kembalikan device terbaik: MPS (Apple) > CUDA (NVIDIA) > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

if __name__ == "__main__":
    print("device:", DEVICE)
    print("PYTORCH_ENABLE_MPS_FALLBACK:", os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
