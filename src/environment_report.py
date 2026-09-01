# src/environment_report.py
# Menghasilkan results/environment.txt sesuai Panduan Pelaksanaan Skripsi:
# "environment.txt -> OS, CPU, GPU, Python, PyTorch, FFmpeg | Audit lingkungan komputasi."
#
#   python src/environment_report.py
import os, sys, platform, subprocess, datetime
sys.path.insert(0, os.path.dirname(__file__))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15).stdout.strip() or "-"
    except Exception:
        return "-"


def main():
    import torch
    import torchaudio
    baris = []
    add = baris.append

    add("LAPORAN LINGKUNGAN KOMPUTASI")
    add(f"dibuat            : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    add("")
    add("-- Perangkat keras dan sistem --")
    add(f"OS                : {platform.system()} {platform.release()} "
        f"({sh('sw_vers -productVersion') if platform.system() == 'Darwin' else platform.version()})")
    add(f"Arsitektur        : {platform.machine()}")
    add(f"CPU               : {sh('sysctl -n machdep.cpu.brand_string') if platform.system() == 'Darwin' else platform.processor()}")
    add(f"Inti CPU          : {os.cpu_count()}")
    if platform.system() == "Darwin":
        mem = sh("sysctl -n hw.memsize")
        add(f"Memori            : {int(mem)//(1024**3) if mem.isdigit() else '-'} GB (terpadu)")
        gpu_cmd = "system_profiler SPDisplaysDataType | grep 'Chipset Model' | head -1 | cut -d: -f2"
        core_cmd = "system_profiler SPDisplaysDataType | grep 'Total Number of Cores' | head -1 | cut -d: -f2"
        add(f"GPU               : {sh(gpu_cmd).strip()} ({sh(core_cmd).strip()} inti)")
    add("")
    add("-- Akselerator --")
    add(f"CUDA tersedia     : {torch.cuda.is_available()}")
    add(f"MPS tersedia      : {torch.backends.mps.is_available()}")
    add(f"MPS terbangun     : {torch.backends.mps.is_built()}")
    add(f"PYTORCH_ENABLE_MPS_FALLBACK : {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '-')}")
    add("")
    add("-- Perangkat lunak --")
    add(f"Python            : {platform.python_version()} ({sys.executable})")
    add(f"PyTorch           : {torch.__version__}")
    add(f"torchaudio        : {torchaudio.__version__}")
    for mod in ("numpy", "scipy", "sklearn", "pandas", "soundfile", "yaml", "matplotlib"):
        try:
            m = __import__(mod)
            add(f"{mod:<18}: {getattr(m, '__version__', '-')}")
        except Exception:
            add(f"{mod:<18}: TIDAK TERPASANG")
    add(f"FFmpeg            : {sh('ffmpeg -version | head -1')}")
    add(f"Git commit        : {sh('git rev-parse --short HEAD')}")
    add("")
    add("-- Daftar paket lengkap (pip freeze) --")
    add(sh(f"{sys.executable} -m pip freeze"))

    os.makedirs("results", exist_ok=True)
    out = "results/environment.txt"
    open(out, "w").write("\n".join(baris) + "\n")
    print("\n".join(baris[:26]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
