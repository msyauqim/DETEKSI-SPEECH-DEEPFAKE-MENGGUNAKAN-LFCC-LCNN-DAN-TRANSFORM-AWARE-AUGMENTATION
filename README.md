# Deteksi Speech Deepfake dari Generator yang Tidak Dikenal
### Menggunakan LFCC-LCNN dan Transform-Aware Augmentation

Skripsi ini membangun detektor *speech deepfake* berbasis **LFCC-LCNN** dan menguji
seberapa jauh kinerjanya bertahan pada generator sintesis suara yang **tidak pernah
dilihat saat pelatihan** (*unseen generator*), serta apakah augmentasi sadar-transformasi
(*transform-aware augmentation*) membantu ketahanan tersebut.

## Hasil utama (rerata ± simpangan baku, 3 seed)

| Skema | Kondisi | EER |
|---|---|---:|
| baseline (clean-only) | seen (A16, A19) | 0,11% ± 0,06 |
| baseline (clean-only) | **unseen** (A07–A18) | 18,13% ± 1,42 |
| augmented | seen (A16, A19) | 1,46% ± 0,22 |
| augmented | **unseen** (A07–A18) | **8,48% ± 1,02** |

Augmentasi menurunkan EER pada serangan *unseen* lebih dari separuh, dengan
penurunan kinerja *matched* yang kecil. Rincian lengkap (S1–S4, *stress-test*,
per-serangan) ada di `results/tabel_hasil.csv`.

## Struktur proyek

```
src/            kode sumber (ekstraksi fitur, model, augmentasi, training, evaluasi)
configs/        konfigurasi eksperimen (parameter tunggal, tidak tersebar di kode)
manifests/      daftar berkas dataset + label (path relatif, tidak berisi audio)
checkpoints/    model terlatih (.pt)
results/        skor, ringkasan metrik, log, bukti lingkungan komputasi
data/           dataset mentah -- TIDAK disertakan di repo (lihat di bawah)
```

## Dataset

Dataset **tidak disertakan** di repo ini (±96 GB, berlisensi riset). Unduh sendiri:

- [ASVspoof 2019 LA](https://www.asvspoof.org/index2019.html) -> letakkan di `data/raw/LA/`
- [WaveFake](https://zenodo.org/records/5642694) -> `data/raw/wavefake/`
- [In-the-Wild](https://deepfake-total.com/in_the_wild) -> `data/raw/in_the_wild/`

Setelah dataset tersedia, susun manifest:

```bash
python src/build_manifests_all.py
python src/make_attack_split.py
python src/build_external_manifests.py --itw --wavefake
python src/audit_data.py --all
python src/compute_norm_stats.py --max-files 8000
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Perangkat diuji di Apple Silicon (MPS). Berjalan otomatis di CUDA/CPU juga
lewat `src/device.py`.

## Menjalankan eksperimen

```bash
# baseline & augmented, 3 seed
python src/train.py --seed 2026
python src/train.py --augment --seed 2026
# ...ulangi untuk seed 2027, 2028 (lihat run_all.sh untuk versi otomatis)

# evaluasi 4 skenario (S1 seen, S2 unseen, S3 stress-test, S4 lintas dataset)
python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S1
python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S2 --per-attack
python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S3
python src/evaluate.py --checkpoint checkpoints/baseline_seed2026.pt --scenario S4 \
       --manifest manifests/in_the_wild.csv --condition in_the_wild

# ringkasan akhir (rerata ± simpangan baku 3 seed)
python src/aggregate_results.py
```

Skrip `run_all.sh`, `run_ablation.sh`, dan `run_eval_all.sh` menjalankan seluruh
kombinasi di atas secara otomatis dan aman dijeda-lanjut (melewati proses yang
checkpoint-nya sudah ada).

## Konfigurasi

Semua parameter (LFCC, arsitektur LCNN, augmentasi, hiperparameter training)
ada di `configs/baseline.yaml` -- tidak ada nilai tersembunyi di kode.
