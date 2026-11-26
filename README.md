# Deteksi Speech Deepfake dari Generator yang Tidak Dikenal
### Menggunakan LFCC-LCNN dan Transform-Aware Augmentation

Skripsi ini membangun detektor *speech deepfake* berbasis **LFCC-LCNN** dan menguji
seberapa jauh kinerjanya bertahan pada generator sintesis suara yang **tidak pernah
dilihat saat pelatihan** (*unseen generator*), serta apakah augmentasi sadar-transformasi
(*transform-aware augmentation*) membantu ketahanan tersebut.

Pembagian data ASVspoof 2019 LA: latih A01, A02, A04, A06; validasi A03, A05;
uji A07–A19. Seluruh model dilatih pada tiga seed (2026, 2027, 2028).

## Hasil utama (EER, rerata ± simpangan baku 3 seed)

| Kondisi uji | Clean-only | Augmented |
|---|---:|---:|
| S1 seen (A16, A19) | 0,08% ± 0,03 | 1,35% ± 0,23 |
| S2 **unseen** (A07–A15, A17, A18) | 20,42% ± 2,38 | **10,22% ± 1,06** |
| S4 WaveFake (7 vokoder LJSpeech) | 23,73% ± 4,52 | 43,69% ± 7,41 |
| S4 In-the-Wild | 77,70% ± 7,11 | 68,30% ± 3,01 |

Augmentasi menurunkan EER pada serangan *unseen* sekitar separuhnya, dengan kenaikan
EER *seen* yang kecil. Manfaat ini tidak terbawa ke dataset lain: pada WaveFake EER justru
naik, dan pada In-the-Wild kedua skema tetap di atas 50%. Rincian per skenario ada di
`results/v2_validasi_terpisah/eval_summary.csv`.

## Struktur proyek

```
src/                               kode sumber (fitur, model, augmentasi, pelatihan, evaluasi)
configs/v2_validasi_terpisah.yaml  seluruh parameter eksperimen
manifests/                         daftar berkas dataset + label (tanpa audio)
manifests/v2_validasi_terpisah/    manifest latih/validasi dan statistik CMVN yang dibekukan
checkpoints/v2_validasi_terpisah/  27 model terlatih (.pt)
results/v2_validasi_terpisah/      ringkasan metrik dan log
results/figures/v2/                grafik naskah
results/audit/                     audit integritas data
data/                              dataset mentah -- TIDAK disertakan (lihat di bawah)
```

## Dataset

Dataset **tidak disertakan** di repo ini (±96 GB, berlisensi riset). Unduh sendiri:

- [ASVspoof 2019 LA](https://www.asvspoof.org/index2019.html) -> letakkan di `data/raw/LA/`
- [WaveFake](https://zenodo.org/records/5642694) -> `data/raw/wavefake/`
- [In-the-Wild](https://deepfake-total.com/in_the_wild) -> `data/raw/in_the_wild/`

Setelah dataset tersedia, susun dan periksa manifest:

```bash
python src/build_manifests_all.py
python src/make_attack_split.py
python src/build_external_manifests.py --itw --wavefake
python src/build_wavefake_ljspeech.py
python src/audit_data.py --all
python src/dedup_sha256.py
python src/buat_manifest_v2.py --out /tmp/cek_manifest   # cocokkan dengan freeze.json
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
# 27 proses pelatihan (6 model utama + 21 ablation) beserta evaluasi S1-S4;
# aman dijeda-lanjut (melewati proses yang sudah selesai)
bash run_v2_validasi_terpisah.sh

# analisis tambahan skor multi-jendela pada dataset eksternal
bash run_multijendela.sh

# grafik naskah
python src/grafik_skripsi_v2.py
```

Contoh satu proses secara manual:

```bash
python src/train.py --config configs/v2_validasi_terpisah.yaml --augment --seed 2026
python src/evaluate.py --checkpoint checkpoints/v2_validasi_terpisah/augmented_seed2026.pt \
       --scenario S2 --per-attack
```

## Demo

```bash
python src/web_demo.py                      # antarmuka web lokal, http://127.0.0.1:7860
python src/nilai_rekaman.py rekaman.wav     # menilai satu berkas audio
```

Demo hanya ilustrasi; hasil pada rekaman di luar dataset bukan pengukuran kinerja.
