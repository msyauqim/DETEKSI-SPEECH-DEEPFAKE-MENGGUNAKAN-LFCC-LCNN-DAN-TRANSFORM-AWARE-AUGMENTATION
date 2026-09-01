#!/bin/bash
# run_ablation.sh - menjalankan 21 proses ablation sesuai Tabel 3.7 proposal
# (belum dijalankan otomatis oleh run_all.sh; skrip ini melengkapinya).
#
#   Kelompok augmentasi (RM 3): 5 konfigurasi x 3 seed = 15
#     matikan satu transformasi dari skema augmented penuh
#   Kelompok fitur LFCC (RM 4): 2 konfigurasi x 3 seed = 6
#     statis+delta, statis saja -- dibandingkan dgn delta2 (baseline clean-only)
#
# Total 21 proses tambahan, di atas 6 proses eksperimen utama yang sudah selesai.
# Idempotent seperti run_all.sh: melewati proses yang checkpoint-nya sudah ada.
#
#   bash run_ablation.sh
#   tail -f results/run_ablation.log
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOG=results/run_ablation.log
mkdir -p results checkpoints
echo "=== MULAI ABLATION $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

latih () {                 # $1=tag, sisanya argumen train.py
  local tag=$1; shift
  if [ -f "checkpoints/${tag}.pt" ]; then
    echo "[$(date '+%H:%M:%S')] LEWAT  ${tag} (checkpoint sudah ada)" | tee -a "$LOG"
    return
  fi
  echo "[$(date '+%H:%M:%S')] MULAI  ${tag}" | tee -a "$LOG"
  python -u src/train.py "$@" >> "results/train_${tag}.log" 2>&1
  if [ -f "checkpoints/${tag}.pt" ]; then
    echo "[$(date '+%H:%M:%S')] SELESAI ${tag} -> $(tail -1 results/train_${tag}.log)" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] GAGAL  ${tag} - lihat results/train_${tag}.log" | tee -a "$LOG"
  fi
}

# ---------------- kelompok 1: komponen augmentasi (RM 3), 5 x 3 seed ----------------
for seed in 2026 2027 2028; do
  latih "abl_noaug_noise_seed${seed}"    --augment --disable noise    --seed "$seed"
  latih "abl_noaug_resample_seed${seed}" --augment --disable resample --seed "$seed"
  latih "abl_noaug_lowpass_seed${seed}"  --augment --disable lowpass  --seed "$seed"
  latih "abl_noaug_reverb_seed${seed}"   --augment --disable reverb   --seed "$seed"
  latih "abl_noaug_gain_seed${seed}"     --augment --disable gain     --seed "$seed"
done

# ---------------- kelompok 2: komponen fitur LFCC (RM 4), 2 x 3 seed ----------------
# pembanding: baseline_seed{2026,2027,2028} (delta2, clean-only) yang sudah ada
for seed in 2026 2027 2028; do
  latih "abl_feat_delta_seed${seed}"  --components delta  --seed "$seed"
  latih "abl_feat_static_seed${seed}" --components static --seed "$seed"
done

echo "=== SELESAI ABLATION $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
echo "Jalankan berikutnya: bash run_ablation_eval.sh" | tee -a "$LOG"
