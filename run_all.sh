#!/bin/bash
# run_all.sh - menjalankan seluruh eksperimen utama secara berurutan.
#
# Urutan disusun agar pasangan terpenting selesai lebih dulu:
#   baseline seed 2026 + augmented seed 2026  =  perbandingan inti (RM 2)
# Setelah itu seed 2027 dan 2028 untuk simpangan baku (Subbab 3.8.3).
#
# Aman ditinggal: memakai caffeinate agar Mac tidak tidur, dan melewati
# proses yang checkpoint-nya sudah ada sehingga dapat dijalankan ulang.
#
#   bash run_all.sh
#   tail -f results/run_all.log
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1      # agar progres langsung tampil di log, tidak tertahan buffer

LOG=results/run_all.log
mkdir -p results checkpoints
echo "=== MULAI $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

latih () {                 # $1 = tag checkpoint, $2.. = argumen train.py
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

# ---------------- eksperimen utama (6 proses) ----------------
latih baseline_seed2026            --seed 2026
latih augmented_seed2026 --augment --seed 2026
latih baseline_seed2027            --seed 2027
latih augmented_seed2027 --augment --seed 2027
latih baseline_seed2028            --seed 2028
latih augmented_seed2028 --augment --seed 2028

# ---------------- evaluasi seluruh skenario ----------------
echo "[$(date '+%H:%M:%S')] === EVALUASI ===" | tee -a "$LOG"
for ck in checkpoints/*.pt; do
  [ -e "$ck" ] || continue
  nama=$(basename "$ck" .pt)
  echo "[$(date '+%H:%M:%S')] evaluasi ${nama}" | tee -a "$LOG"
  python -u src/evaluate.py --checkpoint "$ck" --scenario S1 >> "results/eval_${nama}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S2 --per-attack >> "results/eval_${nama}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S3 >> "results/eval_${nama}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S4 \
         --manifest manifests/in_the_wild.csv --condition in_the_wild >> "results/eval_${nama}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S4 \
         --manifest manifests/wavefake.csv --condition wavefake >> "results/eval_${nama}.log" 2>&1
done

# ---------------- tabel untuk Bab 4 ----------------
python src/aggregate_results.py 2>&1 | tee -a "$LOG"
echo "=== SELESAI SEMUA $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
