#!/bin/bash
# run_eval_all.sh - evaluasi S1-S4 untuk seluruh checkpoint, melewati yang sudah tercatat.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOG=results/run_eval_all.log
echo "=== MULAI EVALUASI $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"

sudah_ada () {   # $1=checkpoint(.pt), $2=scenario, $3=condition
  [ -f results/eval_summary.csv ] || return 1
  python3 -c "
import pandas as pd, sys
d = pd.read_csv('results/eval_summary.csv')
mask = (d.checkpoint=='$1') & (d.scenario=='$2') & (d.condition=='$3')
sys.exit(0 if mask.any() else 1)
" 2>/dev/null
}

for ck in checkpoints/*.pt; do
  nama=$(basename "$ck" .pt)
  ckfile=$(basename "$ck")

  if sudah_ada "$ckfile" "S1" "seen"; then
    echo "[$(date '+%H:%M:%S')] LEWAT  ${nama} S1" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] EVAL   ${nama} S1" | tee -a "$LOG"
    python -u src/evaluate.py --checkpoint "$ck" --scenario S1 >> "results/eval_${nama}.log" 2>&1
  fi

  if sudah_ada "$ckfile" "S2" "unseen"; then
    echo "[$(date '+%H:%M:%S')] LEWAT  ${nama} S2" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] EVAL   ${nama} S2" | tee -a "$LOG"
    python -u src/evaluate.py --checkpoint "$ck" --scenario S2 --per-attack >> "results/eval_${nama}.log" 2>&1
  fi

  if sudah_ada "$ckfile" "S3" "clean"; then
    echo "[$(date '+%H:%M:%S')] LEWAT  ${nama} S3" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] EVAL   ${nama} S3" | tee -a "$LOG"
    python -u src/evaluate.py --checkpoint "$ck" --scenario S3 >> "results/eval_${nama}.log" 2>&1
  fi

  if sudah_ada "$ckfile" "S4" "in_the_wild"; then
    echo "[$(date '+%H:%M:%S')] LEWAT  ${nama} S4-ITW" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] EVAL   ${nama} S4-ITW" | tee -a "$LOG"
    python -u src/evaluate.py --checkpoint "$ck" --scenario S4 \
           --manifest manifests/in_the_wild.csv --condition in_the_wild >> "results/eval_${nama}.log" 2>&1
  fi

  if sudah_ada "$ckfile" "S4" "wavefake"; then
    echo "[$(date '+%H:%M:%S')] LEWAT  ${nama} S4-WaveFake" | tee -a "$LOG"
  else
    echo "[$(date '+%H:%M:%S')] EVAL   ${nama} S4-WaveFake" | tee -a "$LOG"
    python -u src/evaluate.py --checkpoint "$ck" --scenario S4 \
           --manifest manifests/wavefake.csv --condition wavefake >> "results/eval_${nama}.log" 2>&1
  fi
done

echo "[$(date '+%H:%M:%S')] === MEMBUAT TABEL HASIL ===" | tee -a "$LOG"
python src/aggregate_results.py 2>&1 | tee -a "$LOG"
echo "=== SELESAI EVALUASI $(date '+%Y-%m-%d %H:%M:%S') ===" | tee -a "$LOG"
