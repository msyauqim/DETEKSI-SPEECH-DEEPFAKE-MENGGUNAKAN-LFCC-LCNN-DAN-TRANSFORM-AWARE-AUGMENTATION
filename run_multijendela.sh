#!/bin/bash
# run_multijendela.sh - analisis tambahan: skor multi-jendela pada In-the-Wild dan WaveFake
# (metode ditetapkan sebelumnya, lihat src/evaluate_multijendela.py) untuk 6 model utama v1 dan v2.
# Menunggu sampai keenam model utama v2 selesai dievaluasi (berjalan bersamaan dengan fase S3).
#   bash run_multijendela.sh     |   log: results/v2_validasi_terpisah/logs/multijendela.log
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
LOG=results/v2_validasi_terpisah/logs/multijendela.log
lapor () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
lapor "menunggu 6 model utama v2 selesai dievaluasi ..."
until [ "$(grep -l "INTI SELESAI" results/v2_validasi_terpisah/logs/eval_{baseline,augmented}_seed202[678].log 2>/dev/null | wc -l | tr -d ' ')" = "6" ]; do sleep 300; done
for versi in v2_validasi_terpisah; do
  for sd in 2026 2027 2028; do
    for sk in baseline augmented; do
      ck=checkpoints/${versi}/${sk}_seed${sd}.pt; out=results/${versi}/multijendela
      for kond in in_the_wild:manifests/in_the_wild.csv wavefake_ljspeech:manifests/wavefake_ljspeech.csv; do
        c=${kond%%:*}; m=${kond#*:}
        if [ -f "$out/scores_${c}_${sk}_seed${sd}.csv" ]; then lapor "LEWAT  ${versi} ${sk}_seed${sd} ${c}"; continue; fi
        lapor "MULAI  ${versi} ${sk}_seed${sd} ${c}"
        python -u src/evaluate_multijendela.py --checkpoint "$ck" --manifest "$m" --condition "$c" --out-dir "$out" 2>&1 | tail -1 | tee -a "$LOG"
      done
    done
  done
done
lapor "=== SELESAI MULTI-JENDELA ==="
