#!/bin/bash
# run_v2_validasi_terpisah.sh - eksperimen ulang: latih dan validasi tidak bercampur (saran dosen), TTS dan VC
#   di kedua sisi. Latih A01, A02, A04, A06 | validasi A03, A05 (partisi dev) | uji A07-A19, batch 32.
# Hasil lama (checkpoints/*.pt, results/*) TIDAK disentuh; semua keluaran di
#   checkpoints/v2_validasi_terpisah/  dan  results/v2_validasi_terpisah/
#
# Urutan: (1) 6 model utama, tiap model langsung dievaluasi S1, S2, S4 (cepat);
# (2) stress test S3 model utama (~1 jam/model); (3) 21 ablation (augmentasi: S2;
# fitur: S1, S2, S4), sama dengan rancangan Tabel 6.2.
# Aman ditinggal (caffeinate) dan aman dijalankan ulang (melewati yang sudah ada).
#
#   bash run_v2_validasi_terpisah.sh
#   tail -f results/v2_validasi_terpisah/logs/run.log
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

CFG=configs/v2_validasi_terpisah.yaml
CK=checkpoints/v2_validasi_terpisah
R=results/v2_validasi_terpisah
L=$R/logs
LOG=$L/run.log
mkdir -p "$CK" "$L"
lapor () { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
lapor "=== MULAI split latih A01,A02,A04,A06 / validasi A03,A05 / uji A07-A19 ==="

latih () {                 # $1 = tag, $2.. = argumen train.py
  local tag=$1; shift
  if [ -f "$CK/${tag}.pt" ] && grep -q "SELESAI ${tag}" "$L/train_${tag}.log" 2>/dev/null; then
    lapor "LEWAT  latih ${tag} (sudah selesai)"; return
  fi
  lapor "MULAI  latih ${tag}"
  python -u src/train.py --config "$CFG" "$@" >> "$L/train_${tag}.log" 2>&1
  lapor "SELESAI latih ${tag} -> $(tail -1 "$L/train_${tag}.log")"
}

evaluasi_inti () {         # S1, S2 per serangan, S4 In-the-Wild dan WaveFake (LJSpeech)
  local tag=$1 ck="$CK/$1.pt"
  [ -f "$ck" ] || { lapor "GAGAL  ${tag}: checkpoint tidak ada"; return; }
  if grep -q "INTI SELESAI" "$L/eval_${tag}.log" 2>/dev/null; then lapor "LEWAT  eval inti ${tag}"; return; fi
  lapor "EVAL   ${tag} S1 + S2 + S4"
  python -u src/evaluate.py --checkpoint "$ck" --scenario S1 >> "$L/eval_${tag}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S2 --per-attack >> "$L/eval_${tag}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S4 \
         --manifest manifests/in_the_wild.csv --condition in_the_wild >> "$L/eval_${tag}.log" 2>&1
  python -u src/evaluate.py --checkpoint "$ck" --scenario S4 \
         --manifest manifests/wavefake_ljspeech.csv --condition wavefake_ljspeech >> "$L/eval_${tag}.log" 2>&1
  echo "INTI SELESAI" >> "$L/eval_${tag}.log"
  lapor "SELESAI eval inti ${tag}"
}

# ---------------- latih + evaluasi inti (6 proses) ----------------
for sd in 2026 2027 2028; do
  latih baseline_seed${sd}            --seed ${sd}
  evaluasi_inti baseline_seed${sd}
  latih augmented_seed${sd} --augment --seed ${sd}
  evaluasi_inti augmented_seed${sd}
done

# ---------------- stress test S3 (unseen lalu seen) ----------------
for sd in 2026 2027 2028; do
  for sk in baseline augmented; do
    tag=${sk}_seed${sd}; ck="$CK/${tag}.pt"
    [ -f "$ck" ] || continue
    if grep -q "S3 SELESAI" "$L/eval_${tag}.log" 2>/dev/null; then lapor "LEWAT  S3 ${tag}"; continue; fi
    lapor "EVAL   ${tag} S3 unseen + seen"
    python -u src/evaluate.py --checkpoint "$ck" --scenario S3 >> "$L/eval_${tag}.log" 2>&1
    python -u src/evaluate.py --checkpoint "$ck" --scenario S3 --attacks known_eval >> "$L/eval_${tag}.log" 2>&1
    echo "S3 SELESAI" >> "$L/eval_${tag}.log"
    lapor "SELESAI S3 ${tag}"
  done
done
# ---------------- ablation (21 proses) ----------------
eval_s2 () {               # ablation augmentasi: hanya S2 per serangan (sesuai rancangan)
  local tag=$1 ck="$CK/$1.pt"
  [ -f "$ck" ] || { lapor "GAGAL  ${tag}: checkpoint tidak ada"; return; }
  if grep -q "INTI SELESAI" "$L/eval_${tag}.log" 2>/dev/null; then lapor "LEWAT  eval ${tag}"; return; fi
  lapor "EVAL   ${tag} S2"
  python -u src/evaluate.py --checkpoint "$ck" --scenario S2 --per-attack >> "$L/eval_${tag}.log" 2>&1
  echo "INTI SELESAI" >> "$L/eval_${tag}.log"
}
for sd in 2026 2027 2028; do
  for t in noise resample lowpass reverb gain; do
    latih abl_noaug_${t}_seed${sd} --augment --disable ${t} --seed ${sd} --tag abl_noaug_${t}
    eval_s2 abl_noaug_${t}_seed${sd}
  done
done
for sd in 2026 2027 2028; do
  for c in delta static; do
    latih abl_feat_${c}_seed${sd} --components ${c} --seed ${sd} --tag abl_feat_${c}
    evaluasi_inti abl_feat_${c}_seed${sd}
  done
done
lapor "=== SELESAI SEMUA ==="
