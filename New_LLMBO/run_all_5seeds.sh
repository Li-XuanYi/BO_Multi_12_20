#!/bin/bash
# Run LLMBO-MO 5-seed experiment on Chen2020
# Each seed runs independently in background

BASE_DIR="/d/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO"
OUT_DIR="$BASE_DIR/paper_5seeds"
mkdir -p "$OUT_DIR"

export LLM_API_KEY="sk-d1ee7a7d3e594831be6ad87b4d367e4c"

SEEDS=(8409 8410 8411 8412 8413)

for SEED in "${SEEDS[@]}"; do
    LOG="$OUT_DIR/seed${SEED}_run.log"
    echo "[$(date)] Starting seed $SEED..." >> "$LOG"
    cd "$BASE_DIR" && nohup /d/Users/aa133/Desktop/BO_Multi_12_20/New_LLMBO/.venv/Scripts/python.exe \
        run_llmbo_mo_5seeds.py \
        --seeds "$SEED" \
        --n-evals 56 \
        --output-root "$OUT_DIR/seed${SEED}" \
        >> "$LOG" 2>&1 &
    echo "[$(date)] Seed $SEED started with PID $!" >> "$LOG"
done

echo "All 5 seeds started. Check logs in $OUT_DIR/ for progress."
echo "Monitor with: tail -f $OUT_DIR/seed8409_run.log"
