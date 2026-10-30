#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup .venv/bin/python -u run_train_server.py > runs/train_server_batch40.log 2>&1 < /dev/null &
train_pid=$!
echo "$train_pid" > runs/train_server_batch40.pid
sleep 5
echo "TRAIN_PID=$train_pid"
printf 'CUDA_VISIBLE_DEVICES='
tr '\0' '\n' < "/proc/$train_pid/environ" | grep '^CUDA_VISIBLE_DEVICES=' || true
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
tail -n 8 runs/train_server.log
