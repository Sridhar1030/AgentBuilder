#!/bin/bash
set -e

NPROC=${PET_NPROC_PER_NODE:-${NUM_PROC_PER_NODE:-1}}

if [ "$NPROC" -gt 1 ]; then
    echo "Launching with torchrun --nproc_per_node=$NPROC"
    exec torchrun --nproc_per_node="$NPROC" \
        --master_addr="${MASTER_ADDR:-localhost}" \
        --master_port="${MASTER_PORT:-29500}" \
        /opt/scripts/finetune_job.py
else
    echo "Launching single-GPU mode"
    exec python /opt/scripts/finetune_job.py
fi
