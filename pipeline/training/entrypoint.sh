#!/bin/bash
set -e

NPROC=${PET_NPROC_PER_NODE:-${NUM_PROC_PER_NODE:-1}}
NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${PET_MASTER_ADDR:-${MASTER_ADDR:-localhost}}
MASTER_PORT=${PET_MASTER_PORT:-${MASTER_PORT:-29500}}

if ! [[ "$NPROC" =~ ^[0-9]+$ ]]; then
    NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    NPROC=${NPROC:-1}
    echo "Resolved non-numeric PET_NPROC_PER_NODE to $NPROC GPUs"
fi

if [ "$NPROC" -gt 1 ] || [ "$NNODES" -gt 1 ]; then
    echo "Launching with torchrun --nnodes=$NNODES --node_rank=$NODE_RANK --nproc_per_node=$NPROC --master_addr=$MASTER_ADDR"
    exec torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --nproc_per_node="$NPROC" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        /opt/scripts/finetune_job.py
else
    echo "Launching single-GPU mode"
    exec python /opt/scripts/finetune_job.py
fi
