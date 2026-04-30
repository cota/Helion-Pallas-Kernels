#!/bin/bash
# Usage: benchmark.sh attention.py

export LIBTPU_INIT_ARGS="--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65536"

python3 "$@"
