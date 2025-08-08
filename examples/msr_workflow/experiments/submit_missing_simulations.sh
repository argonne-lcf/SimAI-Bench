#!/bin/bash
# Usage: bash submit_missing_simulations.sh
# This script checks which simulation experiments are missing based on log directories and submits jobs for those.

sizes=(10000 100000 500000 1000000 2000000 4000000 8000000 16000000 32000000 64000000)
if [ -z "$1" ]; then
    echo "Error: server_location argument is required."
    echo "Usage: bash submit_missing_simulations.sh <server_location>"
    exit 1
fi
server_location="$1"
ndb=1
exp=0
declare -A job_scripts
job_scripts=(
    [redis_simulation]="job_script_redis_simulation.sh"
    [redis_training]="job_script_redis_training.sh"
    [filesystem_simulation]="job_script_filesystem.sh"
    [filesystem_training]="job_script_filesystem.sh"
    [dragon_simulation]="job_script_dragon_simulation.sh"
    [dragon_training]="job_script_dragon_training.sh"
)
declare -A configs
configs=(
    [redis]="configs/server/redis.json"
    [filesystem]="configs/server/filesystem.json"
    [dragon]="configs/server/dragon.json"
)

for node in 2 8 32 128 512; do
    node_dir="${node}nodes"
    cd "$node_dir" || continue
    for backend in dragon filesystem redis; do
        label1="${backend}_${server_location}"
        job_script="${job_scripts[$label1]}"
        config="${configs[$backend]}"
        for idx in "${!sizes[@]}"; do
            size="${sizes[$idx]}"
            label2=$(printf "%.2fm_%sdb_exp%d" $(echo "$size / 1000000" | bc -l) "$ndb" "$exp")
            log_dir="logs_${label1}_${label2}"
            if [ ! -d "$log_dir" ]; then
                echo "Missing: $log_dir. Submitting job for $backend size_idx=$idx (size=$size) in $node_dir with $job_script in dir $PWD."
                qsub -v size_idx=$idx $job_script
                sleep 1
            else
                echo "Found: $log_dir. Skipping $backend $size in $node_dir."
            fi
        done
    done
    cd ../
done
