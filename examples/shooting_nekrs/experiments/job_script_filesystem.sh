#!/bin/bash
#PBS -A datascience
#PBS -q prod
#PBS -N pfs
#PBS -l walltime=00:30:00
#PBS -l select=<nodes>
#PBS -l place=scatter
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module load frameworks
module list

source /home/ht1410/.envs/SimAI-Bench/bin/activate

nnodes=$(cat $PBS_NODEFILE | wc -l)
timestamp=$(date +%s)
echo "Current timestamp: $timestamp"

run_experiments() {
    local config=$1
    shift 1
    local sizes=("$@")

    ndb=1
    for size in "${sizes[@]}"; do
        for exp in $(seq 0 0); do
            label2=$(printf "%.2fm_%sdb_exp%d" $(echo "$size / 1000000" | bc -l) "$ndb" "$exp")
            label1=$(basename "$config" .json)
            export WFMINI_LOG_DIR="logs_${label1}_${label2}"
            echo "Running size=$size, exp=$exp, config=$config" 
            backend=$(basename "$config" .json)

            if [ "$backend" == "dragon" ];then
                dragon workflow.py --server_config "$config" --data_size "$size" --staging_dir "/lus/flare/projects/datascience/hari/staging/.tmp_${nnodes}_${label2}_${timestamp}"
            else
                python3 workflow.py --server_config "$config" --data_size "$size" --staging_dir "/lus/flare/projects/datascience/hari/staging/.tmp_${nnodes}_${label2}_${timestamp}"
            fi
            # rm -r "/lus/flare/projects/datascience/hari/staging/.tmp_${nnodes}_$label2"
        done
    done
}

config=configs/server/filesystem.json
sizes=(10000 100000 500000 1000000 2000000 4000000 8000000 16000000 32000000 64000000)
if [ -z "$size_idx" ]; then
    run_experiments $config "${sizes[@]}"
else
    run_experiments $config "${sizes[${size_idx}]}"
fi