#!/bin/bash
#PBS -A datascience
#PBS -q debug
#PBS -N redis_simulation
#PBS -l walltime=01:00:00
#PBS -l select=2
#PBS -l place=scatter
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR
mkdir run_dir
export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module load frameworks
module list

source /home/ht1410/.envs/wfminiapps/bin/activate


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
                dragon workflow.py --server_config "$config" --data_size "$size"
            else
                python3 workflow.py --server_config "$config" --data_size "$size"
            fi
        done
    done
}

config=configs/server/redis.json
sizes=(319488)
if [ -z "$size_idx" ]; then
    run_experiments $config "${sizes[@]}"
else
    run_experiments $config "${sizes[${size_idx}]}"
fi