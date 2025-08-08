#!/bin/bash

for node in 2;do
    cd ${node}nodes
    # rm -r *
    mkdir outputs
    # # ####
    ln -s ../../configs configs
    ln -s ../../workflow.py workflow.py
    ln -s ../../sim_exec.py sim_exec.py
    ln -s ../../train_ai_exec.py train_ai_exec.py
    #
    # Define array of job script names
    job_scripts=("job_script_dragon.sh" "job_script_filesystem.sh" "job_script_redis.sh" "job_script_nodelocal.sh")

    # Loop through each job script
    for script in "${job_scripts[@]}"; do
        cp "../${script}" "${script}"
        sed -i "s|#PBS -l select=<nodes>|#PBS -l select=${node}|g" "${script}"
        # qsub "${script}"
    done
    ##
    cd ../
    echo "Created job script for ${node} nodes"
done