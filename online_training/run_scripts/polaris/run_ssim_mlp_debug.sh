#!/bin/bash

# Set env
module load conda/2023-10-04
conda activate
source /eagle/datascience/balin/Polaris/SmartSim_envs/venv_conda-2023-10-04/_ssim_env_24_4/bin/activate
echo Loaded modules:
module list

# Set executables
BASE_DIR=/eagle/datascience/balin/SimAI-Bench/SimAI-Bench
DRIVER=$BASE_DIR/online_training/src/backends/smartsim/ssim_driver.py
SIM_EXE=$BASE_DIR/online_training/src/data_producers/smartredis/load_data.py
ML_EXE=$BASE_DIR/online_training/src/train/main.py
TRAIN_CONFIG=$PWD

# Set up run
NODES=$(cat $PBS_NODEFILE | wc -l)
SIM_PROCS_PER_NODE=4
SIM_RANKS=$((NODES * SIM_PROCS_PER_NODE))
ML_PROCS_PER_NODE=2
ML_RANKS=$((NODES * ML_PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of simulation ranks per node: $SIM_PROCS_PER_NODE
echo Number of simulation total ranks: $SIM_RANKS
echo Number of ML ranks per node: $ML_PROCS_PER_NODE
echo Number of ML total ranks: $ML_RANKS
echo

# Sent env vars
#export SR_LOG_FILE=stdout
export SR_LOG_LEVEL=QUIET
export SR_CONN_INTERVAL=10 # default is 1000 ms
export SR_CONN_TIMEOUT=1000 # default is 100 ms
export SR_CMD_INTERVAL=10 # default is 1000 ms
export SR_CMD_TIMEOUT=1000 # default is 100 ms
export SR_THREAD_COUNT=4 # default is 4

# Run
SIM_ARGS="--model\=mlp --problem_size\=small --db_launch\=colocated --ppn\=${SIM_RANKS}  --reproducibility\=True --tolerance\=0.001"
python $DRIVER --config-path $PWD \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config=${TRAIN_CONFIG} \
    run_args.simprocs=${SIM_RANKS} run_args.simprocs_pn=${SIM_PROCS_PER_NODE} \
    run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_PROCS_PER_NODE} 

    
