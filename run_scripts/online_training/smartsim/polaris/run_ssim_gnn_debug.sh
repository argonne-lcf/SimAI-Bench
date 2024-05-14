#!/bin/bash

# Set env
source /eagle/datascience/balin/SimAI-Bench/env_ssim.sh
echo Loaded modules:
module list

# Set executables
BASE_DIR=/eagle/datascience/balin/SimAI-Bench/SimAI-Bench
DRIVER=$BASE_DIR/src/online_training/drivers/ssim_driver.py
SIM_EXE=$BASE_DIR/src/online_training/data_producers/sim.py
ML_EXE=$BASE_DIR/src/online_training/train/train.py
DRIVER_CONFIG_PATH=$PWD/conf
DRIVER_CONFIG_NAME="ssim_config_colocated"
TRAIN_CONFIG_PATH=$PWD/conf
TRAIN_CONFIG_NAME="train_config_gnn_debug"

# Set up run
NODES=$(cat $PBS_NODEFILE | wc -l)
SIM_PROCS_PER_NODE=4
SIM_RANKS=$((NODES * SIM_PROCS_PER_NODE))
ML_PROCS_PER_NODE=4
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
SIM_ARGS="--backend\=smartredis --model\=gnn --problem_size\=debug --launch\=colocated --ppn\=${SIM_RANKS} --tolerance\=0.002"
python $DRIVER --config-path $DRIVER_CONFIG_PATH --config-name $DRIVER_CONFIG_NAME \
    database.deployment="colocated" \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config_path=${TRAIN_CONFIG_PATH} train.config_name=${TRAIN_CONFIG_NAME} \
    run_args.simprocs=${SIM_RANKS} run_args.simprocs_pn=${SIM_PROCS_PER_NODE} \
    run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_PROCS_PER_NODE} 

    
