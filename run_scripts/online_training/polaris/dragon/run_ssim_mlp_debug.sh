#!/bin/bash

# Set env
#source /eagle/datascience/balin/SimAI-Bench/env_dragon.sh
echo Loaded modules:
module list

# Set executables
BASE_DIR=/eagle/datascience/balin/SimAI-Bench/SimAI-Bench
DRIVER=$BASE_DIR/src/online_training/drivers/dragon_driver.py
SIM_EXE=$BASE_DIR/src/online_training/data_producers/sim.py
ML_EXE=$BASE_DIR/src/online_training/train/train.py
DRIVER_CONFIG_PATH=$PWD/conf
TRAIN_CONFIG_PATH=$PWD/conf
TRAIN_CONFIG_NAME="train_config_mlp_debug"

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

# Run
SIM_ARGS="--backend\=dragon --model\=mlp --problem_size\=debug --launch\=colocated --ppn\=${SIM_RANKS} --tolerance\=0.002"
dragon $DRIVER --config-path $DRIVER_CONFIG_PATH \
    deployment="colocated" \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    sim.procs=${SIM_RANKS} sim.procs_pn=${SIM_PROCS_PER_NODE} \
    train.executable=$ML_EXE train.config_path=${TRAIN_CONFIG_PATH} train.config_name=${TRAIN_CONFIG_NAME} \
    train.procs=${ML_RANKS} train.procs_pn=${ML_PROCS_PER_NODE} 

    
