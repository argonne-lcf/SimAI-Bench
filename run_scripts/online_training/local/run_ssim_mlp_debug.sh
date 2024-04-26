#!/bin/bash

# Set executables
BASE_DIR=/Users/rbalin/Documents/Research/ALCF/SimAI-Bench/SimAI-Bench
DRIVER=$BASE_DIR/src/online_training/drivers/ssim_driver.py
SIM_EXE=$BASE_DIR/src/online_training/data_producers/sim.py
ML_EXE=$BASE_DIR/src/online_training/train/train.py
TRAIN_CONFIG_PATH=$PWD/conf
TRAIN_CONFIG_NAME="train_config_mlp_debug"

# Set up run
SIM_RANKS=4
ML_RANKS=2
echo Number of simulation ranks: $SIM_RANKS
echo Number of ML ranks: $ML_RANKS
echo

# Run
SIM_ARGS="--model\=mlp --problem_size\=debug --db_launch\=colocated --ppn\=${SIM_RANKS} --tolerance\=0.002"
python $DRIVER \
    database.network_interface=lo database.launcher=local \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config_path=${TRAIN_CONFIG_PATH} train.config_name=${TRAIN_CONFIG_NAME} \
    run_args.simprocs=${SIM_RANKS}  run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_RANKS} 

    
