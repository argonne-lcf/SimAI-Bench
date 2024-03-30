#!/bin/bash

# Set executables
BASE_DIR=/Users/rbalin/Documents/Research/ALCF/SimAI-Bench/SimAI-Bench
DRIVER=$BASE_DIR/online_training/src/backends/smartsim/ssim_driver.py
SIM_EXE=$BASE_DIR/online_training/src/data_producers/smartredis/load_data.py
ML_EXE=$BASE_DIR/online_training/src/train/main.py
TRAIN_CONFIG=$PWD

# Set up run
SIM_RANKS=4
ML_RANKS=2
echo Number of simulation ranks: $SIM_RANKS
echo Number of ML ranks: $ML_RANKS
echo

# Run
SIM_ARGS="--model\=mlp --problem_size\=small --db_launch\=colocated --ppn\=${SIM_RANKS} --reproducibility\=True"
python $DRIVER \
    database.network_interface=lo database.launcher=local \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config=${TRAIN_CONFIG} \
    run_args.simprocs=${SIM_RANKS}  run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_RANKS} 

    