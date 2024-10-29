#!/bin/bash

# Set env
module load PrgEnv-gnu miniforge3 rocm/6.1.3
source activate /lustre/orion/proj-shared/gen150/simai-bench/balin/env/_simai 
echo Loaded modules:
module list

# Set executables
BASE_DIR=/lustre/orion/proj-shared/gen150/simai-bench/balin/SimAI-Bench
DRIVER=$BASE_DIR/src/online_training/drivers/ssim_driver.py
SIM_EXE=$BASE_DIR/src/online_training/data_producers/sim.py
ML_EXE=$BASE_DIR/src/online_training/train/train.py
DRIVER_CONFIG_PATH=$PWD/conf
DRIVER_CONFIG_NAME="ssim_config_clustered"
TRAIN_CONFIG_PATH=$PWD/conf
TRAIN_CONFIG_NAME="train_config_gnn_small"

# Set up run
NODES=$(echo $SLURM_NNODES)
DB_NODES=1
SIM_NODES=1
ML_NODES=1
SIM_PROCS_PER_NODE=8
SIM_RANKS=$((SIM_NODES * SIM_PROCS_PER_NODE))
ML_PROCS_PER_NODE=8
ML_RANKS=$((ML_NODES * ML_PROCS_PER_NODE))
JOBID=$(echo $SLURM_JOB_ID)
echo Number of total nodes: $NODES
echo Number of database nodes: $DB_NODES
echo Number of simulation nodes: $SIM_NODES
echo Number of ML nodes: $ML_NODES
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

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MASTER_ADDR=$(hostname -i)
export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/sajal/software/aws-ofi-rccl/src/.libs/:${LD_LIBRARY_PATH}
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1


# Run
SIM_ARGS="--backend\=smartredis --model\=gnn --problem_size\=small --launch\=clustered --ppn\=${SIM_PROCS_PER_NODE} --tolerance\=0.004 --train_interval\=10 --db_max_mem_size\=0.1"
python $DRIVER --config-path $DRIVER_CONFIG_PATH --config-name $DRIVER_CONFIG_NAME \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config_path=${TRAIN_CONFIG_PATH} train.config_name=${TRAIN_CONFIG_NAME} \
    run_args.simprocs=${SIM_RANKS} run_args.simprocs_pn=${SIM_PROCS_PER_NODE} \
    run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_PROCS_PER_NODE} 

    
