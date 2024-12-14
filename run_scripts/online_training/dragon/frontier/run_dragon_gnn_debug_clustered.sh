#!/bin/bash

# Set env
module load PrgEnv-gnu miniforge3 rocm/6.1.3
source activate /lustre/orion/proj-shared/gen150/simai-bench/balin/env/_simai
source /lustre/orion/proj-shared/gen150/simai-bench/balin/env/dragon-0.10/_dragon/bin/activate
module use /lustre/orion/proj-shared/gen150/simai-bench/balin/env/dragon-0.10/modulefiles
module load dragon
echo Loaded modules:
module list

# Set executables
BASE_DIR=/lustre/orion/proj-shared/gen150/simai-bench/balin/SimAI-Bench
DRIVER=$BASE_DIR/src/online_training/drivers/dragon_driver.py
SIM_EXE=$BASE_DIR/src/online_training/data_producers/sim.py
ML_EXE=$BASE_DIR/src/online_training/train/train.py
DRIVER_CONFIG_PATH=$PWD/conf
DRIVER_CONFIG_NAME="dragon_config_gnn"
TRAIN_CONFIG_PATH=$PWD/conf
TRAIN_CONFIG_NAME="train_config_gnn_debug"

# Set up run
NODES=$(echo $SLURM_NNODES)
DICT_NODES=1
SIM_NODES=1
ML_NODES=1
SIM_PROCS_PER_NODE=8
SIM_RANKS=$((SIM_NODES * SIM_PROCS_PER_NODE))
ML_PROCS_PER_NODE=8
ML_RANKS=$((ML_NODES * ML_PROCS_PER_NODE))
JOBID=$(echo $SLURM_JOB_ID)
echo Number of total nodes: $NODES
echo Number of dictionary nodes: $DICT_NODES
echo Number of simulation nodes: $SIM_NODES
echo Number of ML training nodes: $ML_NODES
echo Number of simulation ranks per node: $SIM_PROCS_PER_NODE
echo Number of simulation total ranks: $SIM_RANKS
echo Number of ML ranks per node: $ML_PROCS_PER_NODE
echo Number of ML total ranks: $ML_RANKS
echo

# Sent env vars
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

export MPICH_OFI_NIC_POLICY=BLOCK # Needed to avoid MPICH ERROR: Unable to use a NIC_POLICY of 'NUMA'

# Run
SIM_ARGS="--backend\=dragon --model\=gnn --problem_size\=debug --launch\=clustered --ppn\=${SIM_PROCS_PER_NODE} --tolerance\=0.002"
dragon $DRIVER --config-path $DRIVER_CONFIG_PATH --config-name $DRIVER_CONFIG_NAME \
    deployment="clustered" \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    dict.num_nodes=$DICT_NODES sim.num_nodes=$SIM_NODES train.num_nodes=$ML_NODES \
    sim.procs=${SIM_RANKS} sim.procs_pn=${SIM_PROCS_PER_NODE} \
    train.executable=$ML_EXE train.config_path=${TRAIN_CONFIG_PATH} train.config_name=${TRAIN_CONFIG_NAME} \
    train.procs=${ML_RANKS} train.procs_pn=${ML_PROCS_PER_NODE} 

    
