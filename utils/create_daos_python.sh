#!/bin/bash

dir=$(pwd)
export DAOS_POOL="datascience"
export DAOS_CONT="test_daos"

module use /soft/modulefiles
module load daos
module load daos_perf
module load frameworks

export LD_LIBRARY_PATH=/lus/flare/projects/datasets/softwares/py_daos/daos_client_master_build_may2/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/lus/flare/projects/datasets/softwares/py_daos/just_pydaos_new/:$PYTHONPATH

daos container create --type=PYTHON  --chunk-size=2097152  --properties=rd_fac:2,ec_cell_sz:131072,cksum:crc32,srv_cksum:on ${DAOS_POOL} ${DAOS_CONT}
daos pool query ${DAOS_POOL}
daos cont list ${DAOS_POOL}
# daos container get-prop  $DAOS_POOL $DAOS_CONT

