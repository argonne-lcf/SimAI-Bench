#!/bin/bash

dir=$(pwd)
export DAOS_POOL="datascience"
export DAOS_CONT="test_daos"

module use /soft/modulefiles
module load daos
daos container create --type=POSIX  --chunk-size=2097152  --properties=rd_fac:2,ec_cell_sz:131072,cksum:crc32,srv_cksum:on ${DAOS_POOL} ${DAOS_CONT}
daos pool query ${DAOS_POOL}
daos cont list ${DAOS_POOL}
daos container get-prop  $DAOS_POOL $DAOS_CONT


mkdir -p /tmp/${USER}/daos_mount
start-dfuse.sh -m /tmp/${USER}/daos_mount --pool ${DAOS_POOL} --cont ${DAOS_CONT} # To mount
mount | grep dfuse # To confirm if its mounted

echo hello > ~/test.txt
# Mode 1
ls /tmp/${USER}/daos_mount
cd /tmp/${USER}/daos_mount
cp ~/test.txt ./
cat test.txt

cd $dir
