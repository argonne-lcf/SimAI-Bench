#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

db_stdout=$(/lustre/orion/proj-shared/gen150/simai-bench/balin/env/_simai/bin/python -m smartsim._core.entrypoints.colocated +lockfile smartsim-5c71409.lock +db_cpus 1 +ifname udf +command taskset -c 28,29,30,31 /lustre/orion/gen150/proj-shared/simai-bench/balin/env/_simai/lib/python3.11/site-packages/smartsim/_core/bin/redis-server /lustre/orion/gen150/proj-shared/simai-bench/balin/env/_simai/lib/python3.11/site-packages/smartsim/_core/config/redis.conf --loadmodule /lustre/orion/gen150/proj-shared/simai-bench/balin/env/_simai/lib/python3.11/site-packages/smartsim/_core/lib/redisai.so THREADS_PER_QUEUE 4 INTER_OP_PARALLELISM 1 INTRA_OP_PARALLELISM 1 --port 6780 --logfile /dev/null --maxclients 100000 --cluster-node-timeout 30000)
DBPID=$(echo $db_stdout | sed -n 's/.*__PID__\([0-9]*\)__PID__.*/\1/p')
$@

