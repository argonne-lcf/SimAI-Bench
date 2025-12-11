#!/bin/bash

dir=$(pwd)
export DAOS_POOL="datascience"
export DAOS_CONT="test_daos"

fusermount3 -u /tmp/${USER}/daos_mount # To unmount - very important to clean up afterward on the UAN

daos cont destroy ${DAOS_POOL} ${DAOS_CONT}
