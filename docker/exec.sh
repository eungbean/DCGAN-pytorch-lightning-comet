#!/bin/bash
PROJ_DIR=`dirname $(cd $(dirname $0); pwd)`
source ${PROJ_DIR}/docker/settings.sh

# exec
docker start ${CONTAINER}
docker exec -it ${CONTAINER} /usr/bin/zsh

#TODO cleanup
#/usr/bin/zsh