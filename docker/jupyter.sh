#!/bin/bash
PROJ_DIR=`dirname $(cd $(dirname $0); pwd)`
source ${PROJ_DIR}/docker/settings.sh

# launch jupyter from container
echo "access jupyter sever"
echo "Jupyter port Host:${JUPYTER_PORT_H} --> Container:8888"

# build docker image from project root directory
cd ${PROJ_DIR} && \
jupyter lab --port ${JUPYTER_PORT} --ip=0.0.0.0 --allow-root --no-browser