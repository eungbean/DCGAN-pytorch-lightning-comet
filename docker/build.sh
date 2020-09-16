#!/bin/bash
PROJ_DIR=`dirname $(cd $(dirname $0); pwd)`
source ${PROJ_DIR}/docker/settings.sh

# build docker image from project root directory
cd ${PROJ_DIR} && \
docker build -t ${IMAGE} -f ${PROJ_DIR}/docker/Dockerfile.dev .