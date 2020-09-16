#!/bin/bash
PROJ_DIR=`dirname $(cd $(dirname $0); pwd)`
source ${PROJ_DIR}/docker/settings.sh

echo $PROJ_DIR
echo "mapping SSH port Host:${DEFAULT_PORT_H} --> Container:22"
echo "mapping Jupyter port Host:${JUPYTER_PORT_H} --> Container:8888"
echo "mapping TensorBoard port Host:${TB_PORT_H} --> Container:6006"

# run container
docker run -d -P -it --ipc=host \
	--gpus=${GPUS} \
	-p ${DEFAULT_PORT_H}:22 \
	-p ${JUPYTER_PORT_H}:8888 \
	-p ${TB_PORT_H}:6006 \
	-v ${PROJ_DIR}:/code \
	-v ${DATASET_DIR_H}:/code/dataset \
	-v ${OUTPUT_DIR_H}:/code/output \
	--name ${CONTAINER} \
	${IMAGE} 
	
	#/usr/bin/zsh
	
	#TODO cleanup
	#/usr/bin/zsh