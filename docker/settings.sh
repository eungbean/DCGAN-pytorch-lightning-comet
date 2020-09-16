#!/bin/bash

# #DEFAULT CONFIGURATION
# #================================================
# #Configure Path
# DATASET_DIR=${PROJ_DIR}/dataset # Your dataset path
# OUTPUT_DIR=${PROJ_DIR}/output # Your output path : weight, log, checkpoint, predictions..

# #Configure Settings
# IMAGE="100daysgan" 		# Image name
# CONTAINER="100daysgan" 	# Container name
# GPUS="all"	    # "all", "0,1,2..", "none"
# JUPYTER_PORT_H=18888 	# jupyter port
# DEFAULT_PORT_H=10022 	# SSH port
# TB_PORT_H=16006 		# TensorBoard port
# #================================================

#CUSTOM CONFIGURATION FOR EUNGBEAN
#================================================
#Configure Path
DATASET_DIR=${PROJ_DIR}/dataset # Your dataset path
OUTPUT_DIR=${PROJ_DIR}/output # Your output path : weight, log, checkpoint, predictions..

#Configure Settings
IMAGE="DCGAN_eungbean" 		# Image name
CONTAINER="DCGAN_eungbean" 	# Container name
GPUS="all"    # "all", "0,1,2..", "none"
JUPYTER_PORT_H=18888 	# jupyter port
DEFAULT_PORT_H=10022 	# SSH port
TB_PORT_H=16006 		# TensorBoard port
#================================================