## Setup Docker (Linux / Mac)
### Configure your custom variables
1. ```./docker/Dockerfile.dev``` 파일을 열어 SSH 비밀번호를 설정해줍니다.

```dockerfile
ENV sshpasswd SETYOURPASSWORD
```

2. ```./docker/settings.sh ``` 파일을 열어 데이터셋의 경로 등을 설정해줍니다.

```shell
#================================================
#Configure Path
DATASET_DIR=${PROJ_DIR}/dataset # Your dataset path
OUTPUT_DIR=${PROJ_DIR}/output # Your output path : weight, log, checkpoint, predictions..

#Configure Settings
IMAGE="100daysGAN" 		# Image name
CONTAINER="100daysGAN" 	# Container name
GPUS="all"		# "all", "0,1,2..", "none"
JUPYTER_PORT_H=18888 	# jupyter port
DEFAULT_PORT_H=10022 	# SSH port
TB_PORT_H = 16006 		# TensorBoard port
#================================================
```

### Setup Docker (Linux / Mac)
```sh
# ensure you are in 100-DAYS-OF-GAN directory
# Build the docker image
./docker/build.sh

# Run the docker container
./docker/run.sh

# Attatch into contaier
./docker/exec.sh
```

## Preparing Data
### Create symbolic link to your Dataset
Dataset이 저장되어있는 폴더와 `./dataset/`폴더에 심볼릭 링크를 만들어줍니다.
```sh
ln -s //PATH//TO//YOUR//DATASET// dataset
```

## Train and Test your Model

### Train
```sh
(in container) python train.sh dcgan
```

### Test
```sh
(in container) python test.sh dcgan
```

##