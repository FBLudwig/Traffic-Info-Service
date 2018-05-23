#! /bin/bash


export PYTHONUNBUFFERED="True"

# Parameters
GPU_DEV=0
CONFIG_FILE=models/trancos/ccnn/ccnn_trancos_cfg.yml
CAFFE_MODEL=models/pretrained_models/trancos/ccnn/trancos_ccnn.caffemodel
DEPLOY=models/trancos/ccnn/ccnn_deploy.prototxt

LOG="logs/a5_ccnn_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# Time the task
T="$(date +%s)"

# Run with GPU
python3 src/server.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}
# Run with CPU
#python3 src/server.py --cpu_only --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}

# For quick testing of ccnn
#python3 src/ccnn.py --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}
#python3 src/ccnn.py --cpu_only --dev ${GPU_DEV} --prototxt ${DEPLOY} --caffemodel ${CAFFE_MODEL} --cfg ${CONFIG_FILE}

T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"
