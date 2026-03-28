#!/bin/bash

BASE_EBS_DIR=${BASE_EBS_DIR}
MODEL_BUCKET=${MODEL_BUCKET}
OUTPUT_BUCKET=${OUTPUT_BUCKET}
OUTPUT_PATH=${OUTPUT_PATH}

# create fs and mount the nvme ebs volume
DEVICE=/dev/sda1

if ! mountpoint -q $BASE_EBS_DIR; then
    sudo mkdir -p $BASE_EBS_DIR
    sudo mkfs -t xfs /dev/sda1
    sudo mount $DEVICE $BASE_EBS_DIR
    sudo chown -R $USER:$USER $BASE_EBS_DIR
    echo "EBS volume mounted at $BASE_EBS_DIR"
fi

# install mount-s3
wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.rpm
sudo yum install -y ./mount-s3.rpm

# mount the s3 buckets in the filesystem
mkdir -p $BASE_EBS_DIR/model
mount-s3 $MODEL_BUCKET $BASE_EBS_DIR/model

# create dirs
mkdir -p $OUTPUT_PATH