#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

LOGNAME=$1
GPU=$2
NUM_GPU=$3

DEVICES=()

if (($NUM_GPU > 1))
    then
        for i in `seq $GPU $(($GPU + $NUM_GPU - 1))`
        do
            DEVICES+=("$i")
        done
fi

printf -v DEVICE_STRING "%s," "${DEVICES[@]}"
DEVICE_STRING=${DEVICE_STRING%?}

mkdir Output/${LOGNAME}
CUDA_VISIBLE_DEVICES="$DEVICE_STRING" python -u src/__main__.py --config Configs/${LOGNAME}.yml \
    --gpu $GPU > Output/${LOGNAME}/${LOGNAME}.out 2> Output/${LOGNAME}/${LOGNAME}.err
