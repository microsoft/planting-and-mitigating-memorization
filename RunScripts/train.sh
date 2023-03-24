# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

LOGNAME=$1
GPU=$2

mkdir Output/${LOGNAME}
python -u src/__main__.py --config Configs/${LOGNAME}.yml \
    --gpu $GPU > Output/${LOGNAME}/${LOGNAME}.out 2> Output/${LOGNAME}/${LOGNAME}.err
