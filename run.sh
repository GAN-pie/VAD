#!/bin/bash

# Adrien Gresse 2024

# The script help to manage the training runs

set -e

ROOT="${HOME}/fromscratch_vad"

VAD_DATA_FOLDER="${ROOT}/../vad_data"
METADATA_FOLDER="${ROOT}/metadata"

SPLIT_RATIO=0.3
ISOLATE_SPK=1

SKIP_DATA_PREP=1

if [ ${ISOLATE_SPK} -eq 1 ]; then
    ISOLATE_SPK_FLAG="--isolate-speaker"
else
    ISOLATE_SPK_FLAG=""
fi

if [ ${SKIP_DATA_PREP} -eq 0 ]; then
    python3 ./prepare_vad_data.py ${VAD_DATA_FOLDER} ${METADATA_FOLDER} \
        --eval-ratio ${SPLIT_RATIO} \
        ${ISOLATE_SPK_FLAG}
fi

# Statistics can be computed with the utils.py module once data preparation
# have been executed.
MEAN_NORM=-5.474683
VAR_NORM=3.6145234
# MEAN_NORM=-0.96911
# VAR_NORM=1.3280091

TARGET_SIZE=1536
NUM_MEL=40

BATCH_SIZE=2
MAX_EPOCH=100
LR_INIT=1.0

EXP_FOLDER="${ROOT}/runs/base-1d-t${TARGET_SIZE}-b${BATCH_SIZE}-e${MAX_EPOCH}-lr${LR_INIT}"
mkdir -p ${EXP_FOLDER}

python3 ./traintest.py ${METADATA_FOLDER} ${EXP_FOLDER} \
    --batch-size ${BATCH_SIZE} --max-epochs ${MAX_EPOCH} \
    --lr-init ${LR_INIT} \
    --norm-mean ${MEAN_NORM} --norm-var ${VAR_NORM} \
    --target-size ${TARGET_SIZE} \
    --num-mel-bins ${NUM_MEL} \
    --display-charts \
    --use-energy
