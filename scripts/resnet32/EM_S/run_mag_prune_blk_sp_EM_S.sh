# magnitude-based 1 shot retraining
ARCH=${1:-"resnet"} # 
DEPTH=${2:-"32"}
PRUNE_ARGS=${5:-"--sp-retrain --sp-prune-before-retrain"}
LOAD_CKPT=${6:-"XXXXX.pth.tar"}
INIT_LR=${8:-"0.1"}
EPOCHS=${11:-"160"}
WARMUP=${12:-"8"}

SPARSITY_TYPE=${3:-"free_block_prune_column_4"}
DATASET=${20:-"cifar10"}

GLOBAL_BATCH_SIZE=${9:-"64"}
MASK_UPDATE_DECAY_EPOCH=${18:-"90-120"}
SP_MASK_UPDATE_FREQ=${19:-"5"}

SAVE_FOLDER=${7:-"checkpoints/resnet32/MwE_blk/"}

cd ../../..

mkdir -p ${SAVE_FOLDER}

GPU_ID=${15:-"0"}
SEED=${10:-"914"}


# --------------------------------------------------------------------------

LOWER_BOUND=${16:-"0.9-0.9-0.9"} &&
UPPER_BOUND=${17:-"0.85-0.875-0.9"} &&

CONFIG_FILE=${4:-"./profiles/resnet32_cifar/blk/resnet32_0.9.yaml"} &&
REMARK=${13:-"blk_0.9_MwE"} &&
LOG_NAME=${14:-"blk_0.9_MwE"} &&

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u main_sparse_train.py \
        --arch ${ARCH} --depth ${DEPTH} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --dataset ${DATASET} --seed ${SEED} --upper-bound ${UPPER_BOUND} --lower-bound ${LOWER_BOUND} --mask-update-decay-epoch ${MASK_UPDATE_DECAY_EPOCH} --sp-mask-update-freq ${SP_MASK_UPDATE_FREQ} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} \
        --log-filename=${SAVE_FOLDER}/seed_${SEED}_${LOG_NAME}.txt

# --------------------------------------------------------------------------

#LOWER_BOUND=${16:-"0.95-0.95-0.95"} &&
#UPPER_BOUND=${17:-"0.90-0.925-0.95"} &&
#
#CONFIG_FILE=${4:-"./profiles/resnet32_cifar/blk/resnet32_0.95.yaml"} &&
#REMARK=${13:-"blk_0.95_MwE"} &&
#LOG_NAME=${14:-"blk_0.95_MwE"} &&
#
#CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u main_sparse_train.py \
#        --arch ${ARCH} --depth ${DEPTH} --optmzr sgd --batch-size ${GLOBAL_BATCH_SIZE} --lr ${INIT_LR} --lr-scheduler cosine --save-model ${SAVE_FOLDER} --epochs ${EPOCHS} --dataset ${DATASET} --seed ${SEED} --upper-bound ${UPPER_BOUND} --lower-bound ${LOWER_BOUND} --mask-update-decay-epoch ${MASK_UPDATE_DECAY_EPOCH} --sp-mask-update-freq ${SP_MASK_UPDATE_FREQ} --remark ${REMARK} ${PRUNE_ARGS} --sp-admm-sparsity-type=${SPARSITY_TYPE} --sp-config-file=${CONFIG_FILE} \
#        --log-filename=${SAVE_FOLDER}/seed_${SEED}_${LOG_NAME}.txt