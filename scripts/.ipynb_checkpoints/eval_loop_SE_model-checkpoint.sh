DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python ../eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=se_block