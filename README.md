# CBAM-TensorFlow-Slim
This is a Tensorflow implementation of ["CBAM: Convolutional Block Attention Module"](https://arxiv.org/pdf/1807.06521) aiming to be compatible on the [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim).
This repository includes the implementation of ["SENet-tensorflow-slim"](https://github.com/kobiso/SENet-tensorflow-slim).

If you want to use simpler implementation, check the repository **"[CBAM-TensorFlow](https://github.com/kobiso/CBAM-tensorflow)"** which includes simple tensorflow implementation of [*ResNext*](https://arxiv.org/abs/1611.05431), [*Inception-V4*, and *Inception-ResNet-V2*](https://arxiv.org/abs/1602.07261) on Cifar10 dataset.

## CBAM: Convolutional Block Attention Module
**CBAM** proposes an architectural unit called *"Convolutional Block Attention Module" (CBAM)* block to improve representation power by using attention mechanism: focusing on important features and supressing unnecessary ones.
This research can be considered as a descendant and an improvement of ["Squeeze-and-Excitation Networks"](https://arxiv.org/pdf/1709.01507).

### Diagram of a CBAM_block
<div align="center">
  <img src="https://github.com/kobiso/CBAM-tensorflow-slim/blob/master/figures/overview.png">
</div>

### Diagram of each attention sub-module
<div align="center">
  <img src="https://github.com/kobiso/CBAM-tensorflow-slim/blob/master/figures/submodule.png">
</div>

### Classification results on ImageNet-1K

<div align="center">
  <img src="https://github.com/kobiso/CBAM-tensorflow-slim/blob/master/figures/exp4.png">
</div>

<div align="center">
  <img src="https://github.com/kobiso/CBAM-tensorflow-slim/blob/master/figures/exp5.png"  width="750">
</div>

## Prerequisites
- Python 3.x
- TensorFlow 1.x
- TF-slim
  - Check the ['installation' part of TF-Slim image models README](https://github.com/tensorflow/models/tree/master/research/slim#installation).

## Prepare Data set
You should prepare your own dataset or open dataset (Cifar10, flowers, MNIST, ImageNet).
For preparing dataset, you can follow the ['preparing the datasets' part in TF-Slim image models README](https://github.com/tensorflow/models/tree/master/research/slim#preparing-the-datasets).

## CBAM_block and SE_block Supportive Models
This project is based on TensorFlow-Slim image classification model library.
Every image classification model in TensorFlow-Slim can be run the same.
And, you can run **CBAM_block** or **SE_block** added models in the below list by adding one argument `--attention_module=cbam_block` or `--attention_module=se_block` when you train or evaluate a model.

- Inception V4 + CBAM / + SE
- Inception-ResNet-v2 + CBAM / + SE
- ResNet V1 50 + CBAM / + SE
- ResNet V1 101 + CBAM / + SE
- ResNet V1 152 + CBAM / + SE
- ResNet V1 200 + CBAM / + SE
- ResNet V2 50 + CBAM / + SE
- ResNet V2 101 + CBAM / + SE
- ResNet V2 152 + CBAM / + SE
- ResNet V2 200 + CBAM / + SE

### Change *Reduction ratio*
To change *reduction ratio*, you have to manually set the ratio on `def cbam_block(input_feature, name, ratio=16)` method for cbam_block or `def se_block(residual, name, ratio=8)` method for se_block in `CBAM-tensorflow-slim/nets/attention_module.py`.

## Train a Model
You can find example of training script in `CBAM-tensorflow-slim/scripts/`.

### Train a model with CBAM_block
Below script gives you an example of training a model with CBAM_block.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=cbam_block
```

### Train a model with SE_block
Below script gives you an example of training a model with SE_block.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=se_block
```


### Train a model without attention module
Below script gives you an example of training a model without attention module.
```
DATASET_DIR=/DIRECTORY/TO/DATASET
TRAIN_DIR=/DIRECTORY/TO/TRAIN
CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_50 \
    --batch_size=100
```

## Evaluate a Model
You can find example of evaluation script in `CBAM-tensorflow-slim/scripts/`.
To keep track of validation accuracy while training, you can use `eval_image_classifier_loop.py` which evaluate the performance at multiple checkpoints during training.
If you want to just evaluate a model once, you can use `eval_image_classifier.py`.

### Evaluate a model with CBAM_block
Below script gives you an example of evaluating a model with CBAM_block during training.


```
DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=cbam_block
```

### Evaluate a model with SE-block
Below script gives you an example of evaluating a model with SE_block during training.

```
DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 \
    --attention_module=se_block
```

### Evaluate a model without attention module
Below script gives you an example of evaluating a model without attention module during training.

```
DATASET_DIR=/DIRECTORY/TO/DATASET
CHECKPOINT_FILE=/DIRECTORY/TO/CHECKPOINT
EVAL_DIR=/DIRECTORY/TO/EVAL
CUDA_VISIBLE_DEVICES=0 python eval_image_classifier_loop.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50 \
    --batch_size=100 
```

## Related Works
- Blog: [CBAM: Convolutional Block Attention Module](https://kobiso.github.io//research/research-CBAM/)
- Repository: [CBAM-TensorFlow](https://github.com/kobiso/CBAM-tensorflow)
- Repository: [CBAM-Keras](https://github.com/kobiso/CBAM-keras)
- Repository: [SENet-TensorFlow-Slim](https://github.com/kobiso/SENet-tensorflow-slim)

## Reference
- Paper: [CBAM: Convolutional Block Attention Module](https://arxiv.org/pdf/1807.06521)
- Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
- Repository: [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)
  
## Author
Byung Soo Ko / kobiso62@gmail.com
