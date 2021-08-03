# A Pytorch Implementation of Unbiased Mean Teacher for Cross-domain Object Detection (CVPR 2021)

## Introduction
Follow the implementation of [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) to set up the environment. In our implementation, we use Pytorch 0.4.0 on a single GeForce GTX 1080 Ti.

## Environment Preparation

## Data Preparation

Please follow the instructions in [DA_detection](https://github.com/VisionLearningGroup/DA_Detection) to prepare **PASCAL_VOC 07+12**, **Clipart1k**, **WaterColor2k**, and **SIM10K**. We use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pi) to generate the source/target-like images.

All the data arrangements follow the format of PASCAL_VOC. Our dataset config system also follow the [DA_detection](https://github.com/VisionLearningGroup/DA_Detection). 

## Train

```
 CUDA_VISIBLE_DEVICES=$GPU_ID python umt_train.py \
                    --dataset {SOURCE DATASET} --dataset_t {Target DATASET} --net {vgg16 or res101}
```

Taking clipart as an example:

```
 CUDA_VISIBLE_DEVICES=$GPU_ID python umt_train.py \
                    --dataset pascal_voc_07_12 --dataset_t clipart --net res101
```
## Test

```shell
./test.sh {GUP_ID} {MODEL_PATH}
```

## Citation

Please cite the following reference if you utilize this repository for your project.

``` text
@inproceedings{deng2021unbiased,
  title={Unbiased Mean Teacher for Cross-Domain Object Detection},
  author={Deng, Jinhong and Li, Wen and Chen, Yuhua and Duan, Lixin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4091--4101},
  year={2021}
}
```
