# Faceshopping: Enhancing Pose-Conditioned Human Image Generation with GANs

This is Pytorch implementation for pose transfer on both Market1501 and DeepFashion dataset. The code is written by [Tengteng Huang](https://github.com/tengteng95) and [Zhen Zhu](https://github.com/jessemelpolio). We build on this code by alterting the perceptual loss funtion, neural network architecture by using ELU activation and more PAT blocks and also implement LayerNorm2d.

## Requirement
* pytorch 1.0.1
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate


## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/tengteng95/Pose-Transfer.git
cd Pose-Transfer
```

### Data Preperation

We use [OpenPose](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) to generate keypoints. We also provide our extracted keypoints files for convience.

#### DeepFashion
<!-- - Download the DeepFashion dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) -->
- Download the DeepFashion dataset from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg). Unzip ```train.zip``` and ```test.zip``` into the ```fashion_data``` directory.
- Download train/test splits and train/test key points annotations from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg), including **fasion-resize-pairs-train.csv**, **fasion-resize-pairs-test.csv**, **fasion-resize-annotation-train.csv**, **fasion-resize-annotation-train.csv**. Put these four files under the ```fashion_data``` directory.
- Launch ```python tool/generate_pose_map_fashion.py``` to generate the pose heatmaps.

<!-- #### Pose Estimation
- Download the pose estimator from [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).
- Launch ```python compute_cordinates.py``` to get the pose estimation for both datasets.

OR you can download our generated pose estimations from here. (Coming soon.) --> 

### Train a model

DeepFashion
```bash
python train.py --dataroot ./fashion_data/ --name fashion_PATN --model PATN --lambda_GAN 5 --lambda_A 1 --lambda_B 1 --dataset_mode keypoint --n_layers 3 --norm instance --batchSize 7 --pool_size 0 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --niter 500 --niter_decay 200 --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-train.csv --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 0
```


### Test the model

DeepFashion
```bash
python test.py --dataroot ./fashion_data/ --name fashion_PATN --model PATN --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PATN --checkpoints_dir ./checkpoints --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch latest --results_dir ./results
```

### Evaluation
We adopt SSIM, mask-SSIM, IS, mask-IS, DS, and PCKh for evaluation of Market-1501. SSIM, IS, DS, PCKh for DeepFashion.

#### 1) SSIM and mask-SSIM, IS and mask-IS, mask-SSIM

For evaluation, **Tensorflow 1.4.1(python3)** is required. Please see ``requirements_tf.txt`` for details.

For DeepFashion:
```bash
python tool/getMetrics_fashion.py
```

If you still have problems for evaluation, please consider using **docker**. 

```bash
docker run -v <Pose-Transfer path>:/tmp -w /tmp --runtime=nvidia -it --rm tensorflow/tensorflow:1.4.1-gpu-py3 bash
# now in docker:
$ pip install scikit-image tqdm 
$ python tool/getMetrics_market.py
```

Refer to [this Issue](https://github.com/tengteng95/Pose-Transfer/issues/4).

#### 2) DS Score
Download pretrained on VOC 300x300 model and install propper caffe version [SSD](https://github.com/weiliu89/caffe/tree/ssd). Put it in the ssd_score forlder. 

For DeepFashion:
```bash
python compute_ssd_score_fashion.py --input_dir path/to/generated/images
```

#### 3) PCKh
- First, run ``tool/crop_market.py`` or ``tool/crop_fashion.py``.
- Download pose estimator from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg). Put it under the root folder ``Pose-Transfer``.
- Change the paths **input_folder**  and **output_path** in ``tool/compute_coordinates.py``. And then launch
```bash
python2 compute_coordinates.py
```
- run ``tool/calPCKH_fashion.py`` 



### Pre-trained model 
Coming Soon. We shall upload the model that we trained.

##### Notes:
In pytorch 1.0, **running_mean** and **running_var** are not saved for the **Instance Normalization layer** by default. To reproduce our result in the paper, launch ``python tool/rm_insnorm_running_vars.py`` to remove corresponding keys in the pretrained model. (Only for the DeepFashion dataset.)

### Acknowledgments
Our code is based on the popular [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
