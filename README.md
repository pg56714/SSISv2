# SSISv2: Instance Shadow Detection with A Single-Stage Detector

Here is only the SSISv2 file.

SSIS (CVPR 2021 **Oral**) & SSISv2 (**TPAMI** 2022)

[Tianyu Wang](https://stevewongv.github.io), [Xiaowei Hu](https://xw-hu.github.io), Pheng-Ann Heng, [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/)

**Instance Shadow Detection** aims to find shadow instances, object instances and shadow-object associations; this task benefits many vision applications, such as light direction estimation and photo editing.

To approach this task, we first compile a new dataset with the masks for shadow instances, object instances, and shadow-object associations. We then design an evaluation metric for quantitative evaluation of the performance of instance shadow detection. Further, we design a single-stage detector to perform instance shadow detection in an end-to-end manner, where the bidirectional relation learning module and the deformable maskIoU head are proposed in the detector to directly learn the relation between shadow instances and object instances and to improve the accuracy of the predicted masks.

[[📄 TPAMI - SSISv2](http://arxiv.org/abs/2207.04614)] [[📄 CVPR - SSIS](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Single-Stage_Instance_Shadow_Detection_With_Bidirectional_Relation_Learning_CVPR_2021_paper.pdf)] [[👇🏼 Video](http://www.youtube.com/watch?v=p0b_2SsFypw)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y9UpS5uA1YuoMyvYVzcKL4ltA_FDu_x0?usp=sharing)

[![YouTube](https://cdn.jsdelivr.net/gh/stevewongv/image-hosting@master/20210618/CVPR2021.273zljpaxzpc.jpg)](http://www.youtube.com/watch?v=p0b_2SsFypw)

Note that we tested on CUDA10.2 / PyTorch 1.6.0, CUDA11.1-11.5 / PyTorch 1.8.0 - 1.11.0 and [Colab](https://colab.research.google.com/drive/1y9UpS5uA1YuoMyvYVzcKL4ltA_FDu_x0?usp=sharing).

<img width="1024" src="./examples/network.png">

## Dataset and pre-trained model

### Performance on SOBA-testing set

|                            Method                             | SOAP mask | SOAP bbox | mask AP | box AP |
| :-----------------------------------------------------------: | :-------: | :-------: | :-----: | :----: |
| [LISA](https://github.com/stevewongv/InstanceShadowDetection) |   23.5    |   21.9    |  39.2   |  37.6  |
|                           **SSIS**                            |   30.2    |   27.1    |  43.5   |  41.3  |
|                          **SSISv2**                           |   35.3    |   29.0    |  50.2   |  44.4  |

### Performance on SOBA-challenge set

|                            Method                             | SOAP mask | SOAP bbox | mask AP | box AP |
| :-----------------------------------------------------------: | :-------: | :-------: | :-----: | :----: |
| [LISA](https://github.com/stevewongv/InstanceShadowDetection) |   10.4    |   10.1    |  23.8   |  24.3  |
|                           **SSIS**                            |   12.7    |   12.8    |  25.6   |  26.2  |
|                          **SSISv2**                           |   17.7    |   15.1    |  31.0   |  28.4  |

Download the dataset `SOBA_v2.zip`, `model_ssis_final.pth`, and `model_ssisv2_final.pth` from [Google drive](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP). Put dataset file in the `../datasets/SOBA_v2/annotations`、`../datasets/SOBA_v2/SOBA` and put pretrained model in the `tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl/`.

## Installation

This repo is implemented on [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), so first build it with:

only for linux

2024-07-21 test

Ubuntu 22.04 follow the steps below

[Google drive](https://drive.google.com/drive/folders/1MKxyq3R6AUeyLai9i9XWzG2C_n5f0ppP)

SOBA_v2.zip unzip to ./dataset/

model_ssisv2_final.pth to ./tools/output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl

```bash
# conda env remove --name ssisv2 -y

conda create -n ssisv2 python=3.10 -y
conda activate ssisv2

sudo apt install build-essential
sudo apt install g++
sudo apt-get install libgeos-dev
sudo apt-get install cmake

# pip install numpy==1.25.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install opencv-python scikit-image kornia==0.5.6
# pip install setuptools==58.2.0

pip install -r requirements.txt

# true
python -c "import torch; print(torch.cuda.is_available())"

pip install git+https://github.com/facebookresearch/detectron2.git

cd PythonAPI
mv common ../
sed -i 's|./common|../common|g' setup.py
python setup.py install

cd ..
# pip install shapely
# pip install scikit-build
python setup.py develop
```

## Quick Start

### Demo

To evaluate the results, try the command example:

```bash
cd demo
python demo.py --input ./samples
```

### Training

```bash
# SSISv2
cd tools
python train_net.py \
    --config-file ../configs/MS_R_101_BiFPN_SSISv2.yaml \
    --num-gpus 1
```

### Evaluation

```bash
# SSISv2
cd tools
python train_net.py \
    --config-file ../configs/MS_R_101_BiFPN_SSISv2.yaml \
    --num-gpus 1 --resume --eval-only

python SOAP.py --path PATH_TO_YOUR_DATASET/SOBA \
    --input-name ./output/SSISv2_MS_R_101_bifpn_with_offset_class_maskiouv2_da_bl
```

# Source

[SSIS](https://github.com/stevewongv/SSIS)

# Citation

If you use LISA, SSIS, SSISv2, SOBA, or SOAP, please use the following BibTeX entry.

```
@ARTICLE{Wang_2022_TPAMI,
author    = {Wang, Tianyu and Hu, Xiaowei and Heng, Pheng-Ann and Fu, Chi-Wing},
journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
title     = {Instance Shadow Detection with A Single-Stage Detector},
year      = {2022},
volume    = {},
number    = {},
pages     = {1-14},
doi       = {10.1109/TPAMI.2022.3185628}
}

@InProceedings{Wang_2021_CVPR,
author    = {Wang, Tianyu and Hu, Xiaowei and Fu, Chi-Wing and Heng, Pheng-Ann},
title     = {Single-Stage Instance Shadow Detection With Bidirectional Relation Learning},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
Year      = {2021},
pages     = {1-11}
}

@InProceedings{Wang_2020_CVPR,
author    = {Wang, Tianyu and Hu, Xiaowei and Wang, Qiong and Heng, Pheng-Ann and Fu, Chi-Wing},
title     = {Instance Shadow Detection},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month     = {June},
year      = {2020}
}
```
