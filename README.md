# Cross Attention-based Prior Deformation for Category-level 6D Pose Estimation
Code for "Cross Attention-based Prior Deformation for Category-level 6D Pose Estimation". 

<!-- [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_2) [[Arxiv]](https://arxiv.org/abs/2207.05444) -->

Created by [yangyc275](https://github.com/yangyc275/CAPD_code), Shuai Guo, Lifeng Zhang, Chunge Cao and Yazhou Hu.

![image](https://github.com/yangyc275/CAPD_code/blob/master/pic/pipeline.jpg)


## Requirements
The code has been tested with
- python 3.9.21
- pytorch 1.9.1
- CUDA 11.1

Some dependent packages：
- [gorilla](https://github.com/Gorilla-Lab-SCUT/gorilla-core) 
```
pip install gorilla-core==0.2.5.6
```
- [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch)
```
cd model/pointnet2
python setup.py install
```



## Data Processing

Download the data provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019) ([camera_train](http://download.cs.stanford.edu/orion/nocs/camera_train.zip), [camera_test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip), [camera_composed_depths](http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip), [real_train](http://download.cs.stanford.edu/orion/nocs/real_train.zip), [real_test](http://download.cs.stanford.edu/orion/nocs/real_test.zip),
[ground truths](http://download.cs.stanford.edu/orion/nocs/gts.zip),
and [mesh models](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)) and segmentation results ([Link](https://drive.google.com/file/d/1hNmNRr7YRCgg-c_qdvaIzKEd2g4Kac3w/view?usp=sharing)), and unzip them in data folder as follows:

```
data
├── CAMERA
│   ├── train
│   └── val
├── camera_full_depths
│   ├── train
│   └── val
├── Real
│   ├── train
│   └── test
├── gts
│   ├── val
│   └── real_test
├── obj_models
│   ├── train
│   ├── val
│   ├── real_train
│   └── real_test
├── segmentation_results
│   ├── train_trainedwoMask
│   ├── test_trainedwoMask
│   └── test_trainedwithMask
└── mean_shapes.npy
```
Run the following scripts to prepare the dataset:

```
python data_processing.py
```
## Training CAPD under Settings

```
python train.py --gpus 0 --config config/supervised.yaml
```
## Evaluation
Download trained models and test results [[Link](https://drive.google.com/file/d/1_Wn2W9Gy9MO_0ixWKdHJMN_Bd8vGz9KP/view?usp=drive_link)]. Extract the file to the code directory; the file will be at /log/supervised/epoch_30.pth. Evaluate our models as follows:
```
python test.py --config config/supervised.yaml
```

## Results
Qualitative results on CAMERA25 and REAL275 test set:

|          | IoU25 | IoU75 | 5 degree 2 cm | 5 degree 5 cm | 10 degree 2 cm | 10 degree 5 cm |
|----------|-------|-------|---|---|---|---|
| CAMERA25 | 94.7  | 90.7  | 67.3 | 75.9 | 78.4 | 90.3 |
| REAL275  | 84.3  | 76.4  | 47.5 | 54.3 | 70.6 | 79.6 |



## Acknowledgements

Our implementation leverages the code from [NOCS](https://github.com/hughw19/NOCS_CVPR2019), [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), [SPD](https://github.com/mentian/object-deformnet), and [Self-DPDN](https://github.com/JiehongLin/Self-DPDN).

## Contact

`guoshuaidoc@zzu.edu.cn`
