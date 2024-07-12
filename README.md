# 6DGS
## [Project page](https://mbortolon97.github.io/6dgs/) |  [Paper](https://arxiv.org/)
This repository contains a PyTorch implementation for the paper: [6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model](https://arxiv.org/).

## Installation

#### Tested on Ubuntu 22.04 + Pytorch 1.10.1 

Install environment:
```
conda create -n TensoRF python=3.12
conda activate TensoRF
pip install torch torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```


## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)


## Quick start
### Training the base NeRF model (using TensoRF)
The training script is in `train.py`, to train a TensoRF:

```
python train.py --config configs/lego.txt
```

We provide two scripts that it is necessary only to edit with the correct paths to the dataset:
```
sh tools/launch_all_blender_training.sh
sh tools/launch_all_tanks_and_temple_training.sh
```

### Run the pose estimation
The training and testing script for the pose estimation is located in `train_eval_pose_est.py`, for training and testing on all the objects from Blender:

```
python train_eval_pose_est.py --config configs/lego.txt --datadir datasets/nerf_synthetic --out_path test_results_synthetic.json
```

```
python train_eval_pose_est.py --config configs/truck.txt --datadir datasets/TanksAndTemple --out_path test_results_tt.json
```



## Citation
If you find our code or paper helps, please consider citing:
```
@INPROCEEDINGS{Bortolon2024IFFNeRF,
  author = {Bortolon, Matteo and Tsesmelis, Theodore and James, Stuart and Poiesi, Fabio and Del Bue, Alessio},
  title = {IFFNeRF: Initialisation Free and Fast 6DoF pose estimation from a single image and a NeRF model},
  booktitle = {ICRA},
  year = {2024}
}
```