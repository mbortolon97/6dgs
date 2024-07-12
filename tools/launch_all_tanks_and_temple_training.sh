#!/bin/sh

python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/truck.txt --datadir /home/mbortolon/data/datasets/TanksAndTemple/Barn --expname tensorf_Barn_VMtt
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/truck.txt --datadir /home/mbortolon/data/datasets/TanksAndTemple/Caterpillar --expname tensorf_Caterpillar_VMtt
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/truck.txt --datadir /home/mbortolon/data/datasets/TanksAndTemple/Family --expname tensorf_Family_VMtt
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/truck.txt --datadir /home/mbortolon/data/datasets/TanksAndTemple/Ignatius --expname tensorf_Ignatius_VMtt
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/truck.txt --datadir /home/mbortolon/data/datasets/TanksAndTemple/Truck --expname tensorf_Truck_VMtt
