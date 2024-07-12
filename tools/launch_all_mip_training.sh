#!/bin/sh

python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/bicycle.txt --datadir /home/mbortolon/data/datasets/360_v2/bicycle --expname tensorf_bicycle_VMsphere
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/bonsai.txt --datadir /home/mbortolon/data/datasets/360_v2/bonsai --expname tensorf_bonsai_VMsphere
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/counter.txt --datadir /home/mbortolon/data/datasets/360_v2/counter --expname tensorf_counter_VMsphere
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/garden.txt --datadir /home/mbortolon/data/datasets/360_v2/garden --expname tensorf_garden_VMsphere
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/kitchen.txt --datadir /home/mbortolon/data/datasets/360_v2/kitchen --expname tensorf_kitchen_VMsphere
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/room.txt --datadir /home/mbortolon/data/datasets/360_v2/room --expname tensorf_room_VMsphere
python3 train.py --config /home/mbortolon/data/std_nerf_repair/inerf_TensoRF_sampling/configs/stump.txt --datadir /home/mbortolon/data/datasets/360_v2/stump --expname tensorf_stump_VMsphere
