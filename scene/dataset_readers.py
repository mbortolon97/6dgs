#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.colmap import read_colmap_scene_info
from scene.synthetic import read_nerf_synthetic_info
from scene.tanksandtemples import read_tanksandtemples_scene_info

sceneLoadTypeCallbacks = {
    "Colmap": read_colmap_scene_info,
    "Blender": read_nerf_synthetic_info,
    "TanksTemple": read_tanksandtemples_scene_info,
}
