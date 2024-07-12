import numpy as np
import os
import sys
from scene.scene_structure import CameraInfo, SceneInfo
from scene.colmap_utils import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
    qvec2rotmat,
)
from scene.datasets_utils import store_ply, fetch_ply, get_nerfpp_norm
from utils.graphics_utils import focal2fov, fov2focal
from PIL import Image
import fpsample
import shutil


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE "
                "cameras) supported!"
            )

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path).convert('RGB')

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def read_colmap_scene_info(path, images, eval, llffhold=8, fps_sampling=-1, save_path = "/data/mbortolon/pose-splatting/train_cameras"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = get_nerfpp_norm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        store_ply(ply_path, xyz, rgb)
    try:
        pcd = fetch_ply(ply_path)
    except:
        pcd = None
    
    if fps_sampling > 0:
        points = np.asarray([train_cam_info.T for train_cam_info in train_cam_infos])

        fps_samples_idx = fpsample.fps_sampling(points, fps_sampling, start_idx=0)
        # fps_samples_idx = [0, 10, 16, 24, 28, 30, 36, 40]
        train_cam_infos = [train_cam_infos[fps_sample_idx] for fps_sample_idx in fps_samples_idx]

    # if save_path != "":
    #     os.makedirs(save_path, exist_ok=True)
    #     for idx, train_cam_info in enumerate(train_cam_infos):
    #         shutil.copy(train_cam_info.image_path, os.path.join(save_path, f"{idx}{os.path.splitext(os.path.basename(train_cam_info.image_path))[-1]}"))

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
