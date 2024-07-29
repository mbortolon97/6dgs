from tqdm import tqdm
import os
from PIL import Image
import os

import numpy as np
from utils.graphics_utils import focal2fov, BasicPointCloud, fov2focal
from utils.sh_utils import SH2RGB
from scene.scene_structure import CameraInfo, SceneInfo
from scene.datasets_utils import get_nerfpp_norm, fetch_ply, store_ply

def readCamerasFromTTPoses(path, split, transformsfiles, imgfiles, intrinsics, white_background, downsample):
    cam_infos = []
    if split == "train":
        pose_files = [x for x in transformsfiles if x.startswith("0_")]
        img_files = [x for x in imgfiles if x.startswith("0_")]
    elif split == "test":
        test_pose_files = [x for x in transformsfiles if x.startswith("2_")]
        test_img_files = [x for x in imgfiles if x.startswith("2_")]
        if len(test_pose_files) == 0:
            test_pose_files = [x for x in transformsfiles if x.startswith("1_")]
            test_img_files = [x for x in imgfiles if x.startswith("1_")]
        pose_files = test_pose_files
        img_files = test_img_files

    # ray directions for all pixels, same for all images (same H, W, focal)
    image = Image.open(os.path.join(path, "rgb", img_files[-1]).strip())
    w = image.width
    h = image.height
    # ori_directions, dx, dy = get_ray_directions_Ks(h, w, intrinsics)
    # directions = ori_directions / torch.norm(ori_directions, dim=-1, keepdim=True)
    # dx = dx.contiguous()
    # dy = dy.contiguous()

    # K = intrinsics

    for idx, (img_fname, pose_fname) in enumerate(tqdm(zip(img_files, pose_files))):
        c2w = np.loadtxt(os.path.join(path, "pose", pose_fname))  # @ cam_trans
        # c2w = torch.tensor(c2w, dtype=torch.float32).contiguous()

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, "rgb", img_fname)
        # image_paths.append(image_path)
        image = Image.open(image_path)

        if downsample != 1.0:
            img = img.resize([w,h], Image.LANCZOS)
        # img = self.transform(img)  # (4, h, w)
        # img = img.permute(1, 2, 0)  # (h, w, 4) RGBA
        # self.all_rgbs.append(img)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(intrinsics[0,0], w)
        fovy = focal2fov(intrinsics[1, 1], h)

        FovX = fovx
        FovY = fovy

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=img_fname,
                width=image.size[0],
                height=image.size[1],
            )
        )

    return cam_infos

def center_crop(img, new_width=None, new_height=None):        
    width = img.width
    height = img.height

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    center_cropped_img = img.crop((left, top, right, bottom))

    return center_cropped_img


def read_tanksandtemples_scene_info(path, eval, white_background=True, extension=".png", downsample=1.0):

    np_intrinsic = np.loadtxt(os.path.join(path, "intrinsics.txt"))
    intrinsics = np_intrinsic[:3, :3]
    # intrinsics = (torch.from_numpy(np_intrinsic)[:3, :3].to(dtype=torch.float32))[None]
    # intrinsics[:, :2] /= (downsample)  # modify focal length and principal point to match size self.img_wh
    # intrinsics = intrinsics.contiguous()

    pose_files = sorted(os.listdir(os.path.join(path, "pose")))
    img_files = sorted(os.listdir(os.path.join(path, "rgb")))


    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTTPoses(path, "train", pose_files, img_files, intrinsics, white_background, downsample)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTTPoses(path, "test", pose_files, img_files, intrinsics, white_background, downsample)

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = get_nerfpp_norm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    bbox_path = os.path.join(path, "bbox.txt")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the provided bounding boxes
        if os.path.exists(bbox_path):
            bbox = np.loadtxt(os.path.join(path, "bbox.txt"))
            x = np.random.default_rng().uniform(bbox[0], bbox[3], num_pts)
            y = np.random.default_rng().uniform(bbox[1], bbox[4], num_pts)
            z = np.random.default_rng().uniform(bbox[2], bbox[5], num_pts)
            xyz = np.vstack([x,y,z]).T
        else:
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        store_ply(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetch_ply(ply_path)
    except:
        pcd = None
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
