import os
from scene.dataset_readers import sceneLoadTypeCallbacks


def load_data(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            args.source_path, args.images, args.eval, fps_sampling=args.fps_sampling
        )
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            args.source_path, args.white_background, args.eval
        )
    elif os.path.exists(os.path.join(args.source_path, "intrinsics.txt")):
        print("Found intrinsics.txt file, assuming Tanks And Temple data set!")
        scene_info = sceneLoadTypeCallbacks["TanksTemple"](args.source_path, args.white_background, args.eval)
    else:
        assert False, "Could not recognize scene type!"
    return scene_info

def get_dataset_prefix(source_path):
    if os.path.exists(os.path.join(source_path, "sparse")):
        return "mip_360"
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        return "synthetic"
    elif os.path.exists(os.path.join(source_path, "intrinsics.txt")):
        return "tt"
    elif os.path.exists(os.path.join(source_path, "reconstruction.nvm")):
        return "cl"
    else:
        assert False, "Could not recognize scene type!"