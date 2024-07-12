import os
from cfg_grammar import parse_config


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_checkpoint_arguments(root_dir):
    with open(os.path.join(root_dir, "cfg_args")) as filehandle:
        config_dict = parse_config(filehandle.read())
    return dotdict(config_dict)


def get_highest_valid_checkpoint(root_dir):
    ckpt_dir = os.path.join(root_dir, "point_cloud")
    sorted_filenames = sorted(os.listdir(ckpt_dir), reverse=True)

    largest_checkpoint = -1
    largest_checkpoint_path = ""
    for dir_name in sorted_filenames:
        ckpt_components = dir_name.split("_")
        if ckpt_components[0] != "iteration":
            continue
        try:
            val = int(ckpt_components[1])  # or int
            # here goes the code that relies on val
        except ValueError:
            continue
        ckpt_filepath = os.path.join(ckpt_dir, dir_name, "point_cloud.ply")
        if not os.path.exists(ckpt_filepath):
            continue
        if largest_checkpoint > val:
            continue

        largest_checkpoint = val
        largest_checkpoint_path = ckpt_filepath

    return largest_checkpoint_path


def parse_exp_dir(exp_dir, prefix):
    objects_checkpoints = {}
    exp_dirs_filenames = os.listdir(exp_dir)
    exp_dirs_filenames = sorted(exp_dirs_filenames)
    for exp_dir_filename in exp_dirs_filenames:
        exp_dir_filepath = os.path.join(exp_dir, exp_dir_filename)
        if not (
            os.path.isdir(exp_dir_filepath) and exp_dir_filename.startswith(prefix)
        ):
            continue

        name_components = exp_dir_filename.split("_")
        sequence_id = name_components[-1]
        category_name = "_".join(name_components[:-1])
        checkpoint_filepath = get_highest_valid_checkpoint(exp_dir_filepath)
        if checkpoint_filepath == "":
            print(
                f"Object {sequence_id} of category {category_name} skipped because no valid checkpoint found"
            )
            continue
        objects_checkpoints[sequence_id] = {
            "exp_dir_filepath": exp_dir_filepath,
            "checkpoint_filepath": checkpoint_filepath,
            "sequence_id": sequence_id,
            "category_name": category_name,
        }
    return objects_checkpoints
