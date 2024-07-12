import configargparse


def parse_args():
    parser = configargparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--exp_path",
        type=str,
        required=True,
        default="./log",
        help="experiment directory",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        default="pose_eval.json",
        help="experiment directory",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["blender", "mip360", "tankstemple", "cambridge_landmark", "all"],
        default="all",
        help="the type of data to validate",
    )

    args, extras = parser.parse_known_args()
    return args, extras
