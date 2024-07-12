import torch

def compute_translation_error(translation1, translation2):
    return torch.linalg.norm(translation1 - translation2)

def compute_angular_error(rotation_gt, rotation_est):
    cos_angle = (torch.trace(rotation_gt @ torch.linalg.inv(rotation_est)) - 1) / 2
    return torch.rad2deg(torch.arccos(torch.clamp(cos_angle, min=-1, max=1)))
