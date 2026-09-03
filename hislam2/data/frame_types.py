from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StereoFisheyeFrame:
    """One synchronized frame from a calibrated stereo fisheye rig.

    All images stay at their native resolution. Transform names follow the
    ``T_destination_from_source`` convention.
    """

    index: int
    frame_number: int
    timestamp: float
    rgb: torch.Tensor
    camera_model: str
    camera_params: torch.Tensor
    image_size: torch.Tensor
    T_rig_from_camera: torch.Tensor
    gt_T_world_from_rig: torch.Tensor
    gt_range: torch.Tensor
    range_observation_valid: torch.Tensor
