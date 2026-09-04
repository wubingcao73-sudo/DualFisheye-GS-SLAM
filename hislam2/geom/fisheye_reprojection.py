"""Reference Double Sphere reprojection for a fixed multi-camera rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from hislam2.camera import CameraModel


@dataclass(frozen=True)
class ReprojectionValidity:
    """Separated validity components for one reprojection."""

    source_model_valid: torch.Tensor
    source_image_valid: torch.Tensor
    range_valid: torch.Tensor
    target_model_valid: torch.Tensor
    target_image_valid: torch.Tensor
    geometric_valid: torch.Tensor


@dataclass(frozen=True)
class ReprojectionJacobians:
    """Jacobians of predicted target pixels using lietorch twist ordering."""

    source_pose: torch.Tensor
    target_pose: torch.Tensor
    inverse_range: torch.Tensor


@dataclass(frozen=True)
class ReprojectionResult:
    """Projected pixels, target range, validity and optional Jacobians."""

    pixels: torch.Tensor
    target_range: torch.Tensor
    validity: ReprojectionValidity
    jacobians: Optional[ReprojectionJacobians]


def skew_symmetric(points: torch.Tensor) -> torch.Tensor:
    """Return matrices ``[p]_x`` such that ``[p]_x q = p x q``."""
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if not points.is_floating_point() or points.shape[-1] != 3:
        raise ValueError(f"points must have floating shape [..., 3], got {points.shape}")
    x, y, z = points.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(points.shape[:-1] + (3, 3))


def se3_point_action_jacobian(points: torch.Tensor) -> torch.Tensor:
    """Return ``d(Exp(delta) * p)/d(delta)`` at zero.

    The tangent order is lietorch's ``[tx, ty, tz, rx, ry, rz]`` and the
    perturbation is on the left. The result is ``[I, -[p]_x]``.
    """
    skew = skew_symmetric(points)
    identity = torch.eye(3, dtype=points.dtype, device=points.device)
    identity = identity.expand(points.shape[:-1] + (3, 3))
    return torch.cat((identity, -skew), dim=-1)


def _check_transform(transform: torch.Tensor, name: str) -> None:
    if not isinstance(transform, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4], got {transform.shape}")
    if transform.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if not bool(torch.isfinite(transform).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    tolerance = 1e-5 if transform.dtype == torch.float32 else 1e-6
    expected_bottom = transform.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not bool(torch.allclose(transform[3], expected_bottom, atol=tolerance, rtol=0.0)):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    identity = torch.eye(3, dtype=transform.dtype, device=transform.device)
    orthonormal = bool(
        torch.allclose(
            rotation.transpose(0, 1) @ rotation,
            identity,
            atol=tolerance,
            rtol=tolerance,
        )
    )
    proper = bool(
        torch.allclose(
            torch.det(rotation),
            transform.new_tensor(1.0),
            atol=tolerance,
            rtol=tolerance,
        )
    )
    if not orthonormal or not proper:
        raise ValueError(f"{name} rotation is not a proper orthonormal matrix")


def _inverse_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = torch.eye(4, dtype=transform.dtype, device=transform.device)
    inverse[:3, :3] = rotation.transpose(0, 1)
    inverse[:3, 3] = -(rotation.transpose(0, 1) @ translation)
    return inverse


def _transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]


class FisheyeRigReprojector:
    """CPU/CUDA PyTorch reference reprojection for a fixed fisheye rig.

    Rig pose states are ``T_rig_from_world``. Pose Jacobians use left
    perturbations and lietorch tangent order ``[translation, rotation]``.
    """

    def __init__(
        self,
        cameras: Sequence[CameraModel],
        T_rig_from_camera: torch.Tensor,
    ) -> None:
        if not cameras:
            raise ValueError("at least one camera is required")
        if not all(isinstance(camera, CameraModel) for camera in cameras):
            raise TypeError("all cameras must implement CameraModel")
        if not isinstance(T_rig_from_camera, torch.Tensor):
            raise TypeError("T_rig_from_camera must be a torch.Tensor")
        if T_rig_from_camera.shape != (len(cameras), 4, 4):
            raise ValueError(
                "T_rig_from_camera must have shape "
                f"[{len(cameras)}, 4, 4], got {T_rig_from_camera.shape}"
            )
        if T_rig_from_camera.dtype not in (torch.float32, torch.float64):
            raise TypeError("T_rig_from_camera must use float32 or float64")
        if not bool(torch.isfinite(T_rig_from_camera).all()):
            raise ValueError("T_rig_from_camera contains NaN or Inf")
        for index in range(len(cameras)):
            _check_transform(T_rig_from_camera[index], f"T_rig_from_camera[{index}]")
        self.cameras = tuple(cameras)
        self.T_rig_from_camera = T_rig_from_camera.detach().clone()

    def _camera(self, index: int) -> CameraModel:
        if not isinstance(index, int) or not 0 <= index < len(self.cameras):
            raise IndexError(f"camera index {index} is out of range")
        return self.cameras[index]

    def reproject(
        self,
        source_pixels: torch.Tensor,
        source_inverse_range: torch.Tensor,
        T_rig_from_world_source: torch.Tensor,
        T_rig_from_world_target: torch.Tensor,
        source_camera_index: int,
        target_camera_index: int,
        compute_jacobians: bool = False,
    ) -> ReprojectionResult:
        """Reproject source pixels into a target camera.

        Pose Jacobians differentiate predicted target pixels, not residuals.
        Invalid outputs are zero placeholders and must be masked.
        """
        source_camera = self._camera(source_camera_index)
        target_camera = self._camera(target_camera_index)
        if not isinstance(source_pixels, torch.Tensor):
            raise TypeError("source_pixels must be a torch.Tensor")
        if source_pixels.dtype not in (torch.float32, torch.float64):
            raise TypeError("source_pixels must use float32 or float64")
        if source_pixels.ndim == 0 or source_pixels.shape[-1] != 2:
            raise ValueError(
                f"source_pixels must have shape [..., 2], got {source_pixels.shape}"
            )
        if not isinstance(source_inverse_range, torch.Tensor):
            raise TypeError("source_inverse_range must be a torch.Tensor")
        if source_inverse_range.shape != source_pixels.shape[:-1]:
            raise ValueError(
                "source_inverse_range shape must equal source_pixels.shape[:-1], "
                f"got {source_inverse_range.shape} and {source_pixels.shape}"
            )
        if source_inverse_range.dtype != source_pixels.dtype:
            raise TypeError("pixels and inverse range must use the same dtype")
        if source_inverse_range.device != source_pixels.device:
            raise ValueError("pixels and inverse range must use the same device")

        for transform, name in (
            (T_rig_from_world_source, "T_rig_from_world_source"),
            (T_rig_from_world_target, "T_rig_from_world_target"),
        ):
            _check_transform(transform, name)
            if transform.dtype != source_pixels.dtype or transform.device != source_pixels.device:
                raise ValueError(f"{name} must match source pixel dtype and device")

        source_rays, source_model_valid = source_camera.unproject(source_pixels)
        source_image_valid = source_camera.image_valid(source_pixels)
        range_valid = torch.isfinite(source_inverse_range) & (source_inverse_range > 0.0)
        safe_inverse_range = torch.where(
            range_valid, source_inverse_range, torch.ones_like(source_inverse_range)
        )
        points_source_camera = source_rays / safe_inverse_range[..., None]

        E_source = self.T_rig_from_camera[source_camera_index].to(
            dtype=source_pixels.dtype, device=source_pixels.device
        )
        E_target = self.T_rig_from_camera[target_camera_index].to(
            dtype=source_pixels.dtype, device=source_pixels.device
        )
        T_world_from_rig_source = _inverse_rigid_transform(T_rig_from_world_source)
        T_target_camera_from_rig = _inverse_rigid_transform(E_target)

        points_source_rig = _transform_points(E_source, points_source_camera)
        points_world = _transform_points(T_world_from_rig_source, points_source_rig)
        points_target_rig = _transform_points(T_rig_from_world_target, points_world)
        points_target_camera = _transform_points(
            T_target_camera_from_rig, points_target_rig
        )

        if compute_jacobians:
            projected_pixels, target_model_valid, projection_jacobian = (
                target_camera.project_jacobian(points_target_camera)
            )
        else:
            projected_pixels, target_model_valid = target_camera.project(
                points_target_camera
            )
            projection_jacobian = None
        target_image_valid = target_camera.image_valid(projected_pixels)
        geometric_valid = (
            source_model_valid
            & source_image_valid
            & range_valid
            & target_model_valid
            & target_image_valid
        )

        target_range = torch.linalg.vector_norm(points_target_camera, dim=-1)
        pixels = torch.where(
            geometric_valid[..., None], projected_pixels, torch.zeros_like(projected_pixels)
        )
        target_range = torch.where(
            geometric_valid, target_range, torch.zeros_like(target_range)
        )

        jacobians = None
        if compute_jacobians:
            rotation_target_camera_from_target_rig = E_target[:3, :3].transpose(0, 1)
            rotation_target_camera_from_source_rig = (
                rotation_target_camera_from_target_rig
                @ T_rig_from_world_target[:3, :3]
                @ T_rig_from_world_source[:3, :3].transpose(0, 1)
            )
            target_action = se3_point_action_jacobian(points_target_rig)
            target_point_jacobian = torch.matmul(
                rotation_target_camera_from_target_rig, target_action
            )

            source_action = se3_point_action_jacobian(points_source_rig)
            source_inverse_action = -source_action
            source_point_jacobian = torch.matmul(
                rotation_target_camera_from_source_rig, source_inverse_action
            )

            rotation_target_camera_from_source_camera = (
                rotation_target_camera_from_source_rig @ E_source[:3, :3]
            )
            source_point_inverse_range_jacobian = (
                -source_rays / (safe_inverse_range * safe_inverse_range)[..., None]
            )
            target_point_inverse_range_jacobian = torch.matmul(
                rotation_target_camera_from_source_camera,
                source_point_inverse_range_jacobian[..., None],
            )

            source_pose_jacobian = torch.matmul(
                projection_jacobian, source_point_jacobian
            )
            target_pose_jacobian = torch.matmul(
                projection_jacobian, target_point_jacobian
            )
            inverse_range_jacobian = torch.matmul(
                projection_jacobian, target_point_inverse_range_jacobian
            )
            source_pose_jacobian = torch.where(
                geometric_valid[..., None, None],
                source_pose_jacobian,
                torch.zeros_like(source_pose_jacobian),
            )
            target_pose_jacobian = torch.where(
                geometric_valid[..., None, None],
                target_pose_jacobian,
                torch.zeros_like(target_pose_jacobian),
            )
            inverse_range_jacobian = torch.where(
                geometric_valid[..., None, None],
                inverse_range_jacobian,
                torch.zeros_like(inverse_range_jacobian),
            )
            jacobians = ReprojectionJacobians(
                source_pose=source_pose_jacobian,
                target_pose=target_pose_jacobian,
                inverse_range=inverse_range_jacobian,
            )

        validity = ReprojectionValidity(
            source_model_valid=source_model_valid,
            source_image_valid=source_image_valid,
            range_valid=range_valid,
            target_model_valid=target_model_valid,
            target_image_valid=target_image_valid,
            geometric_valid=geometric_valid,
        )
        return ReprojectionResult(
            pixels=pixels,
            target_range=target_range,
            validity=validity,
            jacobians=jacobians,
        )
