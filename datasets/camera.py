# Partially adapted from: https://github.com/author/StructDiffusion
# Original file: src/StructDiffusion/utils/brain2/camera.py
# Author: Weiyu Liu
# License: MIT — see original repository for full details

import h5py
import numpy as np

from typing import Dict, Union, Self, cast


class StructDiffusionCamera(object):
    def __init__(
        self,
        proj_near: float = 0.01,
        proj_far: float = 5.0,
        proj_fov: float = 60.0,
        img_width: int = 640,
        img_height: int = 480,
        camera_pose: np.ndarray = np.eye(4, 4),
    ) -> None:
        assert camera_pose.shape == (4, 4)
        self.pose = camera_pose
        self.img_width = img_width
        self.img_height = img_height

        # Compute focal params
        aspect_ratio = self.img_width / self.img_height
        e: float = 1 / (np.tan(np.radians(proj_fov / 2.0)))
        t = proj_near / e
        b = -t
        r = t * aspect_ratio
        l = -r
        # pixels per meter
        alpha = self.img_width / (r - l)
        self.focal_length = proj_near * alpha
        self.fx = self.focal_length
        self.fy = self.focal_length
        self.cx = self.img_width / 2.0
        self.cy = self.img_height / 2.0

    @classmethod
    def from_h5(cls, h5: Union[h5py.File, Dict[str, h5py.Dataset]]) -> Self:
        proj_near = cast(h5py.Dataset, h5["cam_near"])[()]
        proj_far = cast(h5py.Dataset, h5["cam_far"])[()]
        proj_fov = cast(h5py.Dataset, h5["cam_fov"])[()]
        width = cast(h5py.Dataset, h5["cam_width"])[()]
        height = cast(h5py.Dataset, h5["cam_height"])[()]()
        return cls(proj_near, proj_far, proj_fov, width, height)

    def update_pose(self, pose: np.ndarray) -> None:
        assert pose.shape == (4, 4)
        self.pose = pose

    def update_pose_from_h5(
        self, h5: Union[h5py.File, Dict[str, h5py.Dataset]], step_t: int = -1
    ) -> None:
        self.pose = cast(np.ndarray, cast(h5py.Dataset, h5["ee_camera_view"])[step_t])
        translation: np.ndarray = cast(h5py.Dataset, h5["ee_cam_pose"])[:][:3, 3]
        self.pose[:3, 3] = translation
        assert self.pose.shape == (4, 4)

    def compute_xyz(self, depth: np.ndarray, max_clip_depth: float = 5) -> np.ndarray:
        assert depth.shape[0] == self.img_height
        assert depth.shape[1] == self.img_width

        u: np.ndarray
        v: np.ndarray
        v, u = np.indices(
            (self.img_height, self.img_width)
        )  # v: row indices, u: col indices
        u = u.flatten()
        v = v.flatten()
        z = depth.flatten()
        # clip points that are farther than max distance
        z[z > max_clip_depth] = 0
        x = (u * z - self.cx * z) / self.fx
        y = (v * z - self.cy * z) / self.fy
        raw_pc = np.stack([x, y, z], -1).reshape(self.img_height, self.img_width, 3)
        return raw_pc

    def transform_to_world_coord(self, xyz: np.ndarray) -> np.ndarray:
        assert len(xyz.shape) == 2 and xyz.shape[1] == 3
        if len(xyz) == 0:
            return xyz.copy()
        # Quickly check to see if self.pose is an identity matrix
        if np.abs(self.pose - np.eye(4, 4)).max() < 1e-8:
            return np.ascontiguousarray(xyz.copy())
        # apply translation and rotation
        stack = np.column_stack((xyz, np.ones(xyz.shape[0])))
        return np.dot(self.pose, stack.T).T
