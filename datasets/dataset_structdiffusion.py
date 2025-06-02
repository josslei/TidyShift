from ast import Str
import h5py
from sympy import hyper
import torch
import cv2
import numpy as np

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any, cast
from numpy.typing import NDArray

from camera import StructDiffusionCamera


class StructDiffusionDataset(torch.utils.data.Dataset):

    def __init__(
        self, h5_path: Path, num_points: int = 1024, ignore_rgb: bool = True
    ) -> None:
        self.raw: Dict[str, h5py.Dataset] = {
            key: val for key, val in h5py.File(h5_path).items()
        }
        self.num_points = num_points
        self.ignore_rgb = ignore_rgb

    def __len__(self) -> int:
        raise NotImplementedError
        return 0

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ids = get_ids(self.raw)
        goal_specification = json.loads(
            np.asarray(self.raw["goal_specification"]).item().decode()
        )
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        target_objs = all_objs[:num_rearrange_objs]
        other_objs = all_objs[num_rearrange_objs:]

        # important: only using the last step
        step_t = num_rearrange_objs

        # Load RGB image
        rgb_img = PNGToNumpy(self.raw["ee_rgb"][step_t]) / 255.0
        # Load depth image (assumed to be single-channel depth in a PNG or similar)
        d_min = cast(float, self.raw["ee_depth_min"][step_t])
        d_max = cast(float, self.raw["ee_depth_max"][step_t])
        depth_img = cast(
            NDArray[np.float64],
            self.raw["ee_depth"][step_t] / 20000.0 * (d_max - d_min) + d_min,
        )
        validity_img: NDArray[np.bool] = np.logical_and(
            depth_img > 0.1, depth_img < 2.0
        )
        # Load segmentation mask image
        seg_img: NDArray[np.uint8] = PNGToNumpy(self.raw["ee_seg"][step_t])

        # Compute the XYZ coordinate image from the depth image using intrinsics
        # Then convert to points under the world coord
        camera = StructDiffusionCamera.from_h5(self.raw)
        camera.update_pose_from_h5(self.raw, step_t)
        xyz = camera.compute_xyz(depth_img)  # shape (H, W, 3)
        _h, _w, _ = xyz.shape
        xyz = xyz.reshape(_h * _w, -1)  # shape (H * W, 3)
        xyz = camera.transform_to_world_coord(xyz)

        # Get object point clouds
        obj_pcs: List[np.ndarray] = []
        other_obj_pcs: List[np.ndarray] = []
        for obj in all_objs:
            obj_mask: NDArray[np.bool] = np.logical_and(
                seg_img == ids[obj], validity_img
            ).flatten()
            assert np.sum(obj_mask) > 0
            obj_xyz, obj_rgb = sample_masked_points(
                xyz, rgb_img.reshape(-1, 3), obj_mask, self.num_points
            )

            pc: np.ndarray = (
                obj_xyz if self.ignore_rgb else np.concat([obj_xyz, obj_rgb], axis=-1)
            )

            if obj in target_objs:
                obj_pcs += [pc]
            elif obj in other_objs:
                other_obj_pcs += [pc]
            else:
                assert False, "shouldn't happen"

        # ---------------------------------------------------------

        # Load pose data (assumed to be stored as 4x4 numpy arrays)
        obj_start_pose = np.load(obj_start_path)  # shape (4,4)
        obj_target_pose = np.load(obj_target_path)  # shape (4,4)
        if os.path.exists(camera_pose_path):
            camera_pose = np.load(camera_pose_path)
        else:
            # If camera pose not provided, assume identity (camera at world origin)
            camera_pose = np.eye(4, dtype=np.float32)

        # Convert all relevant data to torch.Tensors
        point_cloud_tensor = torch.from_numpy(
            object_points.astype(np.float32)
        )  # (N,3) float32
        xyz_tensor = torch.from_numpy(xyz.astype(np.float32))  # (H,W,3) float32
        rgb_tensor = torch.from_numpy(
            rgb_img.astype(np.float32)
        )  # (H,W,3) float32, values 0-255
        seg_tensor = torch.from_numpy(
            seg_img.astype(np.int64)
        )  # (H,W) int64 for labels
        obj_start_tensor = torch.from_numpy(
            obj_start_pose.astype(np.float32)
        )  # (4,4) float32
        obj_target_tensor = torch.from_numpy(
            obj_target_pose.astype(np.float32)
        )  # (4,4) float32
        camera_pose_tensor = torch.from_numpy(
            camera_pose.astype(np.float32)
        )  # (4,4) float32

        # Prepare the sample dictionary
        sample: Dict[str, torch.Tensor] = {
            "point_cloud": point_cloud_tensor,
            "xyz": xyz_tensor,
            "rgb": rgb_tensor,
            "segmentation": seg_tensor,
            "obj_start_pose": obj_start_tensor,
            "obj_target_pose": obj_target_tensor,
            "camera_pose": camera_pose_tensor,
        }

        # Apply the optional transform for augmentation, if provided
        if self.transform:
            sample = self.transform(sample)
        # The transform is expected to handle the sample dict (e.g., jitter points, normalize images, etc.)

        return sample


def get_ids(h5: Union[h5py.File, Dict]) -> Dict[str, np.int64]:
    ids = {}
    for k in h5.keys():
        if k.startswith("id_"):
            ids[k[3:]] = cast(h5py.Dataset, h5[k])[()]
    return ids


def PNGToNumpy(png: np.bytes_) -> np.ndarray:
    buf = np.frombuffer(png, dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def sample_masked_points(
    xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray, num_points: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    assert np.sum(mask) > 0
    replace: bool = xyz.shape[0] < num_points
    idx = np.random.choice(xyz.shape[0], num_points, replace=replace)
    return xyz[idx], rgb[idx]


# Minimal usage example
if __name__ == "__main__":
    pass
