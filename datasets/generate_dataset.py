import random
from pathlib import Path
from multiprocessing import Pool, set_start_method
from typing import Dict, List, Tuple, Callable
from tqdm import tqdm

import cv2
import numpy as np
import open3d as o3d
import pybullet as pb
import pybullet_data
from pybullet_utils.bullet_client import BulletClient


def random_pose(table_z: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random object pose (position + )

    Args:
        table_z (float, optional): Height of the table. Defaults to 0.75.

    Returns:
        Tuple[List[float], List[float]]: A random 6-DoF pose
    """
    x, y = np.random.uniform(-0.25, 0.25, 2)
    z: float = table_z + 0.2
    yaw: float = np.random.uniform(-np.pi, np.pi)
    quat: Tuple[float, float, float, float] = pb.getQuaternionFromEuler([0, 0, yaw])
    return np.asarray([x, y, z]), np.asarray(quat)


class RandomObjectLoader:
    def __init__(self, obj_dir: Path) -> None:
        _dir_list: List[Path] = [d for d in obj_dir.iterdir() if d.is_dir()]
        self.urdf_path_list: List[Path] = [
            d / "model.urdf" for d in _dir_list if (d / "model.urdf").exists()
        ]
        if len(self.urdf_path_list) == 0:
            raise FileNotFoundError("No URDF files found in the objects directory.")

    def __call__(self, bc: BulletClient) -> int:
        """Load a random URDF object from the objects directory into the simulation.

        Returns:
            int: The unique ID assigned by PyBullet
        """
        urdf_path: str = random.choice(self.urdf_path_list).as_posix()
        pos, quat = random_pose()
        return bc.loadURDF(urdf_path, pos.tolist(), quat.tolist(), useFixedBase=False)


def setup_simulation(bc: BulletClient) -> int:
    """Reset the simulation environment and load the plane.

    Returns:
        int: The ID of the plane object
    """
    bc.resetSimulation()
    bc.setGravity(0, 0, -9.81)
    bc.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id: int = bc.loadURDF("plane.urdf")
    return plane_id


class PointCloudRenderer:
    def __init__(self, width: int, height: int) -> None:
        self._near, self._far = 0.1, 5.0
        self.width, self.height = width, height
        self.view_matrix: List[float] = pb.computeViewMatrix(
            cameraEyePosition=[0, 0, 1],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
        )
        self.projection_matrix: List[float] = pb.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=self._near, farVal=self._far
        )
        self.fx = self.fy = width / (2 * np.tan(np.deg2rad(60) / 2))
        self.cx, self.cy = width / 2, height / 2
        self.coef_u: np.ndarray = (np.arange(width)[None, :] - self.cx) / self.fx
        self.coef_v: np.ndarray = (np.arange(height)[:, None] - self.cy) / self.fy

    def __call__(
        self, bc: BulletClient
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray
    ]:
        # Raw camera image
        img = bc.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.projection_matrix,
            renderer=bc.ER_TINY_RENDERER,
        )

        rgb_pixels: np.ndarray = np.asarray(img[2]).astype(np.float32)[:, :, :3]
        depth_buffer: np.ndarray = (
            self._far
            * self._near
            / (
                self._far
                - (self._far - self._near) * np.asarray(img[3]).astype(np.float32)
            )
        )  # Convert to meters
        segmentation_mask: np.ndarray = np.asarray(img[4]).astype(np.int32)

        # Compute 3D coordinates
        xs: np.ndarray = self.coef_u * depth_buffer
        ys: np.ndarray = self.coef_v * depth_buffer
        zs: np.ndarray = depth_buffer
        points: np.ndarray = np.dstack((xs, -ys, -zs)).reshape(-1, 3)
        colors: np.ndarray = (rgb_pixels / 255.0).reshape(-1, 3)

        # Extract object IDs
        object_ids: np.ndarray = segmentation_mask.reshape(-1)
        return points, colors, object_ids, (rgb_pixels, depth_buffer), segmentation_mask


def save_scene(
    out_dir: Path,
    scene_idx: int,
    language: str,
    obj_pointclouds: Dict[int, Dict[str, np.ndarray]],
    rgbd: Tuple[np.ndarray, np.ndarray],
    segmentation_mask: np.ndarray,
) -> None:
    scene_dir: Path = out_dir / f"{scene_idx:06d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Save language instruction
    (scene_dir / "language.txt").write_text(language)

    # Combine and save point clouds
    points: np.ndarray = np.empty((0, 3))
    colors: np.ndarray = np.empty((0, 3))
    segmentation: np.ndarray = np.empty((0, 1), dtype=np.int32)
    for oid, pts in obj_pointclouds.items():
        points = np.vstack((points, pts["points"]))
        colors = np.vstack((colors, pts["colors"]))
        segmentation = np.vstack(
            (segmentation, np.full((pts["points"].shape[0], 1), oid))
        )

    pcd_t: o3d.t.geometry.PointCloud = o3d.t.geometry.PointCloud(
        device=o3d.core.Device("CPU:0")
    )
    pcd_t.point["positions"] = o3d.core.Tensor(points)
    pcd_t.point["colors"] = o3d.core.Tensor(colors)
    pcd_t.point["segmentation"] = o3d.core.Tensor(segmentation)
    o3d.t.io.write_point_cloud(str(scene_dir / "pcl_init.ply"), pcd_t)

    # Save RGB-D image
    rgb: np.ndarray = rgbd[0]
    depth: np.ndarray = rgbd[1]
    cv2.imwrite(
        str(scene_dir / "rgb.png"),
        cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(scene_dir / "depth.png"), (depth * 1000).astype(np.uint16)
    )  # convert to mm

    # Save segmentation mask
    cv2.imwrite(str(scene_dir / "seg_mask.png"), segmentation_mask.astype(np.uint8))
    cv2.imwrite(str(scene_dir / "seg_mask16.png"), segmentation_mask.astype(np.uint16))


class GenerateSceneWorker:
    def __init__(
        self,
        obj_dir: Path,
        out_dir: Path,
        obj_count_range: Tuple[int, int] = (3, 5),
        camera_width: int = 640,
        camera_height: int = 480,
    ) -> None:
        self.obj_dir: Path = obj_dir
        self.out_dir: Path = out_dir
        self.obj_count_range: Tuple[int, int] = obj_count_range
        self.render_pointcloud = PointCloudRenderer(camera_width, camera_height)
        self.load_random_object: Callable[[BulletClient], int] = RandomObjectLoader(
            self.obj_dir
        )

    def __call__(self, scene_idx: int) -> int:
        bc = BulletClient(connection_mode=pb.DIRECT)
        plane_id: int = setup_simulation(bc)
        assert plane_id == 0
        obj_ids: List[int] = [plane_id] + [
            self.load_random_object(bc)
            for _ in range(random.randint(*self.obj_count_range))
        ]

        # Step the simulation to settle objects
        for _ in range(100):
            bc.stepSimulation()

        points, colors, obj_mask, rgbd, segmentation_mask = self.render_pointcloud(bc)
        obj_clouds: Dict[int, Dict[str, np.ndarray]] = {
            oid: {"points": points[obj_mask == oid], "colors": colors[obj_mask == oid]}
            for oid in np.unique(obj_mask)
            if oid in obj_ids
        }

        language_instruction: str = "Place a small circle in the middle of the table."
        save_scene(
            self.out_dir,
            scene_idx,
            language_instruction,
            obj_clouds,
            rgbd,
            segmentation_mask,
        )
        bc.resetSimulation()
        bc.disconnect()
        return 1


def main(
    obj_dir: Path,
    out_dir: Path,
    n_scenes: int = 1000,
    obj_count_range: Tuple[int, int] = (3, 5),
    max_processes: int = 8,
) -> None:
    generate_scene_worker: Callable[[int], int] = GenerateSceneWorker(
        obj_dir, out_dir, obj_count_range
    )
    with Pool(processes=max_processes, maxtasksperchild=1) as pool:
        with tqdm(total=n_scenes, desc="Generating Scenes") as pbar:
            for _ in pool.imap_unordered(generate_scene_worker, range(n_scenes)):
                pbar.update(1)


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    main(
        Path("GSO/models"),
        Path("GSO/scenes"),
        n_scenes=1000,
        max_processes=24,
    )
