import random
import pathlib
from multiprocessing import Manager, Pool
from typing import Dict, List, Tuple, Callable
from tqdm import tqdm

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


def make_random_object_loader(
    bc: BulletClient, obj_dir: pathlib.Path
) -> Callable[[], int]:
    dir_list: List[pathlib.Path] = [d for d in obj_dir.iterdir() if d.is_dir()]
    urdf_path_list: List[pathlib.Path] = [
        d / "model.urdf" for d in dir_list if (d / "model.urdf").exists()
    ]
    if len(urdf_path_list) == 0:
        raise FileNotFoundError("No URDF files found in the objects directory.")

    def _load_random_object() -> int:
        """Load a random URDF object from the objects directory into the simulation.

        Returns:
            int: The unique ID assigned by PyBullet
        """
        urdf_path: str = random.choice(urdf_path_list).as_posix()
        pos, quat = random_pose()
        return bc.loadURDF(urdf_path, pos.tolist(), quat.tolist(), useFixedBase=False)

    return _load_random_object


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


def render_pointcloud(
    bc: BulletClient, width: int = 640, height: int = 480
) -> Tuple[np.ndarray, np.ndarray]:
    view_matrix: List[float] = bc.computeViewMatrix(
        cameraEyePosition=[0, 0, 1],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0],
    )
    projection_matrix: List[float] = bc.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=5.0
    )
    img = bc.getCameraImage(
        width,
        height,
        view_matrix,
        projection_matrix,
        renderer=bc.ER_TINY_RENDERER,
    )
    import cv2

    cv2.imwrite("tmp.png", img[2])
    depth_buffer: np.ndarray = (
        np.array(img[3]).astype(np.float32) * 1000.0
    )  # Convert from mm to meters
    segmentation_mask: np.ndarray = np.array(img[4]).astype(np.int32)

    # Compute 3D coordinates
    fx = fy = width / (2 * np.tan(np.deg2rad(60) / 2))
    cx, cy = width / 2, height / 2
    xs: np.ndarray = (np.arange(width)[None, :] - cx) * depth_buffer / fx
    ys: np.ndarray = (np.arange(height)[:, None] - cy) * depth_buffer / fy
    zs: np.ndarray = depth_buffer
    points: np.ndarray = np.dstack((xs, -ys, -zs)).reshape(-1, 3)

    # Extract object IDs
    object_ids: np.ndarray = segmentation_mask.reshape(-1)
    return points, object_ids


def save_scene(
    out_dir: pathlib.Path,
    scene_idx: int,
    language: str,
    obj_pointclouds: Dict[int, np.ndarray],
) -> None:
    scene_dir: pathlib.Path = out_dir / f"{scene_idx:06d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Save language instruction
    (scene_dir / "language.txt").write_text(language)

    # Combine and save point clouds
    combined_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    for pts in obj_pointclouds.values():
        pcd_part: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(pts)
        combined_pcd += pcd_part

    o3d.io.write_point_cloud(str(scene_dir / "pcl_init.pcd"), combined_pcd)


class GenerateSceneWorker:
    def __init__(
        self,
        obj_dir: pathlib.Path,
        out_dir: pathlib.Path,
        obj_count_range: Tuple[int, int] = (3, 5),
    ) -> None:
        self.obj_dir = obj_dir
        self.out_dir = out_dir
        self.obj_count_range = obj_count_range

    def __call__(self, scene_idx: int) -> int:
        bc = BulletClient(connection_mode=pb.DIRECT)
        plane_id: int = setup_simulation(bc)
        assert plane_id == 0
        _load_random_object: Callable[[], int] = make_random_object_loader(
            bc, self.obj_dir
        )
        obj_ids: List[int] = [plane_id] + [
            _load_random_object() for _ in range(random.randint(*self.obj_count_range))
        ]

        # Step the simulation to settle objects
        for _ in range(100):
            bc.stepSimulation()

        points, obj_mask = render_pointcloud(bc)
        obj_clouds: Dict[int, np.ndarray] = {
            oid: points[obj_mask == oid]
            for oid in np.unique(obj_mask)
            if oid in obj_ids
        }

        language_instruction: str = "Place a small circle in the middle of the table."
        save_scene(self.out_dir, scene_idx, language_instruction, obj_clouds)
        bc.disconnect()
        return 1


def main(
    obj_dir: pathlib.Path,
    out_dir: pathlib.Path,
    n_scenes: int = 1000,
    obj_count_range: Tuple[int, int] = (3, 5),
    max_processes: int = 1,
) -> None:
    generate_scene_worker: Callable[[int], int] = GenerateSceneWorker(
        obj_dir, out_dir, obj_count_range
    )
    with Pool(processes=max_processes) as pool:
        with tqdm(total=n_scenes, desc="Generating Scenes") as pbar:
            for _ in pool.imap_unordered(generate_scene_worker, range(n_scenes)):
                pbar.update(1)


if __name__ == "__main__":
    main(pathlib.Path("GSO/models"), pathlib.Path("GSO/scenes"), 1)
