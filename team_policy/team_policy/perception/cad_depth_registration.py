from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from sensor_msgs.msg import CameraInfo


class CadDepthPoseEstimator:
    """
    Estimate a 6D pose from a depth crop and an optional CAD model.

    Pipeline:
      1. Crop depth inside the YOLO bbox.
      2. Back-project to a point cloud in the camera frame.
      3. Estimate an initial pose from PCA on the point cloud.
      4. If a CAD model exists for the detection family, refine with ICP.

    All output poses are camera-frame poses of the CAD/object center.
    """

    def __init__(
        self,
        assets_models_dir: Optional[str] = None,
        logger: Optional[Callable[[str], None]] = None,
        voxel_size_m: float = 0.004,
        min_points: int = 60,
    ) -> None:
        self._logger = logger or (lambda msg: None)
        self._voxel_size_m = float(voxel_size_m)
        self._min_points = int(min_points)
        self._mesh_cache: Dict[str, Dict] = {}
        self._assets_models_dir = self._resolve_assets_models_dir(assets_models_dir)

        self._family_to_asset_dir = {
            "taskboard": "Task Board Base",
            "nic": "NIC Card",
            "sc": "SC Port",
            "sfp_module": "SFP Module",
        }

    def _resolve_assets_models_dir(self, assets_models_dir: Optional[str]) -> Optional[Path]:
        candidates = []
        if assets_models_dir:
            candidates.append(Path(assets_models_dir))
        env = os.environ.get("YOLO_DEPTH_POSE_AIC_ASSETS_MODELS_DIR") or os.environ.get("AIC_ASSETS_MODELS_DIR")
        if env:
            candidates.append(Path(env))
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "aic_assets" / "models",
                cwd.parent / "aic_assets" / "models",
                Path(__file__).resolve().parents[2] / "aic_assets" / "models",
                Path(__file__).resolve().parents[3] / "aic_assets" / "models",
            ]
        )
        for cand in candidates:
            try:
                if cand.exists() and cand.is_dir():
                    self._logger(f"CAD assets directory: {cand}")
                    return cand
            except Exception:
                continue
        self._logger("CAD assets directory not found; ICP refinement will be skipped when no mesh can be resolved")
        return None

    def estimate_pose(
        self,
        camera_info: CameraInfo,
        depth_image: np.ndarray,
        det: Dict,
        family: Optional[str],
        init_pose_camera: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if depth_image is None:
            return None
        scene_cloud, stats = self._depth_crop_to_point_cloud(camera_info, depth_image, det)
        if scene_cloud is None or len(scene_cloud.points) < self._min_points:
            return None

        pca_pose = self._estimate_pose_from_pca(np.asarray(scene_cloud.points))
        if pca_pose is None:
            return None

        result = {
            "R": pca_pose["R"].astype(np.float32),
            "t": pca_pose["t"].astype(np.float32),
            "q": self._matrix_to_quaternion(pca_pose["R"]),
            "source": "depth_pca",
            "num_points": int(len(scene_cloud.points)),
            "crop_stats": stats,
        }

        mesh_bundle = self._get_mesh_bundle_for_family(family)
        if mesh_bundle is None:
            return result

        candidate_inits = []
        if init_pose_camera is not None:
            candidate_inits.append(self._pose_dict_to_T(init_pose_camera))
        candidate_inits.extend(self._candidate_icp_inits(mesh_bundle, pca_pose))

        refined = self._run_icp(mesh_bundle, scene_cloud, candidate_inits)
        if refined is None:
            return result

        R = refined[:3, :3].astype(np.float32)
        t = refined[:3, 3].astype(np.float32)
        result.update(
            {
                "R": R,
                "t": t,
                "q": self._matrix_to_quaternion(R),
                "source": "depth_icp",
            }
        )
        return result

    def _depth_crop_to_point_cloud(self, camera_info: CameraInfo, depth_image: np.ndarray, det: Dict):
        h, w = depth_image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in det.get("bbox_xyxy", [0, 0, 0, 0])]
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)
        crop_margin = 0.08
        ix1 = int(np.clip(np.floor(x1 + crop_margin * bw), 0, w - 1))
        iy1 = int(np.clip(np.floor(y1 + crop_margin * bh), 0, h - 1))
        ix2 = int(np.clip(np.ceil(x2 - crop_margin * bw), ix1 + 1, w))
        iy2 = int(np.clip(np.ceil(y2 - crop_margin * bh), iy1 + 1, h))

        patch = np.asarray(depth_image[iy1:iy2, ix1:ix2], dtype=np.float32)
        if patch.size == 0:
            return None, None

        valid = np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)
        if not np.any(valid):
            return None, None

        # Robustly suppress distant board/background points while keeping the object cluster.
        valid_depths = patch[valid]
        z_med = float(np.median(valid_depths))
        z_mad = float(np.median(np.abs(valid_depths - z_med)))
        z_band = max(0.02, 3.5 * z_mad)
        z_min = z_med - z_band
        z_max = z_med + z_band
        valid &= (patch >= z_min) & (patch <= z_max)
        if np.count_nonzero(valid) < self._min_points:
            valid = np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)

        ys, xs = np.where(valid)
        if xs.size < self._min_points:
            return None, None
        xs_img = xs.astype(np.float32) + float(ix1)
        ys_img = ys.astype(np.float32) + float(iy1)
        zs = patch[ys, xs].astype(np.float32)

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx <= 1e-6 or fy <= 1e-6:
            return None, None

        X = (xs_img - cx) * zs / fx
        Y = (ys_img - cy) * zs / fy
        pts = np.column_stack([X, Y, zs]).astype(np.float32)

        # Mild outlier rejection.
        centroid = np.median(pts, axis=0)
        d = np.linalg.norm(pts - centroid.reshape(1, 3), axis=1)
        keep = d <= np.percentile(d, 92.0)
        pts = pts[keep]
        if len(pts) < self._min_points:
            return None, None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(self._voxel_size_m)
        if len(pcd.points) < self._min_points:
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.015, 3.0 * self._voxel_size_m), max_nn=30))
        return pcd, {
            "bbox_xyxy": [ix1, iy1, ix2, iy2],
            "median_depth_m": z_med,
            "num_points": int(len(pcd.points)),
        }

    def _estimate_pose_from_pca(self, points_xyz: np.ndarray) -> Optional[Dict]:
        if points_xyz is None or len(points_xyz) < 8:
            return None
        pts = np.asarray(points_xyz, dtype=np.float64)
        center = np.median(pts, axis=0)
        X = pts - center.reshape(1, 3)
        cov = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        normal = eigvecs[:, order[0]]
        tangent = eigvecs[:, order[-1]]

        # Make the local z-axis face the camera.
        if float(np.dot(normal, center)) > 0.0:
            normal = -normal

        tangent = tangent - normal * float(np.dot(tangent, normal))
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-9:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tangent /= tangent_norm

        # Keep x-axis roughly image-right in the camera frame.
        if tangent[0] < 0.0:
            tangent = -tangent

        bitangent = np.cross(normal, tangent)
        bitangent_norm = float(np.linalg.norm(bitangent))
        if bitangent_norm < 1e-9:
            return None
        bitangent /= bitangent_norm
        tangent = np.cross(bitangent, normal)
        tangent /= max(1e-9, float(np.linalg.norm(tangent)))

        R = np.column_stack([tangent, bitangent, normal]).astype(np.float64)
        return {"R": R, "t": center.astype(np.float64)}

    def _get_mesh_bundle_for_family(self, family: Optional[str]) -> Optional[Dict]:
        if family is None:
            return None
        family = str(family)
        if family in self._mesh_cache:
            return self._mesh_cache[family]
        asset_dir_name = self._family_to_asset_dir.get(family)
        if asset_dir_name is None or self._assets_models_dir is None:
            self._mesh_cache[family] = None
            return None
        asset_dir = self._assets_models_dir / asset_dir_name
        if not asset_dir.exists():
            self._logger(f"CAD asset directory missing for family '{family}': {asset_dir}")
            self._mesh_cache[family] = None
            return None

        mesh_file = self._find_mesh_file(asset_dir)
        if mesh_file is None:
            self._logger(f"No CAD mesh file found in {asset_dir}")
            self._mesh_cache[family] = None
            return None

        try:
            mesh = trimesh.load(mesh_file, force="mesh")
            if hasattr(mesh, "geometry"):
                geometries = [g for g in mesh.geometry.values()]
                mesh = trimesh.util.concatenate(geometries)
        except Exception as exc:
            self._logger(f"Failed to load CAD mesh {mesh_file}: {exc}")
            self._mesh_cache[family] = None
            return None

        if mesh is None or len(mesh.vertices) == 0:
            self._mesh_cache[family] = None
            return None

        mesh = mesh.copy()
        center = mesh.bounding_box.centroid
        mesh.vertices = mesh.vertices - center.reshape(1, 3)
        sampled = mesh.sample(5000)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(np.asarray(sampled, dtype=np.float64))
        target = target.voxel_down_sample(self._voxel_size_m)
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.015, 3.0 * self._voxel_size_m), max_nn=30))

        pca = self._estimate_pose_from_pca(np.asarray(target.points))
        if pca is None:
            self._mesh_cache[family] = None
            return None

        bundle = {
            "mesh_file": str(mesh_file),
            "target": target,
            "model_basis": np.asarray(pca["R"], dtype=np.float64),
        }
        self._mesh_cache[family] = bundle
        return bundle

    def _candidate_icp_inits(self, mesh_bundle: Dict, pca_pose: Dict):
        scene_R = np.asarray(pca_pose["R"], dtype=np.float64)
        scene_t = np.asarray(pca_pose["t"], dtype=np.float64)
        model_basis = np.asarray(mesh_bundle["model_basis"], dtype=np.float64)
        base_R = scene_R @ model_basis.T
        inits = []
        for yaw_deg in (0.0, 90.0, 180.0, 270.0):
            yaw = np.deg2rad(yaw_deg)
            Rz = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            R = scene_R @ Rz @ model_basis.T
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = scene_t
            inits.append(T)
        base_T = np.eye(4, dtype=np.float64)
        base_T[:3, :3] = base_R
        base_T[:3, 3] = scene_t
        inits.append(base_T)
        return inits

    def _run_icp(self, mesh_bundle: Dict, scene_cloud: o3d.geometry.PointCloud, init_transforms):
        if scene_cloud is None or len(scene_cloud.points) < self._min_points:
            return None
        target = mesh_bundle["target"]
        best_T = None
        best_score = None

        scene = scene_cloud
        if len(scene.points) > 3000:
            scene = scene.voxel_down_sample(self._voxel_size_m)
            scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.015, 3.0 * self._voxel_size_m), max_nn=30))

        for init in init_transforms:
            try:
                reg1 = o3d.pipelines.registration.registration_icp(
                    target,
                    scene,
                    0.035,
                    np.asarray(init, dtype=np.float64),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
                reg2 = o3d.pipelines.registration.registration_icp(
                    target,
                    scene,
                    0.015,
                    reg1.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
            except Exception:
                continue

            fitness = float(reg2.fitness)
            rmse = float(reg2.inlier_rmse)
            score = (fitness, -rmse)
            if best_score is None or score > best_score:
                best_score = score
                best_T = np.asarray(reg2.transformation, dtype=np.float64)

        if best_T is None:
            return None
        if best_score[0] < 0.05:
            return None
        return best_T

    def _find_mesh_file(self, asset_dir: Path) -> Optional[Path]:
        for pattern in ("*.glb", "*.obj", "*.ply", "*.stl", "*.dae"):
            hits = sorted(asset_dir.glob(pattern))
            if hits:
                return hits[0]
        # Also search one level deeper for meshes referenced beside model.sdf.
        for pattern in ("**/*.glb", "**/*.obj", "**/*.ply", "**/*.stl", "**/*.dae"):
            hits = sorted(asset_dir.glob(pattern))
            if hits:
                return hits[0]
        return None

    def _pose_dict_to_T(self, pose: Dict) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        if "R" in pose and pose["R"] is not None:
            R = np.asarray(pose["R"], dtype=np.float64).reshape(3, 3)
        elif "q" in pose and pose["q"] is not None:
            R = self._quaternion_to_matrix(np.asarray(pose["q"], dtype=np.float64).reshape(4)).astype(np.float64)
        else:
            raise KeyError("Pose dictionary must contain either 'R' or 'q'")
        T[:3, :3] = R
        T[:3, 3] = np.asarray(pose["t"], dtype=np.float64).reshape(3)
        return T

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = q / n
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def _matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(R))
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float64)
        qn = np.linalg.norm(q)
        if qn < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        q = q / qn
        if q[3] < 0.0:
            q = -q
        return q.astype(np.float32)
