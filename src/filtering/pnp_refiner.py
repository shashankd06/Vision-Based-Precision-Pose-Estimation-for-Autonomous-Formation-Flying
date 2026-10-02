"""
Continuous Pose Refinement using Perspective-n-Point (PnP) and Levenberg-Marquardt.
Proposed Key Enhancement over Punnoose's baseline to overcome discrete grid coarseness.
"""
from typing import Tuple, Optional
import cv2
import numpy as np

from src.config import CAMERA_MATRIX, DISTORTION_COEFFS
from src.simulation.aircraft_model import AircraftModel
from src.simulation.camera import GimbaledCamera, euler_to_rotation_matrix


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Extract aerospace Euler angles [roll_deg, pitch_deg, yaw_deg] from rotation matrix."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


class PnPPoseRefiner:
    def __init__(self,
                 aircraft_model: Optional[AircraftModel] = None,
                 camera: Optional[GimbaledCamera] = None):
        self.aircraft = aircraft_model if aircraft_model is not None else AircraftModel()
        self.camera = camera if camera is not None else GimbaledCamera()
        self.K = self.camera.K
        self.dist_coeffs = DISTORTION_COEFFS
        self.keypoints_3d = self.aircraft.get_keypoints()  # (14, 3)

    def refine_pose(self,
                    detected_uv: np.ndarray,
                    occlusions: np.ndarray,
                    coarse_pose: np.ndarray,
                    R_cam: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Refine pose using Levenberg-Marquardt PnP warm-started by coarse ANN prediction.
        
        Args:
            detected_uv: (14, 2) detected pixel coordinates
            occlusions: (14,) boolean occlusion flags
            coarse_pose: [x_c, y_c, z_c, phi_c] from ANN classifier or PF
            R_cam: Optional (3, 3) follower camera orientation matrix.
                   If None, estimated from coarse_pose line-of-sight.
            
        Returns:
            refined_pose: [x, y, z, roll_deg, pitch_deg, yaw_deg]
            reprojection_rmse: float pixel error
        """
        visible_mask = ~occlusions
        num_visible = int(np.sum(visible_mask))

        # Fallback if insufficient points for PnP
        if num_visible < 4:
            x_c, y_c, z_c, phi_c = coarse_pose[:4]
            return np.array([x_c, y_c, z_c, phi_c, 0.0, 0.0], dtype=np.float64), 999.0

        obj_pts = self.keypoints_3d[visible_mask].astype(np.float64)  # (K, 3)
        img_pts = detected_uv[visible_mask].astype(np.float64)        # (K, 2)

        # 1. Warm start guess from coarse pose
        x_c, y_c, z_c, phi_c = coarse_pose[:4]
        rel_pos_coarse = np.array([x_c, y_c, z_c], dtype=np.float64)

        if R_cam is None:
            R_cam = self.camera.compute_camera_frame_rotation(rel_pos_coarse)

        R_lead_init = euler_to_rotation_matrix(np.radians(phi_c), 0.0, 0.0)

        # In camera frame: P_c = R_total * P_body + t_cam
        R_total_init = R_cam @ R_lead_init
        t_cam_init = (R_cam @ rel_pos_coarse.reshape(3, 1)).astype(np.float64)
        rvec_init, _ = cv2.Rodrigues(R_total_init)

        try:
            # 2. Levenberg-Marquardt non-linear iterative optimization
            success, rvec_opt, tvec_opt = cv2.solvePnP(
                objectPoints=obj_pts,
                imagePoints=img_pts,
                cameraMatrix=self.K,
                distCoeffs=self.dist_coeffs,
                rvec=rvec_init.copy(),
                tvec=t_cam_init.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return np.array([x_c, y_c, z_c, phi_c, 0.0, 0.0], dtype=np.float64), 999.0

            # 3. Transform refined solution back to follower body coordinate frame
            R_total_opt, _ = cv2.Rodrigues(rvec_opt)
            t_cam_opt = tvec_opt.flatten()

            # Translation in follower frame: rel_pos = R_cam^T * t_cam
            rel_pos_refined = R_cam.T @ t_cam_opt

            # Orientation in follower frame: R_lead = R_cam^T * R_total
            R_lead_refined = R_cam.T @ R_total_opt
            euler_refined = rotation_matrix_to_euler(R_lead_refined)

            # Compute reprojection error
            proj_pts, _ = cv2.projectPoints(obj_pts, rvec_opt, tvec_opt, self.K, self.dist_coeffs)
            proj_pts = proj_pts.reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.sum((img_pts - proj_pts) ** 2, axis=1))))

            refined_pose = np.array([
                rel_pos_refined[0],
                rel_pos_refined[1],
                rel_pos_refined[2],
                euler_refined[0],  # roll
                euler_refined[1],  # pitch
                euler_refined[2]   # yaw
            ], dtype=np.float64)

            return refined_pose, rmse

        except Exception:
            return np.array([x_c, y_c, z_c, phi_c, 0.0, 0.0], dtype=np.float64), 999.0
