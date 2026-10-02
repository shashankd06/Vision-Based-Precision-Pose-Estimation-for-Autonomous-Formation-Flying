"""
Monocular Gimbaled Camera Simulation and Feature Extraction.
Projects 3D aircraft keypoints to 2D image coordinates and computes 42-D feature vector.
"""
from typing import Tuple, Optional
import numpy as np
from src.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_MATRIX,
    FOCAL_LENGTH_X, FOCAL_LENGTH_Y, PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y,
    NUM_FEATURES, FEATURE_DIM
)
from src.simulation.aircraft_model import AircraftModel


def euler_to_rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """
    Compute 3x3 rotation matrix for aerospace Euler angles: Yaw(psi) * Pitch(theta) * Roll(phi).
    x: forward, y: right, z: down (or x: forward, y: right, z: up).
    Using right-handed body frame (x forward, y right, z up):
    """
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


class GimbaledCamera:
    def __init__(self,
                 aircraft_model: Optional[AircraftModel] = None,
                 width: int = CAMERA_WIDTH,
                 height: int = CAMERA_HEIGHT,
                 camera_matrix: np.ndarray = CAMERA_MATRIX):
        self.aircraft = aircraft_model if aircraft_model is not None else AircraftModel()
        self.width = width
        self.height = height
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

    def compute_camera_frame_rotation(self, rel_pos: np.ndarray) -> np.ndarray:
        """
        Compute rotation from follower body frame to gimbaled camera frame.
        rel_pos: [x, y, z] relative position of leader w.r.t follower.
        Optical axis (+Z_cam) points directly along line-of-sight to leader.
        OpenCV camera convention: +X_cam right, +Y_cam down, +Z_cam forward.
        """
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-4:
            z_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            z_cam = rel_pos / dist

        # Follower up vector in body frame is [0, 0, 1]
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # In case line of sight is purely vertical
        if np.abs(np.dot(z_cam, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        # OpenCV +X_cam points right: cross(up, z_cam)
        x_cam = np.cross(up, z_cam)
        x_norm = np.linalg.norm(x_cam)
        if x_norm < 1e-4:
            x_cam = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            x_cam = x_cam / x_norm

        # OpenCV +Y_cam points down: cross(z_cam, x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)

        # R_cam maps vector in follower frame to camera frame coordinates
        R_cam = np.vstack([x_cam, y_cam, z_cam])
        return R_cam

    def project_pose(self,
                     x: float, y: float, z: float,
                     phi_deg: float,
                     pitch_deg: float = 0.0,
                     yaw_deg: float = 0.0,
                     noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project leader aircraft 3D keypoints into follower's 2D camera view.
        
        Args:
            x, y, z: Relative position of leader [meters]
            phi_deg: Relative roll angle [degrees]
            pitch_deg: Relative pitch angle [degrees]
            yaw_deg: Relative yaw angle [degrees]
            noise_std: Gaussian pixel noise standard deviation (e.g. 1.5 pixels)

        Returns:
            uv: (14, 2) pixel coordinates
            occlusions: (14,) boolean occlusion flags (True if occluded)
            pts_cam: (14, 3) 3D keypoints in camera coordinate frame
        """
        rel_pos = np.array([x, y, z], dtype=np.float64)
        R_lead = euler_to_rotation_matrix(np.radians(phi_deg), np.radians(pitch_deg), np.radians(yaw_deg))
        R_cam = self.compute_camera_frame_rotation(rel_pos)

        keypoints_body = self.aircraft.get_keypoints()  # (14, 3)

        # 3D points in follower frame: P_F = rel_pos + R_lead * P_body
        pts_follower = rel_pos + (R_lead @ keypoints_body.T).T  # (14, 3)

        # 3D points in camera frame: P_C = R_cam * P_F
        pts_cam = (R_cam @ pts_follower.T).T  # (14, 3)

        # Camera position in leader body frame for ray-tracing occlusions
        cam_in_body = R_lead.T @ (-rel_pos)

        uv = np.zeros((self.aircraft.num_keypoints, 2), dtype=np.float64)
        occlusions = np.zeros(self.aircraft.num_keypoints, dtype=bool)

        for i in range(self.aircraft.num_keypoints):
            Xc, Yc, Zc = pts_cam[i]
            if Zc <= 0.5:
                # Behind camera or too close
                occlusions[i] = True
                continue

            # Pinhole projection
            u = self.fx * (Xc / Zc) + self.cx
            v = self.fy * (Yc / Zc) + self.cy

            # Check image boundaries
            if u < 0 or u >= self.width or v < 0 or v >= self.height:
                occlusions[i] = True
            else:
                # Check geometric ray-tracing occlusion
                if self.aircraft.check_occlusion_in_body_frame(cam_in_body, i):
                    occlusions[i] = True

            # Add pixel noise to detected (visible) features
            if noise_std > 0 and not occlusions[i]:
                u += np.random.normal(0.0, noise_std)
                v += np.random.normal(0.0, noise_std)

            uv[i] = [u, v]

        return uv, occlusions, pts_cam

    def extract_feature_vector(self,
                               x: float, y: float, z: float,
                               phi_deg: float,
                               pitch_deg: float = 0.0,
                               yaw_deg: float = 0.0,
                               noise_std: float = 0.0,
                               normalize: bool = True) -> np.ndarray:
        """
        Extract 42-D feature vector: 14 keypoints * [u, v, occlusion_flag].
        If normalize=True, u and v are normalized to [-1.0, 1.0].
        If a feature is occluded, (u, v) is set to 0.0 and occlusion_flag is 1.0.
        """
        uv, occlusions, _ = self.project_pose(x, y, z, phi_deg, pitch_deg, yaw_deg, noise_std)
        feat = np.zeros((NUM_FEATURES, 3), dtype=np.float32)

        for i in range(NUM_FEATURES):
            if occlusions[i]:
                feat[i] = [0.0, 0.0, 1.0]  # Occluded
            else:
                u, v = uv[i]
                if normalize:
                    u_norm = (u - self.cx) / (self.width / 2.0)
                    v_norm = (v - self.cy) / (self.height / 2.0)
                    feat[i] = [u_norm, v_norm, 0.0]  # Visible
                else:
                    feat[i] = [u, v, 0.0]

        return feat.flatten()  # (42,)
