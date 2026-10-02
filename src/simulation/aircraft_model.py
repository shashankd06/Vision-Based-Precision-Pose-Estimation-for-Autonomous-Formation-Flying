"""
Aircraft 3D Structural Model and Occlusion Geometry.
Defines the 14 structural keypoints on the leader aircraft and ray-casting occlusion logic.
"""
from typing import List, Tuple
import numpy as np
from src.config import AIRCRAFT_KEYPOINTS_3D, KEYPOINT_NAMES, AIRCRAFT_WIREFRAME_EDGES


class AircraftModel:
    def __init__(self, keypoints: np.ndarray = None):
        self.keypoints_body = keypoints if keypoints is not None else np.copy(AIRCRAFT_KEYPOINTS_3D)
        self.num_keypoints = len(self.keypoints_body)
        self.names = KEYPOINT_NAMES
        self.edges = AIRCRAFT_WIREFRAME_EDGES

        # Occlusion collision geometry in aircraft body frame
        # Fuselage modelled as an ellipsoid centered at (0, 0, 0)
        # Semi-axes: a (x length/2) = 5.5m, b (y width/2) = 0.9m, c (z height/2) = 0.8m
        self.fuselage_radii = np.array([5.2, 0.9, 0.8], dtype=np.float64)

        # Wings modelled as planar bounding boxes in body frame
        # Wing thickness in z is ~0.15m
        self.wing_z_half = 0.12
        self.wing_x_range = (-2.0, 1.8)
        self.wing_y_range = (-5.2, 5.2)

        # Vertical tail fin bounding box
        self.tail_x_range = (-5.0, -3.2)
        self.tail_y_half = 0.15
        self.tail_z_range = (0.2, 2.3)

    def get_keypoints(self) -> np.ndarray:
        """Returns (14, 3) keypoint coordinates in body frame."""
        return self.keypoints_body

    def check_occlusion_in_body_frame(self, cam_in_body: np.ndarray, pt_idx: int) -> bool:
        """
        Check if ray from follower camera to keypoint `pt_idx` is occluded by aircraft structure.
        Both cam_in_body and keypoints are in the leader body frame.
        """
        target_pt = self.keypoints_body[pt_idx]
        ray_dir = target_pt - cam_in_body
        dist_to_target = np.linalg.norm(ray_dir)
        if dist_to_target < 1e-4:
            return False

        ray_unit = ray_dir / dist_to_target

        # Self-occlusion heuristic:
        # Keypoints on the far side of the fuselage or wing surface are blocked
        # 1. Fuselage ray-ellipsoid intersection test:
        # Transform ray into normalized sphere coordinates
        p0 = cam_in_body / self.fuselage_radii
        d = ray_unit / self.fuselage_radii

        A = np.dot(d, d)
        B = 2.0 * np.dot(p0, d)
        C = np.dot(p0, p0) - 1.0

        disc = B * B - 4.0 * A * C
        if disc > 0:
            t1 = (-B - np.sqrt(disc)) / (2.0 * A)
            t2 = (-B + np.sqrt(disc)) / (2.0 * A)
            # Find closest intersection distance along physical ray
            for t_norm in (t1, t2):
                if t_norm > 0:
                    hit_body = cam_in_body + t_norm * ray_unit
                    dist_to_hit = np.linalg.norm(hit_body - cam_in_body)
                    # If hit occurs before reaching the target (with margin for surface points)
                    if 0.5 < dist_to_hit < (dist_to_target - 0.35):
                        return True

        # 2. Wing planar occlusion:
        # If camera is looking from below (+z look angle) or above, opposite features may be blocked
        # Check if ray passes through wing plane (z = 0) between cam and target
        if cam_in_body[2] * target_pt[2] < 0:
            # Ray crosses z=0 plane
            t_wing = -cam_in_body[2] / ray_unit[2]
            if 0 < t_wing < (dist_to_target - 0.2):
                x_cross = cam_in_body[0] + t_wing * ray_unit[0]
                y_cross = cam_in_body[1] + t_wing * ray_unit[1]
                if (self.wing_x_range[0] <= x_cross <= self.wing_x_range[1] and
                    self.wing_y_range[0] <= y_cross <= self.wing_y_range[1]):
                    # Check if target is not on the wing itself
                    if pt_idx not in (2, 3, 4, 5, 6, 7):
                        return True

        return False
