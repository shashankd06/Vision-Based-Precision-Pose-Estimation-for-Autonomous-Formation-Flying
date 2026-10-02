"""
4D Pose Space Discretization (Table 1 in Punnoose)
Maps continuous (x, y, z, phi) into 4800 discrete labels and vice-versa.
"""
from typing import Tuple, Union
import numpy as np
from src.config import GRID_CONFIG, TOTAL_DISCRETE_LABELS


class PoseDiscretizer:
    def __init__(self, config=GRID_CONFIG):
        self.config = config
        self.nx = config['x']['n']
        self.ny = config['y']['n']
        self.nz = config['z']['n']
        self.nphi = config['phi']['n']
        self.total_labels = self.nx * self.ny * self.nz * self.nphi
        assert self.total_labels == TOTAL_DISCRETE_LABELS, f"Expected {TOTAL_DISCRETE_LABELS} labels"

        self.dx = (config['x']['max'] - config['x']['min']) / self.nx
        self.dy = (config['y']['max'] - config['y']['min']) / self.ny
        self.dz = (config['z']['max'] - config['z']['min']) / self.nz
        self.dphi = (config['phi']['max'] - config['phi']['min']) / self.nphi

        # Precompute bin centers for instant lookup
        self.x_centers = config['x']['min'] + (np.arange(self.nx) + 0.5) * self.dx
        self.y_centers = config['y']['min'] + (np.arange(self.ny) + 0.5) * self.dy
        self.z_centers = config['z']['min'] + (np.arange(self.nz) + 0.5) * self.dz
        self.phi_centers = config['phi']['min'] + (np.arange(self.nphi) + 0.5) * self.dphi

    def continuous_to_bin_indices(self, x: float, y: float, z: float, phi_deg: float) -> Tuple[int, int, int, int]:
        """Convert continuous values to 4D grid bin indices (ix, iy, iz, iphi)."""
        ix = int(np.clip(np.floor((x - self.config['x']['min']) / self.dx), 0, self.nx - 1))
        iy = int(np.clip(np.floor((y - self.config['y']['min']) / self.dy), 0, self.ny - 1))
        iz = int(np.clip(np.floor((z - self.config['z']['min']) / self.dz), 0, self.nz - 1))
        iphi = int(np.clip(np.floor((phi_deg - self.config['phi']['min']) / self.dphi), 0, self.nphi - 1))
        return ix, iy, iz, iphi

    def bin_indices_to_label(self, ix: int, iy: int, iz: int, iphi: int) -> int:
        """Convert 4D grid indices to flat integer label [0, 4799]."""
        return int(ix * (self.ny * self.nz * self.nphi) + iy * (self.nz * self.nphi) + iz * self.nphi + iphi)

    def label_to_bin_indices(self, label: int) -> Tuple[int, int, int, int]:
        """Convert flat integer label [0, 4799] to 4D grid indices."""
        label = int(np.clip(label, 0, self.total_labels - 1))
        iphi = label % self.nphi
        rem1 = label // self.nphi
        iz = rem1 % self.nz
        rem2 = rem1 // self.nz
        iy = rem2 % self.ny
        ix = rem2 // self.ny
        return ix, iy, iz, iphi

    def continuous_to_label(self, x: float, y: float, z: float, phi_deg: float) -> int:
        """Directly map continuous pose to flat integer label."""
        ix, iy, iz, iphi = self.continuous_to_bin_indices(x, y, z, phi_deg)
        return self.bin_indices_to_label(ix, iy, iz, iphi)

    def label_to_pose_center(self, label: int) -> np.ndarray:
        """Recover center coordinate [x, y, z, phi_deg] of the given discrete label."""
        ix, iy, iz, iphi = self.label_to_bin_indices(label)
        return np.array([
            self.x_centers[ix],
            self.y_centers[iy],
            self.z_centers[iz],
            self.phi_centers[iphi]
        ], dtype=np.float64)

    def batch_continuous_to_labels(self, poses: np.ndarray) -> np.ndarray:
        """
        Convert batch of continuous poses (N, 4) to integer labels (N,).
        poses: array of [x, y, z, phi_deg]
        """
        x, y, z, phi = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
        ix = np.clip(np.floor((x - self.config['x']['min']) / self.dx).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(np.floor((y - self.config['y']['min']) / self.dy).astype(np.int64), 0, self.ny - 1)
        iz = np.clip(np.floor((z - self.config['z']['min']) / self.dz).astype(np.int64), 0, self.nz - 1)
        iphi = np.clip(np.floor((phi - self.config['phi']['min']) / self.dphi).astype(np.int64), 0, self.nphi - 1)
        return ix * (self.ny * self.nz * self.nphi) + iy * (self.nz * self.nphi) + iz * self.nphi + iphi

    def batch_labels_to_pose_centers(self, labels: np.ndarray) -> np.ndarray:
        """Convert batch of integer labels (N,) to center coordinates (N, 4)."""
        labels = np.asarray(labels, dtype=np.int64)
        iphi = labels % self.nphi
        rem1 = labels // self.nphi
        iz = rem1 % self.nz
        rem2 = rem1 // self.nz
        iy = rem2 % self.ny
        ix = rem2 // self.ny
        return np.column_stack([
            self.x_centers[ix],
            self.y_centers[iy],
            self.z_centers[iz],
            self.phi_centers[iphi]
        ])
