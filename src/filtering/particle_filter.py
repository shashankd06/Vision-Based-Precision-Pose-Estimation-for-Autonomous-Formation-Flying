"""
Bayesian Particle Filter for Relative Pose Estimation (Punnoose Section IV-E).
Propagates particles through flight kinematics, updates weights using inverse feature MSE,
and performs hybrid resampling with ANN pose reinjection.
"""
from typing import Tuple, List, Optional
import numpy as np

from src.config import (
    NUM_FEATURES, PF_NUM_PARTICLES_DEFAULT, PF_ALPHA_DEFAULT,
    PF_PROCESS_NOISE_POS, PF_PROCESS_NOISE_ATT, PF_LIKELIHOOD_EPSILON
)
from src.simulation.camera import GimbaledCamera, euler_to_rotation_matrix
from src.dataset.pose_discretizer import PoseDiscretizer


class RelativePoseParticleFilter:
    def __init__(self,
                 num_particles: int = PF_NUM_PARTICLES_DEFAULT,
                 alpha: float = PF_ALPHA_DEFAULT,
                 dt: float = 0.1,
                 camera: Optional[GimbaledCamera] = None,
                 discretizer: Optional[PoseDiscretizer] = None):
        self.N = num_particles
        self.alpha = alpha  # Fraction of particles resampled from posterior (e.g. 0.9)
        self.dt = dt
        self.camera = camera if camera is not None else GimbaledCamera()
        self.discretizer = discretizer if discretizer is not None else PoseDiscretizer()

        # Particle states: (N, 6) -> [x, y, z, roll_deg, pitch_deg, yaw_deg]
        self.particles = np.zeros((self.N, 6), dtype=np.float64)
        self.weights = np.ones(self.N, dtype=np.float64) / self.N
        self.is_initialized = False

        # Empirical error distribution standard deviations derived from validation set
        self.empirical_error_std = np.array([7.5, 5.0, 2.5, 7.5], dtype=np.float64)

    def initialize_from_ann_label(self,
                                  ann_label: int,
                                  prior_noise_scale: float = 1.0):
        """
        Initialize N particles around ANN coarse pose label perturbed by empirical error distribution.
        (Paper Section IV-E, Step 1).
        """
        center = self.discretizer.label_to_pose_center(ann_label)  # [x, y, z, phi_deg]
        x_c, y_c, z_c, phi_c = center

        noise_std = self.empirical_error_std * prior_noise_scale

        self.particles[:, 0] = x_c + np.random.normal(0, noise_std[0], size=self.N)
        self.particles[:, 1] = y_c + np.random.normal(0, noise_std[1], size=self.N)
        self.particles[:, 2] = z_c + np.random.normal(0, noise_std[2], size=self.N)
        self.particles[:, 3] = phi_c + np.random.normal(0, noise_std[3], size=self.N)
        self.particles[:, 4] = np.random.normal(0, 3.0, size=self.N)  # Pitch [deg]
        self.particles[:, 5] = np.random.normal(0, 3.0, size=self.N)  # Yaw [deg]

        self.weights = np.ones(self.N, dtype=np.float64) / self.N
        self.is_initialized = True

    def predict(self, follower_vx: float = 4.0):
        """
        Propagate particles through aircraft relative dynamics with unknown leader disturbance.
        (Paper Section IV-E, Step 2).
        """
        # Leader velocity treated as uniform disturbance [-1.0, 1.0] m/s
        leader_vx_dist = np.random.uniform(-1.0, 1.0, size=self.N)
        rel_vx = (leader_vx_dist - follower_vx)  # Relative forward velocity

        # Update positions
        self.particles[:, 0] += rel_vx * self.dt + np.random.normal(0, PF_PROCESS_NOISE_POS, size=self.N)
        self.particles[:, 1] += np.random.normal(0, PF_PROCESS_NOISE_POS, size=self.N)
        self.particles[:, 2] += np.random.normal(0, PF_PROCESS_NOISE_POS * 0.5, size=self.N)

        # Update angles with random walk / disturbance
        self.particles[:, 3] += np.random.normal(0, np.degrees(PF_PROCESS_NOISE_ATT), size=self.N)
        self.particles[:, 4] += np.random.normal(0, 0.8, size=self.N)
        self.particles[:, 5] += np.random.normal(0, 0.8, size=self.N)

    def update_weights(self, detected_uv: np.ndarray, occlusions: np.ndarray):
        """
        Compute measurement likelihood: wi = 1 / (eps + MSE(actual features, particle synthetic features)).
        Occluded features are ignored.
        (Paper Section IV-E, Step 3).
        """
        visible_mask = ~occlusions
        num_visible = np.sum(visible_mask)

        if num_visible < 3:
            # Low visibility fallback
            return

        vis_actual = detected_uv[visible_mask]  # (K, 2)
        raw_weights = np.zeros(self.N, dtype=np.float64)

        for i in range(self.N):
            p = self.particles[i]
            # Project particle state to synthetic image features (no pixel noise, fast check)
            synth_uv, _, _ = self.camera.project_pose(
                x=p[0], y=p[1], z=p[2],
                phi_deg=p[3], pitch_deg=p[4], yaw_deg=p[5],
                noise_std=0.0
            )
            vis_synth = synth_uv[visible_mask]
            # Mean squared error in pixel space
            mse = np.mean(np.sum((vis_actual - vis_synth) ** 2, axis=1))
            raw_weights[i] = 1.0 / (mse + PF_LIKELIHOOD_EPSILON)

        weight_sum = np.sum(raw_weights)
        if weight_sum > 1e-12:
            self.weights = raw_weights / weight_sum
        else:
            self.weights = np.ones(self.N, dtype=np.float64) / self.N

    def resample(self, ann_label: Optional[int] = None):
        """
        Hybrid resampling: floor(alpha * N) from likelihood posterior,
        and N - floor(alpha * N) re-injected from ANN classified pose + error distribution.
        (Paper Section IV-E, Step 4).
        """
        num_resampled = int(np.floor(self.alpha * self.N))
        num_injected = self.N - num_resampled

        new_particles = np.zeros_like(self.particles)

        # 1. Systematic resampling for the posterior portion
        if num_resampled > 0:
            indices = np.random.choice(self.N, size=num_resampled, p=self.weights, replace=True)
            new_particles[:num_resampled] = self.particles[indices]

        # 2. ANN Re-injection
        if num_injected > 0 and ann_label is not None:
            center = self.discretizer.label_to_pose_center(ann_label)
            x_c, y_c, z_c, phi_c = center
            new_particles[num_resampled:, 0] = x_c + np.random.normal(0, self.empirical_error_std[0], size=num_injected)
            new_particles[num_resampled:, 1] = y_c + np.random.normal(0, self.empirical_error_std[1], size=num_injected)
            new_particles[num_resampled:, 2] = z_c + np.random.normal(0, self.empirical_error_std[2], size=num_injected)
            new_particles[num_resampled:, 3] = phi_c + np.random.normal(0, self.empirical_error_std[3], size=num_injected)
            new_particles[num_resampled:, 4] = np.random.normal(0, 3.0, size=num_injected)
            new_particles[num_resampled:, 5] = np.random.normal(0, 3.0, size=num_injected)

        self.particles = new_particles
        self.weights = np.ones(self.N, dtype=np.float64) / self.N

    def step(self,
             detected_uv: np.ndarray,
             occlusions: np.ndarray,
             ann_label: Optional[int] = None,
             follower_vx: float = 4.0) -> np.ndarray:
        """Execute one full filtering cycle: Predict -> Update -> Estimate -> Resample."""
        if not self.is_initialized and ann_label is not None:
            self.initialize_from_ann_label(ann_label)

        self.predict(follower_vx=follower_vx)
        self.update_weights(detected_uv, occlusions)
        estimate = self.get_estimate()
        self.resample(ann_label=ann_label)
        return estimate

    def get_estimate(self) -> np.ndarray:
        """Returns weighted mean estimate [x, y, z, roll_deg, pitch_deg, yaw_deg]."""
        return np.average(self.particles, axis=0, weights=self.weights)
