"""
Unit tests for Particle Filter and PnP Refiner.
"""
import unittest
import numpy as np
from src.simulation.camera import GimbaledCamera
from src.filtering.particle_filter import RelativePoseParticleFilter
from src.filtering.pnp_refiner import PnPPoseRefiner
from src.dataset.pose_discretizer import PoseDiscretizer


class TestFilterAndRefiner(unittest.TestCase):
    def setUp(self):
        self.camera = GimbaledCamera()
        self.discretizer = PoseDiscretizer()
        self.pf = RelativePoseParticleFilter(num_particles=200, camera=self.camera, discretizer=self.discretizer)
        self.pnp = PnPPoseRefiner(camera=self.camera)

    def test_particle_filter_step(self):
        # Ground truth pose: x=50, y=5, z=2, phi=10
        true_x, true_y, true_z, true_phi = 50.0, 5.0, 2.0, 10.0
        uv, occlusions, _ = self.camera.project_pose(true_x, true_y, true_z, true_phi, noise_std=1.0)
        label = self.discretizer.continuous_to_label(true_x, true_y, true_z, true_phi)

        self.pf.initialize_from_ann_label(label)
        est = self.pf.step(uv, occlusions, ann_label=label, follower_vx=4.0)

        self.assertEqual(len(est), 6)
        # Position estimate should be finite and within plausible bounds
        self.assertTrue(np.all(np.isfinite(est)))
        self.assertLess(abs(est[0] - true_x), 30.0)

    def test_pnp_refiner_convergence(self):
        # Ground truth pose
        true_x, true_y, true_z, true_phi = 45.0, -8.0, 3.0, 12.0
        uv, occlusions, _ = self.camera.project_pose(true_x, true_y, true_z, true_phi, noise_std=0.5)

        # Follower camera gimbal orientation
        rel_pos = np.array([true_x, true_y, true_z])
        R_cam = self.camera.compute_camera_frame_rotation(rel_pos)

        # Perturbed coarse estimate (e.g. from adjacent bin: +/- 5m error)
        coarse_pose = np.array([true_x + 5.0, true_y - 3.0, true_z + 1.5, true_phi + 5.0])

        refined_pose, rmse = self.pnp.refine_pose(uv, occlusions, coarse_pose, R_cam=R_cam)

        # Refined position should be very close to ground truth (< 1.0m)
        pos_error = np.linalg.norm(refined_pose[:3] - np.array([true_x, true_y, true_z]))
        self.assertLess(pos_error, 1.0, f"PnP position error was {pos_error}m, expected < 1.0m")
        self.assertLess(abs(refined_pose[3] - true_phi), 3.0, "PnP roll error should be < 3 deg")
        self.assertLess(rmse, 2.0, f"Reprojection RMSE was {rmse} px, expected < 2.0")


if __name__ == "__main__":
    unittest.main()
