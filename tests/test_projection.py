"""
Unit tests for Camera Projection, Occlusion Detection, and Keypoint Features.
"""
import unittest
import numpy as np
from src.simulation.camera import GimbaledCamera
from src.simulation.aircraft_model import AircraftModel
from src.config import NUM_FEATURES, FEATURE_DIM


class TestCameraProjection(unittest.TestCase):
    def setUp(self):
        self.aircraft = AircraftModel()
        self.camera = GimbaledCamera(aircraft_model=self.aircraft)

    def test_keypoints_shape(self):
        pts = self.aircraft.get_keypoints()
        self.assertEqual(pts.shape, (14, 3))

    def test_projection_output(self):
        # Leader aircraft at 50m forward, 5m right, 2m up, 10 deg roll
        uv, occlusions, pts_cam = self.camera.project_pose(
            x=50.0, y=5.0, z=2.0, phi_deg=10.0, noise_std=0.0
        )
        self.assertEqual(uv.shape, (NUM_FEATURES, 2))
        self.assertEqual(occlusions.shape, (NUM_FEATURES,))
        self.assertEqual(pts_cam.shape, (NUM_FEATURES, 3))

        # Optical axis is centered, leader center is around (320, 240)
        # All visible points should have coordinates near center
        visible_indices = np.where(~occlusions)[0]
        self.assertGreater(len(visible_indices), 5, "At least some keypoints must be visible")

        for idx in visible_indices:
            u, v = uv[idx]
            self.assertGreaterEqual(u, 0)
            self.assertLess(u, self.camera.width)
            self.assertGreaterEqual(v, 0)
            self.assertLess(v, self.camera.height)

    def test_feature_vector_shape(self):
        feat = self.camera.extract_feature_vector(
            x=60.0, y=0.0, z=0.0, phi_deg=0.0
        )
        self.assertEqual(feat.shape, (FEATURE_DIM,))
        self.assertEqual(len(feat), 42)


if __name__ == "__main__":
    unittest.main()
