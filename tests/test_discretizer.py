"""
Unit tests for 4800-bin 4D pose discretizer.
"""
import unittest
import numpy as np
from src.dataset.pose_discretizer import PoseDiscretizer
from src.config import TOTAL_DISCRETE_LABELS


class TestPoseDiscretizer(unittest.TestCase):
    def setUp(self):
        self.discretizer = PoseDiscretizer()

    def test_total_labels(self):
        self.assertEqual(self.discretizer.total_labels, 4800)

    def test_round_trip_mapping(self):
        # Test various continuous poses
        test_poses = [
            (-75.0, 0.0, 0.0, 0.0),
            (-95.0, -40.0, -15.0, -35.0),
            (40.0, 45.0, 18.0, 40.0),
            (0.0, 0.0, 0.0, 0.0),
            (-50.0, 20.0, -5.0, 15.0),
        ]
        for x, y, z, phi in test_poses:
            label = self.discretizer.continuous_to_label(x, y, z, phi)
            self.assertTrue(0 <= label < TOTAL_DISCRETE_LABELS)

            ix, iy, iz, iphi = self.discretizer.label_to_bin_indices(label)
            re_label = self.discretizer.bin_indices_to_label(ix, iy, iz, iphi)
            self.assertEqual(label, re_label)

            center = self.discretizer.label_to_pose_center(label)
            # Center should be within one bin width of original
            self.assertLessEqual(abs(center[0] - x), self.discretizer.dx)
            self.assertLessEqual(abs(center[1] - y), self.discretizer.dy)
            self.assertLessEqual(abs(center[2] - z), self.discretizer.dz)
            self.assertLessEqual(abs(center[3] - phi), self.discretizer.dphi)

    def test_batch_mapping(self):
        batch = np.array([
            [-75.0, 0.0, 0.0, 0.0],
            [-20.0, 10.0, 5.0, -15.0],
            [30.0, -25.0, -10.0, 25.0]
        ])
        labels = self.discretizer.batch_continuous_to_labels(batch)
        self.assertEqual(len(labels), 3)
        centers = self.discretizer.batch_labels_to_pose_centers(labels)
        self.assertEqual(centers.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
