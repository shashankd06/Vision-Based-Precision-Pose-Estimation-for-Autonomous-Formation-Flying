"""
Unit tests for PoseClassifierNet architecture and forward pass.
"""
import unittest
import torch
import numpy as np
from src.models.ann_classifier import PoseClassifierNet
from src.config import FEATURE_DIM, TOTAL_DISCRETE_LABELS


class TestANNClassifier(unittest.TestCase):
    def setUp(self):
        self.model = PoseClassifierNet(
            input_dim=FEATURE_DIM,
            hidden_dim=100,
            output_dim=TOTAL_DISCRETE_LABELS
        )

    def test_forward_pass_shape(self):
        batch_size = 8
        dummy_input = torch.randn(batch_size, FEATURE_DIM)
        logits = self.model(dummy_input)
        self.assertEqual(logits.shape, (batch_size, TOTAL_DISCRETE_LABELS))

    def test_probabilities_sum(self):
        dummy_input = torch.randn(4, FEATURE_DIM)
        probs = self.model.get_probabilities(dummy_input)
        sums = torch.sum(probs, dim=-1).detach().numpy()
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

    def test_predict_label(self):
        sample = np.zeros(FEATURE_DIM, dtype=np.float32)
        pred_label, conf = self.model.predict_label(sample)
        self.assertTrue(0 <= pred_label < TOTAL_DISCRETE_LABELS)
        self.assertTrue(0.0 <= conf <= 1.0)


if __name__ == "__main__":
    unittest.main()
