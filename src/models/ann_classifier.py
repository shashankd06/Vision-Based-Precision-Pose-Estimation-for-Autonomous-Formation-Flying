"""
Coarse Pose Classification Artificial Neural Network (Punnoose Section IV-D).
Architecture: 42 inputs -> 100 ReLU -> 100 ReLU -> 4800 Softmax.
"""
from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import FEATURE_DIM, TOTAL_DISCRETE_LABELS
from src.dataset.pose_discretizer import PoseDiscretizer


class PoseClassifierNet(nn.Module):
    def __init__(self,
                 input_dim: int = FEATURE_DIM,
                 hidden_dim: int = 100,
                 output_dim: int = TOTAL_DISCRETE_LABELS):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 2 hidden layers with 100 neurons each, matching paper Section IV-D
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        return logits

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 4800-way softmax belief distribution."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_label(self, x_np: np.ndarray) -> Tuple[int, float]:
        """
        Inference on single 42-D feature vector.
        Returns: (predicted_label, confidence_probability)
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x_np).float().unsqueeze(0)
            probs = self.get_probabilities(tensor).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
        return pred_idx, conf

    def predict_batch(self, x_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference on batch of features (B, 42).
        Returns: (predicted_labels (B,), confidences (B,))
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(x_batch).float()
            probs = self.get_probabilities(tensor).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)
        return preds, confs


def compute_metrics(model: PoseClassifierNet,
                    X: np.ndarray,
                    y: np.ndarray,
                    poses_true: np.ndarray,
                    discretizer: Optional[PoseDiscretizer] = None,
                    batch_size: int = 512) -> Dict[str, float]:
    """
    Compute comprehensive accuracy, adjacent bin accuracy, and physical pose errors.
    """
    if discretizer is None:
        discretizer = PoseDiscretizer()

    model.eval()
    all_preds = []
    num_samples = len(X)

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = torch.from_numpy(X[start_idx:end_idx]).float()
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.append(preds)

    preds = np.concatenate(all_preds)

    # 1. Exact top-1 bin accuracy
    exact_acc = float(np.mean(preds == y)) * 100.0

    # 2. Adjacent bin accuracy (within +/- 1 bin in each of x, y, z, phi)
    adj_correct = 0
    for p, t in zip(preds, y):
        ix_p, iy_p, iz_p, iphi_p = discretizer.label_to_bin_indices(p)
        ix_t, iy_t, iz_t, iphi_t = discretizer.label_to_bin_indices(t)
        if (abs(ix_p - ix_t) <= 1 and
            abs(iy_p - iy_t) <= 1 and
            abs(iz_p - iz_t) <= 1 and
            abs(iphi_p - iphi_t) <= 1):
            adj_correct += 1
    adj_acc = float(adj_correct / num_samples) * 100.0

    # 3. Physical pose errors (center of predicted bin vs true continuous pose)
    pred_centers = discretizer.batch_labels_to_pose_centers(preds)  # (N, 4)
    pos_err = np.linalg.norm(pred_centers[:, :3] - poses_true[:, :3], axis=1)  # meters
    att_err = np.abs(pred_centers[:, 3] - poses_true[:, 3])  # degrees

    return {
        "top1_accuracy_percent": exact_acc,
        "adjacent_bin_accuracy_percent": adj_acc,
        "mean_position_error_m": float(np.mean(pos_err)),
        "median_position_error_m": float(np.median(pos_err)),
        "rmse_position_error_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "mean_attitude_error_deg": float(np.mean(att_err)),
        "rmse_attitude_error_deg": float(np.sqrt(np.mean(att_err ** 2)))
    }


def save_model(model: PoseClassifierNet, filepath: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'output_dim': model.output_dim
    }, filepath)


def load_model(filepath: str) -> PoseClassifierNet:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    model = PoseClassifierNet(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
