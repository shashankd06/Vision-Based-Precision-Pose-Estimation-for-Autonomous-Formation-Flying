"""
Synthetic Dataset Generator for Coarse Pose Classification.
Generates (42-D feature vector, discrete_label) pairs with camera projection and occlusions.
"""
import os
import argparse
from typing import Tuple
import numpy as np
from src.config import GRID_CONFIG, TOTAL_DISCRETE_LABELS, FEATURE_DIM
from src.simulation.camera import GimbaledCamera
from src.dataset.pose_discretizer import PoseDiscretizer


def sample_random_poses(num_samples: int,
                        config=GRID_CONFIG,
                        seed: int = 42) -> np.ndarray:
    """
    Uniformly sample continuous (x, y, z, phi_deg) poses across the configuration bounds.
    """
    rng = np.random.default_rng(seed)
    # Ensure follower is behind leader or within operational range
    # In table: x in [-100, 50], but follower camera requires leader in front (depth > 0)
    # Forward distance is positive along boresight.
    # To match Punnoose's Table 1, x represents relative displacement in flight line:
    # If follower is trailing, depth is positive. Let's make sure depth along boresight > 5m.
    x = rng.uniform(config['x']['min'], config['x']['max'], size=num_samples)
    # Ensure forward depth is positive for camera viewing: if x < 5, shift or map to positive depth
    # When follower trailing, relative forward position x > 0 (distance to leader)
    # Let's map x such that minimum distance is at least 10m to avoid camera clipping
    x_valid = np.where(np.abs(x) < 10.0, np.sign(x + 1e-4) * 15.0, x)
    # For line-of-sight gimbal, depth is ||rel_pos||, which is always positive as long as rel_pos != 0
    y = rng.uniform(config['y']['min'], config['y']['max'], size=num_samples)
    z = rng.uniform(config['z']['min'], config['z']['max'], size=num_samples)
    phi = rng.uniform(config['phi']['min'], config['phi']['max'], size=num_samples)

    return np.column_stack([x_valid, y, z, phi])


def generate_pose_dataset(num_samples: int,
                          noise_std: float = 1.5,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate dataset of features (N, 42), integer labels (N,), and true poses (N, 4).
    """
    camera = GimbaledCamera()
    discretizer = PoseDiscretizer()
    poses = sample_random_poses(num_samples, seed=seed)

    features = np.zeros((num_samples, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    rng = np.random.default_rng(seed + 1)
    pitch_perturbs = rng.uniform(-4.0, 4.0, size=num_samples)
    yaw_perturbs = rng.uniform(-4.0, 4.0, size=num_samples)

    for i in range(num_samples):
        x, y, z, phi = poses[i]
        pitch = pitch_perturbs[i]
        yaw = yaw_perturbs[i]

        labels[i] = discretizer.continuous_to_label(x, y, z, phi)
        features[i] = camera.extract_feature_vector(
            x=x, y=y, z=z, phi_deg=phi,
            pitch_deg=pitch, yaw_deg=yaw,
            noise_std=noise_std,
            normalize=True
        )

    return features, labels, poses


def create_and_save_datasets(data_dir: str = "data",
                             num_train: int = 40000,
                             num_val: int = 8000,
                             noise_std: float = 1.5):
    """Generate and save train and validation sets to disk."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"Generating {num_train} training samples...")
    X_train, y_train, poses_train = generate_pose_dataset(num_train, noise_std=noise_std, seed=101)

    print(f"Generating {num_val} validation samples...")
    X_val, y_val, poses_val = generate_pose_dataset(num_val, noise_std=noise_std, seed=202)

    np.savez_compressed(
        os.path.join(data_dir, "train_dataset.npz"),
        features=X_train, labels=y_train, poses=poses_train
    )
    np.savez_compressed(
        os.path.join(data_dir, "val_dataset.npz"),
        features=X_val, labels=y_val, poses=poses_val
    )
    print(f"Successfully saved datasets to {data_dir}/ (train: {len(X_train)}, val: {len(X_val)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for pose classification.")
    parser.add_argument("--num_train", type=int, default=40000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=8000, help="Number of validation samples")
    parser.add_argument("--noise_std", type=float, default=1.5, help="Pixel noise std")
    parser.add_argument("--data_dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    create_and_save_datasets(
        data_dir=args.data_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        noise_std=args.noise_std
    )
