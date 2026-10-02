"""
Training Script for Coarse Pose Classification ANN (Punnoose Section IV-D).
Trains 42 -> 100 -> 100 -> 4800 MLP with Adam optimizer and logs evaluation metrics.
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import FEATURE_DIM, TOTAL_DISCRETE_LABELS
from src.dataset.generate_dataset import generate_pose_dataset
from src.dataset.pose_discretizer import PoseDiscretizer
from src.models.ann_classifier import PoseClassifierNet, compute_metrics, save_model


def train_ann(num_train: int = 40000,
              num_val: int = 8000,
              epochs: int = 25,
              batch_size: int = 256,
              lr: float = 1e-3,
              output_dir: str = "data/models",
              plots_dir: str = "docs/figures"):
    """
    Train and evaluate the coarse pose classification neural network.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("  Vision-Based Formation Flying: Coarse Pose Classifier Training")
    print("=" * 70)

    # 1. Dataset Generation / Loading
    train_cache = "data/train_dataset.npz"
    val_cache = "data/val_dataset.npz"

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Loading cached datasets from {train_cache} and {val_cache}...")
        train_data = np.load(train_cache)
        X_train, y_train, poses_train = train_data["features"], train_data["labels"], train_data["poses"]
        val_data = np.load(val_cache)
        X_val, y_val, poses_val = val_data["features"], val_data["labels"], val_data["poses"]
    else:
        print(f"Generating {num_train} training samples and {num_val} validation samples...")
        X_train, y_train, poses_train = generate_pose_dataset(num_train, noise_std=1.5, seed=42)
        X_val, y_val, poses_val = generate_pose_dataset(num_val, noise_std=1.5, seed=123)
        os.makedirs("data", exist_ok=True)
        np.savez_compressed(train_cache, features=X_train, labels=y_train, poses=poses_train)
        np.savez_compressed(val_cache, features=X_val, labels=y_val, poses=poses_val)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
    discretizer = PoseDiscretizer()

    # PyTorch DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model & Optimizer
    model = PoseClassifierNet(input_dim=FEATURE_DIM, hidden_dim=100, output_dim=TOTAL_DISCRETE_LABELS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print(f"Model Architecture: 42 -> 100(ReLU) -> 100(ReLU) -> 4800(Softmax)")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    train_losses, val_losses, val_top1_accs, val_adj_accs = [], [], [], []
    best_top1_acc = 0.0
    best_model_path = os.path.join(output_dir, "ann_pose_classifier.pt")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_x)
        epoch_val_loss = val_loss / len(val_dataset)

        # Compute full validation metrics
        metrics = compute_metrics(model, X_val, y_val, poses_val, discretizer=discretizer)
        top1 = metrics["top1_accuracy_percent"]
        adj = metrics["adjacent_bin_accuracy_percent"]

        scheduler.step(top1)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_top1_accs.append(top1)
        val_adj_accs.append(adj)

        if top1 > best_top1_acc:
            best_top1_acc = top1
            save_model(model, best_model_path)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Loss (Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}) | "
              f"Top-1 Acc: {top1:.1f}% | Adj Bin Acc: {adj:.1f}% | "
              f"Pos RMSE: {metrics['rmse_position_error_m']:.2f}m")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed:.1f}s. Best Top-1 Acc: {best_top1_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")

    # Generate Figures 4 & 5 (Validation Error Analysis) reproducing paper
    generate_validation_error_figures(model, X_val, y_val, poses_val, discretizer, plots_dir)

    return model, metrics


def generate_validation_error_figures(model: PoseClassifierNet,
                                     X_val: np.ndarray,
                                     y_val: np.ndarray,
                                     poses_val: np.ndarray,
                                     discretizer: PoseDiscretizer,
                                     plots_dir: str):
    """
    Generate and save Figures reproducing Fig 4 & 5 from Punnoose paper:
    - Fig 4: Error between classified pose and true pose (x, y, z, phi)
    - Fig 5: Error between classified pose and correct label center
    """
    model.eval()
    with torch.no_grad():
        preds, _ = model.predict_batch(X_val)

    pred_centers = discretizer.batch_labels_to_pose_centers(preds)
    true_label_centers = discretizer.batch_labels_to_pose_centers(y_val)

    # 1. Error between classified pose and true pose (Figure 4)
    error_true = pred_centers - poses_val  # [dx, dy, dz, dphi]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Validation Set: Error Between Classified Pose and True Pose (Replicating Paper Fig. 4)", fontsize=12)

    labels = ["x [m]", "y [m]", "z [m]", "phi [deg]"]
    for i, ax in enumerate(axs.flat):
        ax.hist(error_true[:, i], bins=50, color="#1f77b4", edgecolor="black", alpha=0.75)
        ax.set_title(f"Error in {labels[i]}")
        ax.set_xlabel(f"Error ({labels[i]})")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig4_path = os.path.join(plots_dir, "fig4_error_classified_vs_true.png")
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved Figure 4 reproduction to: {fig4_path}")

    # 2. Error between classified pose and correct label center (Figure 5)
    error_label = pred_centers - true_label_centers  # [dx, dy, dz, dphi]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Validation Set: Error Between Classified Pose and Correct Label (Replicating Paper Fig. 5)", fontsize=12)

    for i, ax in enumerate(axs.flat):
        ax.hist(error_label[:, i], bins=50, color="#ff7f0e", edgecolor="black", alpha=0.75)
        ax.set_title(f"Label Delta in {labels[i]}")
        ax.set_xlabel(f"Delta ({labels[i]})")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig5_path = os.path.join(plots_dir, "fig5_error_classified_vs_label.png")
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Saved Figure 5 reproduction to: {fig5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pose Classifier ANN.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_train", type=int, default=40000, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=8000, help="Number of validation samples")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_ann(
        num_train=args.num_train,
        num_val=args.num_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
