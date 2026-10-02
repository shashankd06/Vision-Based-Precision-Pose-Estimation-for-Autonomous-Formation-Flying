"""
Comparative Benchmark: Coarse ANN vs Punnoose Particle Filter vs Enhanced PnP Refiner.
Replicates flight approach scenario (Punnoose Fig 6) and error evaluations (Figs 9 & 10).
"""
import os
import time
import argparse
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from src.config import GRID_CONFIG
from src.simulation.camera import GimbaledCamera
from src.simulation.dynamics import FlightSimulation
from src.dataset.pose_discretizer import PoseDiscretizer
from src.models.ann_classifier import PoseClassifierNet, load_model
from src.filtering.particle_filter import RelativePoseParticleFilter
from src.filtering.pnp_refiner import PnPPoseRefiner
from src.evaluation.metrics import compute_trajectory_metrics


def run_benchmark(model_path: str = "data/models/ann_pose_classifier.pt",
                  plots_dir: str = "docs/figures",
                  num_particles: int = 1000,
                  alpha: float = 0.9,
                  noise_std: float = 1.0) -> Dict[str, Any]:
    """
    Execute full comparative benchmark on the 10s flight closing trajectory.
    """
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 75)
    print("  Vision-Based Precision Pose Estimation: Comparative Benchmark")
    print("=" * 75)

    # 1. Initialize Components
    camera = GimbaledCamera()
    discretizer = PoseDiscretizer()
    sim = FlightSimulation(dt=0.1, duration=10.0)
    flight_states = sim.generate_closing_trajectory(seed=42)
    num_steps = len(flight_states)

    # Load ANN classifier
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first!")

    model = load_model(model_path)
    print(f"Loaded trained ANN model from: {model_path}")

    # Initialize Particle Filter and PnP Refiner
    pf = RelativePoseParticleFilter(
        num_particles=num_particles,
        alpha=alpha,
        dt=0.1,
        camera=camera,
        discretizer=discretizer
    )
    pnp = PnPPoseRefiner(camera=camera)

    # Data arrays for logging
    times = np.zeros(num_steps)
    true_poses = np.zeros((num_steps, 6))
    ann_poses = np.zeros((num_steps, 6))
    pf_poses = np.zeros((num_steps, 6))
    pnp_poses = np.zeros((num_steps, 6))

    ann_latencies = []
    pf_latencies = []
    pnp_latencies = []

    print(f"Executing 10.0s trajectory ({num_steps} steps at 10 Hz)...")

    # Step through flight trajectory
    for k, state in enumerate(flight_states):
        times[k] = state.time
        tx, ty, tz = state.rel_pos
        roll, pitch, yaw = state.rel_euler_deg
        true_poses[k] = [tx, ty, tz, roll, pitch, yaw]

        # 1. Follower Camera Sensor Observation (Keypoints + Ray-traced Occlusions)
        uv, occlusions, _ = camera.project_pose(
            x=tx, y=ty, z=tz,
            phi_deg=roll, pitch_deg=pitch, yaw_deg=yaw,
            noise_std=noise_std
        )
        feat_vector = camera.extract_feature_vector(
            x=tx, y=ty, z=tz,
            phi_deg=roll, pitch_deg=pitch, yaw_deg=yaw,
            noise_std=noise_std, normalize=True
        )

        # 2. Method 1: Coarse ANN Classification
        t0 = time.perf_counter()
        pred_label, _ = model.predict_label(feat_vector)
        ann_time = time.perf_counter() - t0
        ann_latencies.append(ann_time)

        coarse_center = discretizer.label_to_pose_center(pred_label)
        ann_poses[k] = [coarse_center[0], coarse_center[1], coarse_center[2], coarse_center[3], 0.0, 0.0]

        # 3. Method 2: Bayesian Particle Filter (Punnoose Baseline)
        t1 = time.perf_counter()
        pf_est = pf.step(
            detected_uv=uv,
            occlusions=occlusions,
            ann_label=pred_label,
            follower_vx=state.follower_ctrl[0]
        )
        pf_time = time.perf_counter() - t1
        pf_latencies.append(pf_time)
        pf_poses[k] = pf_est

        # 4. Method 3: Proposed Enhanced PnP Refinement (Warm-started by ANN)
        t2 = time.perf_counter()
        R_cam = camera.compute_camera_frame_rotation(state.rel_pos)
        pnp_est, _ = pnp.refine_pose(
            detected_uv=uv,
            occlusions=occlusions,
            coarse_pose=coarse_center,
            R_cam=R_cam
        )
        pnp_time = time.perf_counter() - t2
        pnp_latencies.append(pnp_time)
        pnp_poses[k] = pnp_est

    # 5. Compute Metrics for each method
    metrics_ann = compute_trajectory_metrics(true_poses, ann_poses, ann_latencies)
    metrics_pf = compute_trajectory_metrics(true_poses, pf_poses, pf_latencies)
    metrics_pnp = compute_trajectory_metrics(true_poses, pnp_poses, pnp_latencies)

    # Print Formatted Comparison Table
    print("\n" + "=" * 75)
    print("  QUANTITATIVE BENCHMARK RESULTS (10-Second Formation Flight Approach)")
    print("=" * 75)
    headers = ["Estimation Method", "Pos RMSE [m]", "Pos Med [m]", "Pos Max [m]", "Roll RMSE [°]", "Latency [ms]", "FPS"]
    print(f"{headers[0]:<28} | {headers[1]:<12} | {headers[2]:<11} | {headers[3]:<11} | {headers[4]:<13} | {headers[5]:<12} | {headers[6]:<6}")
    print("-" * 105)

    methods = [
        ("Discrete ANN Classifier", metrics_ann),
        ("ANN + Particle Filter (Punnoose)", metrics_pf),
        ("ANN + PnP Refiner (Ours)", metrics_pnp),
    ]

    for name, m in methods:
        print(f"{name:<28} | {m['pos_rmse_m']:<12.3f} | {m['pos_median_m']:<11.3f} | {m['pos_max_m']:<11.3f} | {m['roll_rmse_deg']:<13.2f} | {m['latency_mean_ms']:<12.2f} | {m['fps']:<6.1f}")
    print("=" * 105)

    # 6. Generate Publication Figures
    generate_benchmark_plots(times, true_poses, ann_poses, pf_poses, pnp_poses, plots_dir)

    return {
        "metrics_ann": metrics_ann,
        "metrics_pf": metrics_pf,
        "metrics_pnp": metrics_pnp,
        "times": times,
        "true_poses": true_poses,
        "ann_poses": ann_poses,
        "pf_poses": pf_poses,
        "pnp_poses": pnp_poses
    }


def generate_benchmark_plots(times, true_poses, ann_poses, pf_poses, pnp_poses, plots_dir):
    """Generate and save publication figures matching paper and showing our enhancement."""

    # 1. Figure 6 Reproduction: Relative Position Trajectory
    plt.figure(figsize=(8, 5))
    plt.plot(times, true_poses[:, 0], 'b-', linewidth=2.0, label="Relative X (Forward Distance)")
    plt.plot(times, true_poses[:, 1], 'g--', linewidth=1.5, label="Relative Y (Lateral Distance)")
    plt.plot(times, true_poses[:, 2], 'r-.', linewidth=1.5, label="Relative Z (Vertical Distance)")
    plt.xlabel("Time [s]", fontsize=11)
    plt.ylabel("Relative Position [m]", fontsize=11)
    plt.title("Relative Position Flight Trajectory (Replicating Paper Fig. 6)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    fig6_path = os.path.join(plots_dir, "fig6_flight_trajectory.png")
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    print(f"Saved Figure 6 reproduction to: {fig6_path}")

    # 2. Figures 9 & 10 Reproduction + Our Enhancement Comparison
    # Calculate position errors and attitude errors
    pos_err_ann = np.linalg.norm(ann_poses[:, :3] - true_poses[:, :3], axis=1)
    pos_err_pf = np.linalg.norm(pf_poses[:, :3] - true_poses[:, :3], axis=1)
    pos_err_pnp = np.linalg.norm(pnp_poses[:, :3] - true_poses[:, :3], axis=1)

    roll_err_ann = np.abs(ann_poses[:, 3] - true_poses[:, 3])
    roll_err_pf = np.abs(pf_poses[:, 3] - true_poses[:, 3])
    roll_err_pnp = np.abs(pnp_poses[:, 3] - true_poses[:, 3])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Position Error Comparison
    axs[0].plot(times, pos_err_ann, color="#808080", linestyle=":", alpha=0.8, label="Discrete ANN Classifier")
    axs[0].plot(times, pos_err_pf, color="#1f77b4", linewidth=1.8, label="ANN + Particle Filter (Punnoose Baseline)")
    axs[0].plot(times, pos_err_pnp, color="#2ca02c", linewidth=2.2, label="ANN + PnP Refiner (Proposed Enhanced Pipeline)")
    axs[0].set_ylabel("Position Error [m]", fontsize=11)
    axs[0].set_title("Trajectory Error Analysis: Position & Attitude Tracking Over Time", fontsize=12)
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend(loc="upper right", framealpha=0.9)

    # Bottom: Attitude (Roll) Error Comparison
    axs[1].plot(times, roll_err_ann, color="#808080", linestyle=":", alpha=0.8, label="Discrete ANN Classifier")
    axs[1].plot(times, roll_err_pf, color="#1f77b4", linewidth=1.8, label="ANN + Particle Filter (Punnoose Baseline)")
    axs[1].plot(times, roll_err_pnp, color="#2ca02c", linewidth=2.2, label="ANN + PnP Refiner (Proposed Enhanced Pipeline)")
    axs[1].set_xlabel("Time [s]", fontsize=11)
    axs[1].set_ylabel("Attitude (Roll) Error [deg]", fontsize=11)
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    comp_path = os.path.join(plots_dir, "benchmark_error_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"Saved Comparative Error Benchmark plot to: {comp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pose Estimation Benchmark.")
    parser.add_argument("--model_path", type=str, default="data/models/ann_pose_classifier.pt")
    parser.add_argument("--num_particles", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--noise_std", type=float, default=1.0)
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        num_particles=args.num_particles,
        alpha=args.alpha,
        noise_std=args.noise_std
    )
