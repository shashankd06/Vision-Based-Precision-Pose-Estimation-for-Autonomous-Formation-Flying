"""
Flight Visualization and Synthetic Camera HUD Renderer.
Renders 3D formation flight trajectories, monocular camera projections with keypoint detections,
and live tracking telemetry dashboards.
"""
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

from src.config import AIRCRAFT_WIREFRAME_EDGES, CAMERA_WIDTH, CAMERA_HEIGHT
from src.simulation.camera import GimbaledCamera, euler_to_rotation_matrix


class FlightDashboardVisualizer:
    def __init__(self,
                 camera: Optional[GimbaledCamera] = None,
                 figsize: Tuple[int, int] = (16, 9)):
        self.camera = camera if camera is not None else GimbaledCamera()
        self.figsize = figsize
        self.aircraft = self.camera.aircraft
        self.edges = AIRCRAFT_WIREFRAME_EDGES

    def setup_dashboard(self):
        """Configure 3-panel dashboard: 3D Trajectory, Camera HUD, and Real-Time Error Plots."""
        self.fig = plt.figure(figsize=self.figsize)
        self.gs = GridSpec(2, 2, figure=self.fig, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

        # 1. Panel 1 (Top Left): 3D Formation Flight Trajectory
        self.ax_3d = self.fig.add_subplot(self.gs[:, 0], projection='3d')
        self.ax_3d.set_title("3D Formation Flight Trajectory (Follower Closing on Leader)", fontsize=11, fontweight='bold')
        self.ax_3d.set_xlabel("X (Forward) [m]")
        self.ax_3d.set_ylabel("Y (Lateral) [m]")
        self.ax_3d.set_zlabel("Z (Vertical) [m]")

        # 2. Panel 2 (Top Right): Follower Monocular Camera Sensor View
        self.ax_cam = self.fig.add_subplot(self.gs[0, 1])
        self.ax_cam.set_title("Follower Monocular Camera Sensor HUD (14 Keypoints)", fontsize=11, fontweight='bold')
        self.ax_cam.set_xlim(0, self.camera.width)
        self.ax_cam.set_ylim(self.camera.height, 0)  # Invert y for pixel coordinates
        self.ax_cam.set_facecolor("#0a0f1d")  # Dark cockpit night HUD style
        self.ax_cam.grid(True, color="#1e3a5f", linestyle="--", alpha=0.5)

        # 3. Panel 3 (Bottom Right): Real-Time Position & Attitude Error Curves
        self.ax_err = self.fig.add_subplot(self.gs[1, 1])
        self.ax_err.set_title("Real-Time Tracking Error: Baseline vs Enhanced PnP", fontsize=11, fontweight='bold')
        self.ax_err.set_xlabel("Time [s]")
        self.ax_err.set_ylabel("Position Error [m]")
        self.ax_err.grid(True, linestyle="--", alpha=0.6)

    def draw_camera_frame(self,
                          detected_uv: np.ndarray,
                          occlusions: np.ndarray,
                          true_uv: np.ndarray,
                          est_uv_pnp: Optional[np.ndarray] = None):
        """Draw synthetic camera view with keypoints, wireframe lines, and HUD info."""
        self.ax_cam.clear()
        self.ax_cam.set_xlim(0, self.camera.width)
        self.ax_cam.set_ylim(self.camera.height, 0)
        self.ax_cam.set_facecolor("#0a0f1d")
        self.ax_cam.grid(True, color="#1e3a5f", linestyle="--", alpha=0.5)

        # Draw wireframe edges between visible keypoints
        for (i, j) in self.edges:
            if not occlusions[i] and not occlusions[j]:
                self.ax_cam.plot(
                    [detected_uv[i, 0], detected_uv[j, 0]],
                    [detected_uv[i, 1], detected_uv[j, 1]],
                    color="#00ffcc", linewidth=1.5, alpha=0.8
                )

        # Draw visible keypoints (Red/Amber target markers)
        vis_mask = ~occlusions
        if np.any(vis_mask):
            self.ax_cam.scatter(
                detected_uv[vis_mask, 0], detected_uv[vis_mask, 1],
                c="#ff3366", s=35, edgecolors="#ffffff", zorder=5, label="Detected Keypoints (CNN)"
            )

        # Draw occluded keypoints (Dim grey 'x')
        occ_mask = occlusions
        if np.any(occ_mask):
            self.ax_cam.scatter(
                detected_uv[occ_mask, 0], detected_uv[occ_mask, 1],
                c="#888888", marker="x", s=25, alpha=0.6, label="Occluded (Self-shadow)"
            )

        # Draw PnP estimated wireframe overlay (Lime green)
        if est_uv_pnp is not None:
            for (i, j) in self.edges:
                self.ax_cam.plot(
                    [est_uv_pnp[i, 0], est_uv_pnp[j, 0]],
                    [est_uv_pnp[i, 1], est_uv_pnp[j, 1]],
                    color="#39ff14", linestyle=":", linewidth=1.2, alpha=0.9
                )

        # Camera reticle (crosshairs)
        cx, cy = self.camera.cx, self.camera.cy
        self.ax_cam.axhline(cy, color="#00ffcc", linestyle="--", linewidth=0.6, alpha=0.4)
        self.ax_cam.axvline(cx, color="#00ffcc", linestyle="--", linewidth=0.6, alpha=0.4)
        self.ax_cam.legend(loc="upper right", fontsize=8, facecolor="#0a0f1d", edgecolor="#00ffcc", labelcolor="white")


def render_trajectory_animation(benchmark_data: dict,
                                save_path: Optional[str] = "docs/figures/live_demo.gif",
                                fps: int = 15):
    """Render full animation of the 10-second flight approach."""
    times = benchmark_data["times"]
    true_poses = benchmark_data["true_poses"]
    ann_poses = benchmark_data["ann_poses"]
    pf_poses = benchmark_data["pf_poses"]
    pnp_poses = benchmark_data["pnp_poses"]
    num_steps = len(times)

    camera = GimbaledCamera()
    vis = FlightDashboardVisualizer(camera=camera)
    vis.setup_dashboard()

    pos_err_pf = np.linalg.norm(pf_poses[:, :3] - true_poses[:, :3], axis=1)
    pos_err_pnp = np.linalg.norm(pnp_poses[:, :3] - true_poses[:, :3], axis=1)

    def update(frame):
        # Update 3D Trajectory
        vis.ax_3d.clear()
        vis.ax_3d.set_title(f"3D Formation Flight (t = {times[frame]:.1f}s)", fontsize=11, fontweight='bold')
        vis.ax_3d.set_xlabel("X (Forward) [m]")
        vis.ax_3d.set_ylabel("Y (Lateral) [m]")
        vis.ax_3d.set_zlabel("Z (Vertical) [m]")

        # Plot full trajectory path
        vis.ax_3d.plot(true_poses[:, 0], true_poses[:, 1], true_poses[:, 2], 'k--', alpha=0.4, label="True Path")
        # Current position marker
        vis.ax_3d.scatter(true_poses[frame, 0], true_poses[frame, 1], true_poses[frame, 2], color="red", s=60, label="Leader Aircraft")
        vis.ax_3d.scatter(0, 0, 0, color="blue", marker="^", s=80, label="Follower Aircraft (Cam)")
        vis.ax_3d.plot([0, true_poses[frame, 0]], [0, true_poses[frame, 1]], [0, true_poses[frame, 2]], 'g:', alpha=0.5, label="Line of Sight")
        vis.ax_3d.legend(loc="upper left", fontsize=8)

        # Update Camera HUD
        tx, ty, tz = true_poses[frame, :3]
        roll, pitch, yaw = true_poses[frame, 3:]
        uv, occlusions, _ = camera.project_pose(tx, ty, tz, roll, pitch, yaw, noise_std=0.8)

        # PnP estimate projected
        px, py, pz = pnp_poses[frame, :3]
        proll, ppitch, pyaw = pnp_poses[frame, 3:]
        pnp_uv, _, _ = camera.project_pose(px, py, pz, proll, ppitch, pyaw, noise_std=0.0)

        vis.draw_camera_frame(uv, occlusions, uv, pnp_uv)

        # Update Error Curve
        vis.ax_err.clear()
        vis.ax_err.set_title("Tracking Error Telemetry", fontsize=11, fontweight='bold')
        vis.ax_err.set_xlabel("Time [s]")
        vis.ax_err.set_ylabel("Position Error [m]")
        vis.ax_err.grid(True, linestyle="--", alpha=0.6)
        vis.ax_err.set_xlim(0, 10.0)
        vis.ax_err.set_ylim(0, 40.0)

        vis.ax_err.plot(times[:frame + 1], pos_err_pf[:frame + 1], color="#1f77b4", label="Punnoose PF (Coarse)")
        vis.ax_err.plot(times[:frame + 1], pos_err_pnp[:frame + 1], color="#2ca02c", linewidth=2.0, label="Enhanced PnP (Fine)")
        vis.ax_err.legend(loc="upper right", fontsize=8)

    ani = animation.FuncAnimation(vis.fig, update, frames=num_steps, interval=1000 // fps)

    if save_path:
        print(f"Saving live demo animation to {save_path}...")
        ani.save(save_path, writer="pillow", fps=fps)
        print(f"Saved live demo animation successfully!")
    else:
        plt.show()

    plt.close()
