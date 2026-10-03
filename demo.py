"""
Live Demonstration Script for Formation Flying Precision Pose Estimation.
Presents the interactive 3D flight trajectory, follower camera HUD, and telemetry.
"""
import os
import sys
import argparse
from src.evaluation.benchmark import run_benchmark
from src.visualization.flight_visualizer import render_trajectory_animation


def main():
    parser = argparse.ArgumentParser(description="Live Demonstration of Formation Flight Pose Estimation.")
    parser.add_argument("--gui", type=str, default="opencv", choices=["opencv", "matplotlib"],
                        help="GUI display backend (default: opencv for instant high-fps cockpit HUD)")
    parser.add_argument("--save_gif", action="store_true", help="Save demonstration animation to GIF file")
    parser.add_argument("--open_gif", action="store_true", help="Open the pre-rendered animated flight GIF in Windows Photos")
    parser.add_argument("--gif_path", type=str, default="docs/figures/live_demo.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=15, help="Animation frames per second")
    parser.add_argument("--num_particles", type=int, default=1000, help="Number of particles for PF")
    args = parser.parse_args()

    if args.open_gif:
        gif_file = os.path.abspath(args.gif_path)
        if os.path.exists(gif_file):
            print(f"Opening {gif_file} in default image viewer...")
            os.startfile(gif_file)
            return
        else:
            print(f"GIF file not found at {gif_file}. Run with --save_gif first.")

    model_path = "data/models/ann_pose_classifier.pt"
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Training a model first...")
        from train import train_ann
        train_ann(num_train=20000, num_val=4000, epochs=15)

    print("\nRunning formation flight trajectory evaluation...")
    benchmark_data = run_benchmark(
        model_path=model_path,
        num_particles=args.num_particles
    )

    if args.save_gif:
        from src.visualization.flight_visualizer import render_trajectory_animation
        render_trajectory_animation(
            benchmark_data=benchmark_data,
            save_path=args.gif_path,
            fps=args.fps
        )
    elif args.gui == "matplotlib":
        print("\nOpening Matplotlib 3-panel dashboard...")
        from src.visualization.flight_visualizer import render_trajectory_animation
        render_trajectory_animation(
            benchmark_data=benchmark_data,
            save_path=None,
            fps=args.fps
        )
    else:
        print("\nOpening real-time Cockpit HUD Demonstration (OpenCV)...")
        from src.visualization.flight_visualizer import run_opencv_demo
        run_opencv_demo(
            benchmark_data=benchmark_data,
            fps=args.fps
        )


if __name__ == "__main__":
    main()
