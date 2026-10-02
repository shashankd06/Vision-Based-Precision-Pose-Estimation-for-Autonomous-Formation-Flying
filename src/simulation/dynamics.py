"""
Aircraft Flight Kinematics and 10s Closing Approach Trajectory.
Implements the formation flight dynamics described in Punnoose Section IV-B & V.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class AircraftState:
    time: float
    # Relative position [x, y, z] in meters (x forward, y lateral, z vertical)
    rel_pos: np.ndarray
    # Relative attitude [roll, pitch, yaw] in degrees
    rel_euler_deg: np.ndarray
    # Follower controls [vx, theta, phi]
    follower_ctrl: np.ndarray
    # Leader disturbances [vx_dist, theta_dist, phi_dist]
    leader_dist: np.ndarray


class FlightSimulation:
    def __init__(self, dt: float = 0.1, duration: float = 10.0):
        self.dt = dt
        self.duration = duration
        self.num_steps = int(duration / dt)

    def generate_closing_trajectory(self, seed: int = 42) -> List[AircraftState]:
        """
        Generate a 10-second formation flight closing trajectory matching Figure 6 in the paper.
        The follower aircraft approaches the leader from -75m to -30m forward distance,
        with gentle sinusoidal lateral weave and roll corrections.
        """
        np.random.seed(seed)
        states = []

        # Initial relative position: follower trailing behind at x = 75m, slight lateral & vertical offset
        # Note: In coordinate convention, follower-to-leader forward vector has x in [30, 80] m
        x0 = 75.0
        y0 = -12.0
        z0 = 5.0
        roll0 = 8.0  # degrees

        curr_pos = np.array([x0, y0, z0], dtype=np.float64)
        curr_euler = np.array([roll0, 0.0, 0.0], dtype=np.float64)

        for step in range(self.num_steps):
            t = step * self.dt

            # Closing approach profile:
            # Forward relative distance closes smoothly from 75m down to ~35m
            # x(t) = 75 - 4.0 * t (closing speed ~ 4 m/s relative)
            true_x = 75.0 - 4.0 * t + 0.5 * np.sin(0.5 * t)
            true_y = -12.0 + 1.2 * t + 2.0 * np.sin(0.8 * t)
            true_z = 5.0 - 0.4 * t + 1.0 * np.cos(0.6 * t)

            # Roll oscillations simulating aerodynamic wake & banking maneuvers
            true_roll = 10.0 * np.sin(1.2 * t) + 4.0 * np.cos(2.0 * t)
            true_pitch = 2.0 * np.sin(0.9 * t)
            true_yaw = 3.0 * np.cos(0.7 * t)

            rel_pos = np.array([true_x, true_y, true_z], dtype=np.float64)
            rel_euler = np.array([true_roll, true_pitch, true_yaw], dtype=np.float64)

            # Controls and unknown leader disturbances (sampled uniformly per paper Section IV-E.2)
            follower_ctrl = np.array([4.0, 0.0, np.radians(true_roll)], dtype=np.float64)
            leader_dist = np.random.uniform(low=[-1.0, -0.05, -0.1], high=[1.0, 0.05, 0.1])

            states.append(AircraftState(
                time=t,
                rel_pos=rel_pos,
                rel_euler_deg=rel_euler,
                follower_ctrl=follower_ctrl,
                leader_dist=leader_dist
            ))

        return states
