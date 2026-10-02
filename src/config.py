"""
Configuration and constants for Vision-Based Precision Pose Estimation
Based on Punnoose (Stanford University) and UE24CS352A mini-project specs.
"""
from dataclasses import dataclass
import numpy as np

# Camera Intrinsics (Follower Aircraft Monocular Camera)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV_DEG = 50.0  # Horizontal field of view in degrees
FOCAL_LENGTH_X = CAMERA_WIDTH / (2.0 * np.tan(np.radians(CAMERA_FOV_DEG / 2.0)))
FOCAL_LENGTH_Y = FOCAL_LENGTH_X
PRINCIPAL_POINT_X = CAMERA_WIDTH / 2.0
PRINCIPAL_POINT_Y = CAMERA_HEIGHT / 2.0

CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_X, 0.0, PRINCIPAL_POINT_X],
    [0.0, FOCAL_LENGTH_Y, PRINCIPAL_POINT_Y],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DISTORTION_COEFFS = np.zeros(4, dtype=np.float64)

# Discretization Grid (Table 1 in Punnoose paper)
# 4800 Discrete Pose Labels in 4 dimensions: x, y, z, phi (roll)
GRID_CONFIG = {
    'x': {'min': -100.0, 'max': 50.0, 'n': 10},     # Forward relative distance [m]
    'y': {'min': -50.0,  'max': 50.0, 'n': 10},     # Lateral relative distance [m]
    'z': {'min': -20.0,  'max': 20.0, 'n': 8},      # Vertical relative distance [m]
    'phi': {'min': -45.0, 'max': 45.0, 'n': 6},     # Roll angle [deg]
}
TOTAL_DISCRETE_LABELS = (
    GRID_CONFIG['x']['n'] * 
    GRID_CONFIG['y']['n'] * 
    GRID_CONFIG['z']['n'] * 
    GRID_CONFIG['phi']['n']
)  # 10 * 10 * 8 * 6 = 4800

NUM_FEATURES = 14
FEATURE_DIM = NUM_FEATURES * 3  # (u, v, occlusion_flag) -> 42 features

# Particle Filter Parameters (Section IV-E & Section V)
PF_NUM_PARTICLES_DEFAULT = 1000
PF_ALPHA_DEFAULT = 0.90  # 90% resampled from likelihood, 10% re-injected from ANN
PF_PROCESS_NOISE_POS = 0.35  # meters
PF_PROCESS_NOISE_ATT = np.radians(1.5)  # radians
PF_LIKELIHOOD_EPSILON = 1e-4

# Leader Aircraft 3D Structural Keypoints (in meters, body frame: x forward, y right, z up)
AIRCRAFT_KEYPOINTS_3D = np.array([
    [ 6.0,  0.0,  0.0],  # 0: Nose tip
    [ 3.5,  0.0,  0.8],  # 1: Cockpit canopy apex
    [-0.5, -5.0,  0.0],  # 2: Port wing tip (left)
    [-0.5,  5.0,  0.0],  # 3: Starboard wing tip (right)
    [ 1.5, -1.0,  0.0],  # 4: Port wing root leading edge
    [ 1.5,  1.0,  0.0],  # 5: Starboard wing root leading edge
    [-1.8, -1.0,  0.0],  # 6: Port wing root trailing edge
    [-1.8,  1.0,  0.0],  # 7: Starboard wing root trailing edge
    [-4.5,  0.0,  2.2],  # 8: Vertical stabilizer tip
    [-3.8,  0.0,  0.5],  # 9: Vertical stabilizer base
    [-4.5, -2.2,  0.2],  # 10: Port horizontal stabilizer tip
    [-4.5,  2.2,  0.2],  # 11: Starboard horizontal stabilizer tip
    [ 0.0,  0.0, -0.6],  # 12: Fuselage belly center
    [-5.2,  0.0,  0.0],  # 13: Tail exhaust nozzle
], dtype=np.float64)

KEYPOINT_NAMES = [
    "Nose Tip",
    "Cockpit Canopy Apex",
    "Port Wing Tip",
    "Starboard Wing Tip",
    "Port Wing Root LE",
    "Starboard Wing Root LE",
    "Port Wing Root TE",
    "Starboard Wing Root TE",
    "Vertical Stabilizer Tip",
    "Vertical Stabilizer Base",
    "Port Horiz Stabilizer Tip",
    "Starboard Horiz Stabilizer Tip",
    "Fuselage Belly Center",
    "Tail Exhaust Nozzle"
]

# Wireframe edges for rendering aircraft 3D visualization
AIRCRAFT_WIREFRAME_EDGES = [
    (0, 1), (1, 9), (9, 13), (13, 0),       # Fuselage ridge & spine
    (0, 12), (12, 13),                      # Fuselage bottom keel
    (4, 2), (2, 6), (6, 4),                 # Port wing triangle
    (5, 3), (3, 7), (7, 5),                 # Starboard wing triangle
    (4, 5), (6, 7),                         # Wing root cross beams
    (9, 8), (8, 13),                        # Vertical stabilizer fin
    (13, 10), (10, 9),                      # Port horizontal stab
    (13, 11), (11, 9),                      # Starboard horizontal stab
]
