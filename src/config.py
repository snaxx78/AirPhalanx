from enum import Enum

class DroneStatus(Enum):
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class FormationType(Enum):
    V = "V Formation"
    LINE = "Line Formation"
    CIRCLE = "Circle Formation"
    SQUARE = "Square Formation"
    CUBE = "Cube Formation"
    SPHERE = "Sphere Formation"
    SHOAL = "Shoal Formation"

# Formation parameters
DEFAULT_FORMATION_PARAMS = {
    'spacing': 20.0,
    'min_height': 40.0,
    'weights': {
        'separation': 1.2,
        'alignment': 1.5,
        'cohesion': 1.0,
        'formation': 8.0,  # Increased from 6.0 to emphasize formation
        'target': 2.5,
        'altitude': 1.2
    }
}

# Default waypoints
DEFAULT_WAYPOINTS = [
    [100, 100, 60],
    [150, 200, 70],
    [200, 150, 80],
    [250, 100, 90],
    [300, 200, 70]
]

# Visualization configuration
VISUALIZATION_CONFIG = {
    'xlim': [-50, 350],
    'ylim': [-50, 250],
    'zlim': [0, 150],
}