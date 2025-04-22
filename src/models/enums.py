from enum import Enum

class DroneStatus(Enum):
    """État possible d'un drone"""
    ACTIVE = 1
    FAILING = 2
    FAILED = 3

class DroneRole(Enum):
    """Rôle possible d'un drone dans l'essaim"""
    LEADER = 1
    FOLLOWER = 2

class WaypointStatus(Enum):
    """État de progression vers un waypoint"""
    NAVIGATING = 1
    REACHED = 2