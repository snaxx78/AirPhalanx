import numpy as np

class Drone:
    def __init__(self, x=0, y=0, z=0, role="Follower"):
        """Initialisation d'un drone"""
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.random.uniform(-0.3, 0.3, size=3)  # Vitesse aléatoire
        self.state = "Actif"  # Actif ou En panne
        self.role = role  # Leader ou Follower

    def update_position(self):
        """Met à jour la position si le drone est actif"""
        if self.state == "Actif":
            self.position += self.velocity

    def set_failure(self):
        """Déclare le drone en panne"""
        self.state = "En panne"
        self.velocity = np.array([0, 0, 0])  # Arrêt du drone

    def __repr__(self):
        return f"Drone(pos={self.position}, état={self.state}, batterie={self.battery:.1f}%, rôle={self.role})"
