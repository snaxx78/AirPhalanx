from src.simulation.swarm_simulation import DecentralizedSwarmSimulation
from src.visualization.pyqt_visualization import PyQtGraphSwarmVisualization

class GPUAcceleratedSwarmSimulation:
    """Simulation d'essaim avec visualisation GPU accélérée (PyQtGraph)"""
    def __init__(self, num_drones=10, waypoints=None):
        # Créer la simulation
        self.simulation = DecentralizedSwarmSimulation(num_drones, waypoints)
        
        # Créer la visualisation
        self.visualization = PyQtGraphSwarmVisualization(self.simulation)
    
    def run(self):
        """Lance la simulation et la visualisation"""
        return self.visualization.run()