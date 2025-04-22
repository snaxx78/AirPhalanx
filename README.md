# AirPhalanx

# 🚀 Simulation d'un Essaim de Drones en 3D

AirPhalanx est une simulation en 3D d'un essaim de drones autonomes qui s'auto-organisent de manière décentralisée. Cette simulation démontre des comportements émergents comme la formation en vol, l'élection de leader, et la tolérance aux pannes.

---

## 📌 Installation

### 🔹 1. Cloner le dépot  

```bash
git clone https://github.com/snaxx78/AirPhalanx.git
cd airphalanx
```

### 🔹 2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

ou, si pip n’est pas reconnu, utilisez :

```bash
python -m pip install -r requirements.txt
```

### 🔹 3. Lancer la simulation
Une fois les dépendances installées, exécutez la simulation en lançant :
```bash
python main.py --waypoints 0 0 50 100 100 80 200 0 60 0 0 50
```

## 📌 Structure du projet

```
airphalanx/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── models/
│   │   ├── drone.py  
│   │   ├── messages.py
│   │   └── enums.py
│   ├── simulation/
│   │   └── swarm_simulation.py
│   └── visualization/
│       ├── pyqt_visualization.py
│       └── gpu_accelerated.py
└── utils/
    └── vector_utils.py
```