import time

class Message:
    """Message échangé entre les drones"""
    def __init__(self, sender_id, msg_type, data=None):
        self.sender_id = sender_id
        self.msg_type = msg_type  # position, formation_update, leader_heartbeat, etc.
        self.data = data if data is not None else {}
        self.timestamp = time.time()
        self.ttl = 5  # Time to live (nombre de sauts maximum) - augmenté pour meilleure propagation