import threading
import numpy as np

class Shared:
    def __init__(self):
            
        self.temp_dict = {
            "button_left": 0,
            "button_right": 0,
            "prev_right_arm_action": np.zeros(7),
            "right_arm_action": np.zeros(7),
            "prev_left_arm_action": np.zeros(7),
            "left_arm_action": np.zeros(7),
            "right_gripper_action": np.zeros(1),
            "left_gripper_action": np.zeros(1),
            "prev_right_arm_velocity": np.zeros(7),
            "right_arm_velocity":np.zeros(7),
            "prev_left_arm_velocity":np.zeros(7),
            "left_arm_velocity":np.zeros(7),
            "right_arm_torque": np.zeros(7),
            "left_arm_torque":np.zeros(7),
            "ft_sensor_left": None,
            "ft_sensor_right": None
            
        }
        self.lock = threading.Lock()
        self.ma_step = 0