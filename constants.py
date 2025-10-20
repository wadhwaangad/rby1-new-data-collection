from dataclasses import dataclass
import numpy as np

FRONT_CAM_ID = 2
LEFT_WRIST_CAM_ID = 1
RIGHT_WRIST_CAM_ID = 0

GRIPPER_DIRECTION = False

MA_MAX_STEP = 200

@dataclass
class Pose:
    toros: np.typing.NDArray
    right_arm: np.typing.NDArray
    left_arm: np.typing.NDArray
    head: np.typing.NDArray
    right_gripper: np.typing.NDArray
    left_gripper: np.typing.NDArray

class Settings:
    master_arm_loop_period = 1 / 100

    impedance_stiffness = 30
    impedance_damping_ratio = 1.0
    impedance_torque_limit = 10.0

READY_POSE = {
    "A": Pose(
        # toros=np.deg2rad([0.0, 30.0, -45.0, 30.0, 0.0, 0.0]),
        toros=np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        right_arm=np.deg2rad([-24.0, -30.0, -6.0, -128.0, -30.0, 100.0, -120.0]),
        left_arm=np.deg2rad([-24.0, 30.0, 6.0, -128.0, 30.0, 100.0, 120.0]),
        # right_arm=np.deg2rad([-36.0, -30.0, -10.0, -128.0, 0.0, 100.0, -120.0]),
        # left_arm=np.deg2rad([-36.0, 30.0, 10.0, -128.0, 0.0, 100.0, 120.0]),
        # right_arm=np.deg2rad([-0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0]),
        # left_arm=np.deg2rad([-0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0]),
        # right_arm=np.deg2rad([-36.0, -18.0, -10.0, -45.0, 0.0, 40.0, -15.0]),
        # left_arm=np.deg2rad([-36.0, 18.0, 10.0, -45.0, 0.0, 40.0, 15.0]),
        # right_arm=np.deg2rad([-30.0, -50.0, -15.0, -140.0, -80.0, 100.0, -130.0]),
        # left_arm=np.deg2rad([-30.0, 50.0, 15.0, -140.0, 80.0, 100.0, 130.0]),
        head=np.deg2rad([0.0, 30.0]),
        right_gripper = np.array([1.0]),
        left_gripper = np.array([1.0])
    ),
    "T5": Pose(
        toros=np.deg2rad([45.0, -90.0, 45.0, 0.0, 0.0]),
        right_arm=np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
        left_arm=np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
        head=np.deg2rad([0.0, 30.0]),
        right_gripper = np.array([1.0]),
        left_gripper = np.array([1.0])
    ),
    "M": Pose(
        toros=np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]),
        right_arm=np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
        left_arm=np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
        head=np.deg2rad([0.0, 30.0]),
        right_gripper = np.array([1.0]),
        left_gripper = np.array([1.0])
    ),
}