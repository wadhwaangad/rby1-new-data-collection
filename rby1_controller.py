import numpy as np
import logging
import threading
import time
import os
from functools import partial

import rby1_sdk as rby
from constants import (
    READY_POSE, 
    GRIPPER_DIRECTION, 
    MA_MAX_STEP,
    Settings,
    Pose
)
from shared import Shared

class Gripper:
    """
    Class for gripper
    """

    def __init__(self):
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([1, 1])
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q: np.typing.NDArray = None
        self._running = False
        self._thread = None

    def initialize(self, verbose=False):
        rv = True
        for dev_id in [0, 1]:
            if not self.bus.ping(dev_id):
                if verbose:
                    logging.error(f"Dynamixel ID {dev_id} is not active")
                rv = False
            else:
                if verbose:
                    logging.info(f"Dynamixel ID {dev_id} is active")
        if rv:
            logging.info("Servo on gripper")
            self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        return rv

    def set_operating_mode(self, mode):
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 0.5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q
            # A small value (e.g., 5) was too short and failed to detect limits properly, so a reasonably larger value was chosen.
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        return True

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            if self.target_q is not None:
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                )
            time.sleep(0.1)

    def set_target(self, normalized_q):
        # self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            logging.error("Cannot set target. min_q or max_q is not valid.")
            return

        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q



class RBY1Controller:
    def __init__(self, 
                 address: str, 
                 teleop: bool, 
                 control_freq: int,
                 shared: Shared = None,
    ):
        self.lock = threading.Lock()
        self.control_freq = control_freq
        if shared is not None:
            self.shared = shared
        self.control_mode = "position"  # position , impedance
        self.position_mode = self.control_mode == "position"
        self.power = ".*"
        self.servo = "torso_.*|right_arm_.*|left_arm_.*"
        # self.servo = "torso_.*|right_arm_0|right_arm_1|right_arm_2|right_arm_3|right_arm_5|right_arm_6|left_arm_.*"
        self.ma_q_limit_barrier = 0.5
        self.ma_min_q = np.deg2rad(
            [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
        )
        self.ma_max_q = np.deg2rad(
            [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
        )
        self.ma_torque_limit = np.array([4.0] * 14)
        self.ma_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)
        self.right_q = None
        self.left_q = None
        self.right_qvel=None
        self.left_qvel =None
        self.right_torque =None
        self.left_torque =None
        self.left_ft = None
        self.right_ft = None
        self.right_minimum_time = 1.0
        self.left_minimum_time = 1.0
        
        self.setup_robot(address)
        logging.info("[RBY1 Controller] Robot initialized successfully!")
        self.setup_gripper()
        logging.info("[RBY1 Controller] Gripper initialized successfully!")
        self.stream = self.robot.create_command_stream(priority=1)  # TODO
        self.stream.send_command(
            self.joint_position_command_builder(
                READY_POSE[self.model.model_name],
                minimum_time=5,
                control_hold_time=1e6,
                position_mode=self.position_mode,
            )
        )
        self.target_pose = READY_POSE[self.model.model_name]

        if teleop:
            self.setup_master_arm()
            logging.info("[RBY1 Controller] Master Arm initialized successfully!")
        else:
            self.master_arm = None

    def start_control(self):
        self.robot.start_state_update(
            partial(self.robot_state_callback), 
            1 / Settings.master_arm_loop_period
        )
        self.gripper.start()
        if self.master_arm is not None:
            self.master_arm.start_control(
                partial(self.master_arm_control_loop)
            )
        
    def stop(self):
        self.robot.stop_state_update()
        if self.master_arm is not None:
            self.master_arm.stop_control()
        self.robot.cancel_control()
        time.sleep(0.5)

        self.robot.disable_control_manager()
        self.robot.power_off("12v")
        self.gripper.stop()

    def reset(self):
        self.robot_home()
        self.gripper.homing()
        self.target_pose = READY_POSE[self.model.model_name]
        if self.master_arm is not None:
            self.ma_home()
        
    def robot_state_callback(self, state: rby.RobotState_A):
        with self.lock:
            self.robot_q = state.position
            self.left_ft =state.ft_sensor_left
            self.right_ft = state.ft_sensor_right

    def setup_robot(self, address):
        self.robot = rby.create_robot(address, 'a')
        if not self.robot.connect():
            logging.error(f"Failed to connect robot {address}")
            exit(1)
        supported_model = ["A", "T5", "M"]
        supported_control_mode = ["position", "impedance"]
        self.model = self.robot.model()
        self.dyn_model = self.robot.get_dynamics()
        self.dyn_state =self. dyn_model.make_state([], self.model.robot_joint_names)
        self.robot_q = None
        self.robot_max_q = self.dyn_model.get_limit_q_upper(self.dyn_state)
        self.robot_min_q = self.dyn_model.get_limit_q_lower(self.dyn_state)
        self.robot_max_qdot = self.dyn_model.get_limit_qdot_upper(self.dyn_state)
        self.robot_min_qdot =self.dyn_model.get_limit_qdot_lower(self.dyn_state)
        self.robot_max_qddot = self.dyn_model.get_limit_qddot_upper(self.dyn_state)

        if not self.position_mode:
            self.robot_max_qdot[self.model.right_arm_idx[-1]] *= 10
            self.robot_max_qdot[self.model.left_arm_idx[-1]] *= 10

        if not self.model.model_name in supported_model:
            logging.error(
                f"Model {self.model.model_name} not supported (Current supported model is {supported_model})"
            )
            exit(1)
        if not self.control_mode in supported_control_mode:
            logging.error(
                f"Control mode {self.control_mode} not supported (Current supported control mode is {supported_control_mode})"
            )
            exit(1)
    
        if not self.robot.is_power_on(self.power):
            if not self.robot.power_on(self.power):
                logging.error(f"Failed to turn power ({self.power}) on")
                exit(1)
        if not self.robot.is_servo_on(self.servo):
            if not self.robot.servo_on(self.servo):
                logging.error(f"Failed to servo ({self.servo}) on")
                exit(1)
        self.robot.reset_fault_control_manager()
        if not self.robot.enable_control_manager():
            logging.error(f"Failed to enable control manager")
            exit(1)
        for arm in ["right", "left"]:
            if not self.robot.set_tool_flange_output_voltage(arm, 12):
                logging.error(f"Failed to set tool flange output voltage ({arm}) as 12v")
                exit(1)
        self.robot.set_parameter("joint_position_command.cutoff_frequency", "3")
        
        self.robot_home()
       
    def setup_gripper(self):
        self.gripper = Gripper()
        if not self.gripper.initialize():
            logging.error("Failed to initialize gripper")
            self.robot.stop_state_update()
            self.robot.power_off("12v")
            exit(1)
        self.gripper.homing()

    def setup_master_arm(self):
        rby.upc.initialize_device(rby.upc.MasterArmDeviceName)
        master_arm_model = "/home/nvidia/ws/shuo/rby1-sdk/models/master_arm/model.urdf"
        self.master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
        self.master_arm.set_model_path(master_arm_model)
        self.master_arm.set_control_period(Settings.master_arm_loop_period)
        active_ids = self.master_arm.initialize(verbose=True)
        if len(active_ids) != rby.upc.MasterArm.DeviceCount:
            logging.error(
                f"Mismatch in the number of devices detected for RBY Master Arm (active devices: {active_ids})"
            )
            exit(1)

        self.stream.send_command(
            self.joint_position_command_builder(
                READY_POSE[self.model.model_name],
                minimum_time=5,
                control_hold_time=1e6,
                position_mode=self.position_mode,
            )
        )
    
    def master_arm_control_loop(self, state: rby.upc.MasterArm.State):

        if self.right_q is None:
            self.right_q = state.q_joint[0:7]
        if self.left_q is None:
            self.left_q = state.q_joint[7:14]
        

        if self.shared.ma_step < MA_MAX_STEP: # 2000 / 100 = 20 seconds
            self.shared.ma_step += 1
            self.stream.send_command(
                self.joint_position_command_builder(
                    READY_POSE[self.model.model_name],
                    minimum_time=MA_MAX_STEP / 100,
                    control_hold_time=1e6,
                    position_mode=self.position_mode,
                )
            )
            self.right_q = READY_POSE[self.model.model_name].right_arm  
            self.left_q = READY_POSE[self.model.model_name].left_arm  

            if self.shared.ma_step == MA_MAX_STEP:
               logging.info("press r to start record!")
            return self.ma_home(READY_POSE[self.model.model_name], state)

        ma_input = rby.upc.MasterArm.ControlInput()

        # print(f"--- {datetime.datetime.now().time()} ---")
        # print(f"Button: {state.button_right.button}, {state.button_left.button}")
        # print(f"Trigger: {state.button_right.trigger}, {state.button_left.trigger}")
        self.gripper.set_target(
            np.array(
                [state.button_right.trigger / 1000, state.button_left.trigger / 1000]
            )
        )
        with self.shared.lock:
            self.shared.temp_dict.update(
                button_left = state.button_left.button,
                button_right = state.button_right.button,
                right_gripper_action = np.array([state.button_right.trigger / 1000], dtype=np.float32),
                left_gripper_action = np.array([state.button_left.trigger / 1000], dtype=np.float32),
            )
        # ===== CALCULATE MASTER ARM COMMAND =====
        torque = (
            state.gravity_term
            + self.ma_q_limit_barrier
            * (
                np.maximum(self.ma_min_q - state.q_joint, 0)
                + np.minimum(self.ma_max_q - state.q_joint, 0)
            )
            + self.ma_viscous_gain * state.qvel_joint
        )
        torque = np.clip(torque, -self.ma_torque_limit, self.ma_torque_limit)
        if state.button_right.button == 1:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[0:7] = torque[0:7]
            self.right_q = state.q_joint[0:7]
        else:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[0:7].fill(5)
            ma_input.target_position[0:7] = self.right_q

        if state.button_left.button == 1:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[7:14] = torque[7:14]
            self.left_q = state.q_joint[7:14]
        else:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[7:14].fill(5)
            ma_input.target_position[7:14] = self.left_q

        # Check whether target configure is in collision
        q = self.robot_q.copy()
        q[self.model.right_arm_idx] = self.right_q
        q[self.model.left_arm_idx] = self.left_q
        self.dyn_state.set_q(q)
        self.dyn_model.compute_forward_kinematics(self.dyn_state)
        is_collision = (
            self.dyn_model.detect_collisions_or_nearest_links(self.dyn_state, 1)[0].distance
            < 0.02
        )
        self.right_qvel= state.qvel_joint[0:7]
        self.left_qvel= state.qvel_joint [7:14]
        self.right_torque =state.torque_joint[0:7]
        self.left_torque =state.torque_joint[7:14]
        with self.shared.lock:
            self.shared.temp_dict.update(
                right_arm_action = np.clip(
                    self.right_q,
                    self.robot_min_q[self.model.right_arm_idx],
                    self.robot_max_q[self.model.right_arm_idx],
                ),
                left_arm_action = np.clip(
                    self.left_q,
                    self.robot_min_q[self.model.left_arm_idx],
                    self.robot_max_q[self.model.left_arm_idx],
                ),
                right_arm_velocity = np.clip(
                    self.right_qvel,
                    self.robot_min_qdot[self.model.right_arm_idx],
                    self.robot_max_qdot[self.model.right_arm_idx]
                ),
                left_arm_velocity = np.clip(
                    self.left_qvel,
                    self.robot_min_qdot[self.model.left_arm_idx],
                    self.robot_max_qdot[self.model.left_arm_idx]
                ),
                right_arm_torque = self.right_torque,
                left_arm_torque = self.left_torque,
                ft_sensor_right=self.right_ft,
                ft_sensor_left=self.left_ft

                )
        
        # temp_robot_q = self.robot_q.copy()
        # temp_right_q = self.right_q.copy()
        # temp_right_q[0] = temp_robot_q[self.model.right_arm_idx][0]

        # temp_left_q = self.left_q.copy()
        # temp_left_q[0] = temp_robot_q[self.model.left_arm_idx][0]
        # ===== BUILD ROBOT COMMAND =====
        rc = rby.BodyComponentBasedCommandBuilder()
        if state.button_right.button and not is_collision:
            self.right_minimum_time -= Settings.master_arm_loop_period
            self.right_minimum_time = max(
                self.right_minimum_time, Settings.master_arm_loop_period * 1.01
            )
            right_arm_builder = (
                rby.JointPositionCommandBuilder()
                if self.position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                right_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(
                    np.clip(
                        self.right_q,
                        self.robot_min_q[self.model.right_arm_idx],
                        self.robot_max_q[self.model.right_arm_idx],
                    )
                )
                .set_velocity_limit(self.robot_max_qdot[self.model.right_arm_idx])
                .set_acceleration_limit(self.robot_max_qddot[self.model.right_arm_idx] * 30)
                .set_minimum_time(self.right_minimum_time)
            )
            if not self.position_mode:
                (
                    right_arm_builder.set_stiffness(
                        [Settings.impedance_stiffness] * len(self.model.right_arm_idx)
                    )
                    .set_damping_ratio(Settings.impedance_damping_ratio)
                    .set_torque_limit(
                        [Settings.impedance_torque_limit] * len(self.model.right_arm_idx)
                    )
                )
            rc.set_right_arm_command(right_arm_builder)
        else:
            self.right_minimum_time = 0.8

        if state.button_left.button and not is_collision:
            self.left_minimum_time -= Settings.master_arm_loop_period
            self.left_minimum_time = max(
                self.left_minimum_time, Settings.master_arm_loop_period * 1.01
            )
            left_arm_builder = (
                rby.JointPositionCommandBuilder()
                if self.position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                left_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(
                    np.clip(
                        self.left_q,
                        self.robot_min_q[self.model.left_arm_idx],
                        self.robot_max_q[self.model.left_arm_idx],
                    )
                )
                .set_velocity_limit(self.robot_max_qdot[self.model.left_arm_idx])
                .set_acceleration_limit(self.robot_max_qddot[self.model.left_arm_idx] * 30)
                .set_minimum_time(self.left_minimum_time)
            )
            if not self.position_mode:
                (
                    left_arm_builder.set_stiffness(
                        [Settings.impedance_stiffness] * len(self.model.left_arm_idx)
                    )
                    .set_damping_ratio(Settings.impedance_damping_ratio)
                    .set_torque_limit(
                        [Settings.impedance_torque_limit] * len(self.model.left_arm_idx)
                    )
                )
            rc.set_left_arm_command(left_arm_builder)
        else:
            self.left_minimum_time = 0.8
        self.stream.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(rc)
            )
        )

        return ma_input
    
    def joint_position_command_builder(
        self, pose: Pose, minimum_time, control_hold_time=0, position_mode=True
    ):
        right_arm_builder = (
            rby.JointPositionCommandBuilder()
            if position_mode
            else rby.JointImpedanceControlCommandBuilder()
        )
        (
            right_arm_builder.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
            )
            .set_position(pose.right_arm)
            .set_minimum_time(minimum_time)
        )
        if not position_mode:
            (
                right_arm_builder.set_stiffness(
                    [Settings.impedance_stiffness] * len(pose.right_arm)
                )
                .set_damping_ratio(Settings.impedance_damping_ratio)
                .set_torque_limit([Settings.impedance_torque_limit] * len(pose.right_arm))
            )

        left_arm_builder = (
            rby.JointPositionCommandBuilder()
            if position_mode
            else rby.JointImpedanceControlCommandBuilder()
        )
        (
            left_arm_builder.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
            )
            .set_position(pose.left_arm)
            .set_minimum_time(minimum_time)
        )
        if not position_mode:
            (
                left_arm_builder.set_stiffness(
                    [Settings.impedance_stiffness] * len(pose.left_arm)
                )
                .set_damping_ratio(Settings.impedance_damping_ratio)
                .set_torque_limit([Settings.impedance_torque_limit] * len(pose.left_arm))
            )

        torso_builder = rby.JointPositionCommandBuilder()
        (
            torso_builder.set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
            )
            .set_position(pose.toros)
            .set_minimum_time(minimum_time)
        )
        
        return rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.BodyComponentBasedCommandBuilder()
                .set_torso_command(torso_builder)
                .set_right_arm_command(right_arm_builder)
                .set_left_arm_command(left_arm_builder)
            )
        )

    def robot_home(self,  minimum_time=5.0):
        handler = self.robot.send_command(
            self.joint_position_command_builder(
                READY_POSE[self.model.model_name], 
                minimum_time)
        )
        return handler.get() == rby.RobotCommandFeedback.FinishCode.Ok

    def ma_home(self, pose: Pose, state: rby.upc.MasterArm.State):
        ma_input = rby.upc.MasterArm.ControlInput()

        current_right = state.q_joint[0:7]
        current_left = state.q_joint[7:14]

        factor = self.shared.ma_step / MA_MAX_STEP

        ma_input.target_operating_mode[0:7].fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_torque[0:7].fill(5)
        ma_input.target_position[0:7] = factor * pose.right_arm + (1 - factor) * current_right

        ma_input.target_operating_mode[7:14].fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_torque[7:14].fill(5)
        ma_input.target_position[7:14] = factor * pose.left_arm + (1 - factor) * current_left

        return ma_input
    
    def execute_action_delta(self, delta_pose: Pose):
        # Check whether target configure is in collision
        with self.lock:
            q = self.robot_q.copy()

        # if np.any(delta_pose.toros):
        #     q[self.model.torso_idx] += delta_pose.toros
        # if np.any(delta_pose.right_arm):
        #     q[self.model.right_arm_idx] += delta_pose.right_arm
        # if np.any(delta_pose.left_arm):
        #     q[self.model.left_arm_idx] += delta_pose.left_arm
        # if np.any(delta_pose.head):
        #     q[self.model.head_idx] += delta_pose.head
        # delta_pose.right_arm[4] = 0.0
        if np.any(delta_pose.toros):
            self.target_pose.toros += delta_pose.toros
            q[self.model.torso_idx] = self.target_pose.toros
        if np.any(delta_pose.right_arm):
            self.target_pose.right_arm += delta_pose.right_arm
            q[self.model.right_arm_idx] = self.target_pose.right_arm
        if np.any(delta_pose.left_arm):
            self.target_pose.left_arm += delta_pose.left_arm
            q[self.model.left_arm_idx] = self.target_pose.left_arm
        if np.any(delta_pose.head):
            self.target_pose.head += delta_pose.head
            q[self.model.head_idx] = self.target_pose.head

        self.gripper.set_target(
            np.array(
                [delta_pose.right_gripper, delta_pose.left_gripper]
            )
        )
        
        self.dyn_state.set_q(q)
        self.dyn_model.compute_forward_kinematics(self.dyn_state)
        is_collision = (
            self.dyn_model.detect_collisions_or_nearest_links(self.dyn_state, 1)[0].distance
            < 0.02
        )

        if not is_collision:
            # pose = Pose(
            #     toros = q[self.model.torso_idx],
            #     right_arm = q[self.model.right_arm_idx],
            #     left_arm = q[self.model.left_arm_idx],
            #     head = q[self.model.head_idx],
            #     right_gripper = np.array([1.0]),
            #     left_gripper = np.array([1.0]),
            # )
            self.stream.send_command(
                self.joint_position_command_builder(
                    self.target_pose,
                    minimum_time=0.5,
                    control_hold_time=1e6,
                    position_mode=self.position_mode,
                )
            )
            time.sleep( 1 / self.control_freq )

        
        else:
            logging.error("Target joint pose out of limit!")

    def execute_action_abs(self, pose: Pose):
        # Check whether target configure is in collision
        q = self.robot_q.copy()

        if np.any(pose.right_arm):
            q[self.model.right_arm_idx] = pose.right_arm
        if np.any(pose.left_arm):
            q[self.model.left_arm_idx] = pose.left_arm

        self.gripper.set_target(
            np.array(
                [pose.right_gripper, pose.left_gripper]
            )
        )
        
        self.dyn_state.set_q(q)
        self.dyn_model.compute_forward_kinematics(self.dyn_state)
        is_collision = (
            self.dyn_model.detect_collisions_or_nearest_links(self.dyn_state, 1)[0].distance
            < 0.02
        )

        if not is_collision:
            pose = Pose(
                toros = q[self.model.torso_idx],
                right_arm = q[self.model.right_arm_idx],
                left_arm = q[self.model.left_arm_idx],
                head = q[self.model.head_idx],
                right_gripper = np.array([1.0]),
                left_gripper = np.array([1.0]),
            )
            self.stream.send_command(
                self.joint_position_command_builder(
                    pose,
                    minimum_time=0.5,
                    control_hold_time=1e6,
                    position_mode=self.position_mode,
                )
            )
        
        else:
            logging.error("Target joint pose out of limit!")

        time.sleep( 1 / self.control_freq )


        
