import os
import json
import threading 
import queue
import cv2
from pynput import keyboard
import logging
import time
import numpy as np
from PIL import Image
import pickle
from realsense_camera import gather_realsense_cameras
from shared import Shared
from constants import MA_MAX_STEP, READY_POSE, FRONT_CAM_ID, LEFT_WRIST_CAM_ID, RIGHT_WRIST_CAM_ID


class DataCollector:
    def __init__(self,
            out_dir: str, 
            target_episode_num: int, 
            task: str,
            shared: Shared
        ):
        self.shared = shared

        self.out_dir = out_dir
        self.task = task
        self.train_dir = os.path.join(self.out_dir, "train")
        self.train_pickle_dir = os.path.join(self.out_dir, "train_pickle")
        self.json_file = os.path.join(self.train_dir, f"unified.json")

        self.target_episode_num = target_episode_num
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.train_pickle_dir, exist_ok=True)

        self._set_cameras()
        self.listener = None
        self.thread = None

        self._running = False

        self.episode_idx = 0
        self.lock = threading.Lock()
        self.queue = queue.Queue()

        self.save_freq = 10
        self.prev_time = 0.0
        
        self.rgb_buffer = []
        self.action_buffer = []
        self.qpos_buffer = []
        self.velocity_buffer =[]
        self.acceleration_buffer =[]
        self.torque_buffer =[]
        self.ft_buffer =[]
        self.unified_json_all = []

        self.prev_trigger = False
        self.trigger_now = False
        self.is_recording = False
        self.record_event = threading.Event()
        self.save_event = threading.Event()
        self.discard_event = threading.Event()
        self.init_pose_event = threading.Event()

        self.record_event_key = keyboard.KeyCode.from_char('r')
        self.save_event_key = keyboard.KeyCode.from_char('s')
        self.discard_event_key = keyboard.KeyCode.from_char('d')
        self.init_pose_event_key = keyboard.KeyCode.from_char('i')

        self._set_event_listener()
        self._listen_thread = threading.Thread(target=self._listen_worker, daemon=True)
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)

    
    def _set_cameras(self):
        all_rs_cameras = gather_realsense_cameras()
        
        self.cameras = {
            "wrist_r": all_rs_cameras[RIGHT_WRIST_CAM_ID],
            "wrist_l": all_rs_cameras[LEFT_WRIST_CAM_ID],
            "front": all_rs_cameras[FRONT_CAM_ID]
        }

        self.camera_dirs = {}
        for key, _ in self.cameras.items():
            os.makedirs(
                os.path.join(self.train_dir, key),
                exist_ok=True
            )
            self.camera_dirs[key] = os.path.join(self.train_dir, key)

        # for cam_name, cam in self.cameras.items():
        #     img = Image.fromarray(
        #         cam.read_camera()[0]["array"].astype(np.uint8)
        #     )
        #     filename = f"{cam._serial_number}.png"
        #     img.save(os.path.join(self.train_dir, filename))

    def _set_event_listener(self):
        def _on_press(k):
            if k == self.record_event_key:
                self.record_event.set()
            elif k == self.save_event_key:
                self.save_event.set()
            elif k == self.discard_event_key:
                self.discard_event.set()
            elif k == self.init_pose_event_key:
                self.init_pose_event.set()

        self.keyboard_listener = keyboard.Listener(on_press=_on_press)

    def start(self):
        if not self._running:
            self._running = True
            self.keyboard_listener.start()
            self._listen_thread.start()
            self._save_thread.start()
            logging.info("Data Collector thread started!")

    def stop(self):
        self._running = False
        self.queue.join()
        self.queue.put(None)
        self.keyboard_listener.join()
        if self._listen_thread is not None:
            self._listen_thread.join()
            self._listen_thread = None
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None

    def _listen_worker(self):
        while self._running:
            start_time = time.time()
            try:
                if self.init_pose_event.is_set():
                    if self.shared.ma_step >= MA_MAX_STEP:
                        self.shared.ma_step = 0
                        logging.info("Backing to initial pose!")
                    else:
                        self.init_pose_event.clear()
                        
                with self.shared.lock:
                    self.trigger_now = (self.shared.temp_dict["button_left"] == 1 or self.shared.temp_dict["button_right"] == 1)
                    # self.trigger_now = temp_dict["button_left"] == 1 
            
                if self.record_event.is_set():
                    if self.trigger_now and not self.prev_trigger:
                        self.is_recording = True
                        logging.info(f"Start record Episode_{self.episode_idx}")
                    elif not self.trigger_now and self.prev_trigger:
                        self.is_recording = False
                        logging.info(f"Pause record Episode_{self.episode_idx}")
                self.prev_trigger = self.trigger_now   

                if self.is_recording:
                    with self.shared.lock:
                        data_dict = self.shared.temp_dict.copy()
                    with self.lock:

                        self.rgb_buffer.append({})
                        for cam_name, cam in self.cameras.items():
                            # ret, frame = cam.read()
                            # # cv2.imshow("web cam", frame)
                            # # cv2.waitKey(1)
                            # self.rgb_buffer[-1][cam_name] = frame
                            img_array = cam.read_camera()[0]["array"]
                            self.rgb_buffer[-1][cam_name] = img_array

                        self.action_buffer.append(
                            np.concatenate([
                                data_dict["right_arm_action"] - data_dict["prev_right_arm_action"],
                                np.array([1.0]) if data_dict["right_gripper_action"] > 0.5 else np.array([0.0]),
                                data_dict["left_arm_action"] - data_dict["prev_left_arm_action"],
                                np.array([1.0]) if data_dict["left_gripper_action"] > 0.5 else np.array([0.0]),
                            ])
                        )
                        self.qpos_buffer.append(
                            np.concatenate([
                                data_dict["right_arm_action"],
                                np.array([1.0]) if data_dict["right_gripper_action"] > 0.5 else np.array([0.0]),
                                data_dict["left_arm_action"],
                                np.array([1.0]) if data_dict["left_gripper_action"] > 0.5 else np.array([0.0]),
                            ])
                        )
                        self.velocity_buffer.append(
                            np.concatenate([
                                data_dict["right_arm_velocity"],
                                data_dict["left_arm_velocity"]
                            ])
                        )
                        self.acceleration_buffer.append(
                            np.concatenate([
                            (data_dict["right_arm_velocity"] - data_dict["prev_right_arm_velocity"])/(time.time()-start_time),
                            (data_dict["left_arm_velocity"] - data_dict["prev_left_arm_velocity"])/(time.time()-start_time)
                            ])

                        )
                        self.torque_buffer.append(
                            np.concatenate([
                            data_dict["right_arm_torque"],
                            data_dict["left_arm_torque"]
                            ])
                        )
                        self.ft_buffer.append(
                            np.concatenate([
                                data_dict["ft_sensor_right"],
                                data_dict["ft_sensor_left"]
                            ])
                        )
                        
                    with self.shared.lock:
                        self.shared.temp_dict.update(
                            prev_right_arm_action = data_dict["right_arm_action"],
                            prev_left_arm_action = data_dict["left_arm_action"],
                            prev_right_arm_velocity =data_dict["right_arm_velocity"],
                            prev_left_arm_velocity =data_dict["left_arm_velocity"],

                        )

                        
                if self.save_event.is_set() or self.discard_event.is_set():
                    if self.save_event.is_set():
                        print(f"Saving Episode_{self.episode_idx}...")
                        tmp_idx = self.episode_idx
                        self.queue.put((tmp_idx,
                                        self.rgb_buffer.copy(), 
                                        self.action_buffer.copy(),
                                        self.qpos_buffer.copy(), 
                                        self.velocity_buffer.copy(),
                                        self.acceleration_buffer.copy(),
                                        self.torque_buffer.copy(),
                                        self.ft_buffer.copy()))
                        
                        self.episode_idx += 1
                        
                    if self.discard_event.is_set():
                        print(f"Discarding traj_{self.episode_idx}...")

                    self.reset()


            except Exception as e:
                logging.error(f"Listen worker error: {e}")
                breakpoint
            
            elapsed = time.time() - start_time
            sleep_time = max(0, (1 / self.save_freq) - elapsed)
            time.sleep(sleep_time)
           
    def _save_worker(self):
        while self._running:
            try:
                save_item = self.queue.get()
                if save_item is None:
                    break
                episode_idx, rgb_buffer, action_buffer, qpos_buffer, qvel_buffer , acceleration_buffer, torque_buffer, ft_buffer = save_item
                rgb_episode_path = {}
                for cam_name, cam_path in self.camera_dirs.items():
                    episode_path = os.path.join(cam_path, f"{episode_idx:06d}")
                    os.makedirs(episode_path, exist_ok=True)
                    rgb_episode_path[cam_name] = episode_path
                
                unified_json = []
                for idx, img_dict in enumerate(rgb_buffer):
                    for cam_name, episode_path in rgb_episode_path.items():
                        img = Image.fromarray(
                            img_dict[cam_name].astype(np.uint8)
                        )
                        filename = f"{idx:06d}.png"
                        img.save(os.path.join(episode_path, filename))
                
                    unified_json.append(
                        dict(
                            task = self.task,
                            raw_action = str(action_buffer[idx].tolist()),
                            image = str(os.path.join(rgb_episode_path["front"], filename))
                        )
                    )
                episode_dict = {
                    "lowdim_qpos": np.array(qpos_buffer),
                    "qvel": np.array(qvel_buffer),
                    "acceleration": np.array(acceleration_buffer),
                    "torque": np.array(torque_buffer),
                    "ft_data": np.array(ft_buffer)
                }
                pkl_path = os.path.join(self.train_pickle_dir, f"{episode_idx:06d}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(episode_dict, f)
                
                self.unified_json_all.extend(unified_json)

                with open(self.json_file, 'w') as f:
                    json.dump(self.unified_json_all, f, indent = 4)
                
                self.queue.task_done()
                print(f"Episode_{episode_idx} saved!")

                if episode_idx == self.target_episode_num - 1:
                    logging.info("[Data Collector] All episode saved!")
                
            except Exception as e:                
                logging.error(f"Save worker error: {e}")
                breakpoint

    def reset(self):
        self.prev_trigger = False
        self.shared.temp_dict.update(
            prev_right_arm_action=READY_POSE["A"].right_arm,
            prev_left_arm_action=READY_POSE["A"].left_arm
        )

        self.record_event.clear()
        self.save_event.clear()
        self.discard_event.clear()

        with self.lock:
            self.rgb_buffer = []
            self.qpos_buffer = []
            self.action_buffer = []
            self.velocity_buffer =[]
            self.acceleration_buffer =[]
            self.torque_buffer =[]
            self.ft_buffer =[]
           
        logging.info("press i to back to init pose ")


