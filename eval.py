import glob
import os
import time
import pickle
import argparse
import imageio
import numpy as np
from tqdm import tqdm, trange
import random
import json

#from openrt.scripts.convert_np_to_hdf5 import normalize, unnormalize

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import cv2
from typing import List

# GroundVLA imports
from PIL import Image
import requests
import json_numpy
json_numpy.patch()
import re


from concurrent.futures import ThreadPoolExecutor, as_completed
executor = ThreadPoolExecutor(max_workers=1)

from realsense_camera import gather_realsense_cameras
# rby1 specific imports
from rby1_controller import RBY1Controller
from constants import Pose, FRONT_CAM_ID, LEFT_WRIST_CAM_ID, RIGHT_WRIST_CAM_ID

import cv2
import numpy as np
# 示例：frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(100)]
def save_frames_to_mp4(frames, save_path, fps=10):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))    
    for i, frame in enumerate(frames):
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 转换为BGR    out.release()
    print(f"Video saved: {save_path}")
# save_frames_to_mp4(frames, "output.mp4", fps=30)

def send_request(images: dict[str, np.ndarray], instruction: str, server_url: str, multi_views: bool = False):
    """
    Send the captured image and instruction to the inference server using json_numpy.
    Returns the action output as received from the server.
    """

    names = []
    imgs = []
    for name, img in images.items():
        names.append(name)
        imgs.append(np.array(img))

    for i in range(len(names)):
        Image.fromarray(imgs[i]).save(f"/home/nvidia/ws/shuol/{names[i]}.png")
       
    if not multi_views: 
        # Prepare the payload with the image and instruction from the script
        payload = {
            names[0]: imgs[0], # scene cam
            "instruction": instruction
        }
    else:
        # Prepare the payload with the image and instruction from the script
        payload = {
            "timestamp": time.time(), # add timestamp for debugging
            "instruction": instruction
        }
        for i in range(len(names)):
            payload[names[i]] = imgs[i]

    headers = {"Content-Type": "application/json"}
    response = requests.post(server_url, headers=headers, data=json_numpy.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"Server error: {response.text}")
    print("response: ", response)
    response_data = response.json()

    # import pdb;pdb.set_trace()
    # action = json_numpy.loads(response_data)
    # print(action)
    return response_data

def parse_pose_output(output):
    """
    Parse the model's output to extract pose and gripper information.
    Expected to return 8 numbers: 3 for position, 1 for gripper state, and 4 for quaternion orientation.
    If only 7 numbers are provided (i.e. missing one quaternion component), we compute the missing value assuming
    a unit quaternion.
    :param output: The output from the model (can be bytes, string, list, or numpy array).
    :return: A tuple: (position (x, y, z), gripper_state, orientation (x, y, z, w))
    """
    # If output is bytes, decode it.
    if isinstance(output, bytes):
        output = output.decode('utf-8')

    # If output is a list or numpy array, convert to a list of floats.
    if isinstance(output, (list, np.ndarray)):
        numbers = [float(x) for x in output]
    else:
        # Assume it's a string; use regex to find numbers.
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", output)
        numbers = [float(n) for n in numbers]

    if len(numbers) == 7:
        # Assume order: [x, y, z, gripper_state, q_x, q_y, q_z]
        pos = tuple(numbers[0:3])
        gripper_state = numbers[-1]
        q_xyz = numbers[4:7]
        # Compute the missing quaternion component assuming a unit quaternion:
        s = sum(q * q for q in q_xyz)
        # Protect against small negative due to floating point precision.
        missing = max(0.0, 1.0 - s)
        q_w = missing**0.5
        orientation = tuple(q_xyz + [q_w])
        return pos, gripper_state, orientation
    elif len(numbers) == 8:
        pos = tuple(numbers[0:3])
        gripper_state = numbers[3]
        orientation = tuple(numbers[4:8])
        return pos, gripper_state, orientation
    else:
        raise ValueError(f"[Utils] Expected 7 or 8 numbers in predicted action, got: {len(numbers)}")

def shortest_angle(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi
    
def get_dict(demo):
    dic = {}
    obs_keys = demo[0].keys()
    for key in obs_keys:
        dic[key] = np.stack([d[key] for d in demo])
    return dic

def action_preprocessing(dic):
        # compute actual deltas s_t+1 - s_t (keep gripper actions)
    actions_tmp = dic["lowdim_qpos"].copy()
    actions_tmp[:-1, 0:7] = (
        dic["lowdim_qpos"][1:, 0:7] - dic["lowdim_qpos"][:-1, 0:7]
    )
    actions_tmp[:-1, 8:15] = (
        dic["lowdim_qpos"][1:, 8:15] - dic["lowdim_qpos"][:-1, 8:15]
    )
    actions = actions_tmp[:-1]

    return actions


def replay_episode(demo, env, visual=False):
    # stack data
    dic = get_dict(demo)
    actions = np.stack([d["action"] for d in demo])
    actions = action_preprocessing(dic, actions) # delta action
    demo_length = actions.shape[0]

    for step_idx in tqdm(range(demo_length)):
        act = actions[step_idx]

        obs = env.get_observation()

        if visual:
            cv2.imshow("Camera View", obs["215122252864_rgb"])
            cv2.waitKey(1)  # Small delay to allow the image to refresh
        
        if step_idx == 15:
            breakpoint() 
        env.step(act)
    cv2.destroyAllWindows()

def replay_episode_pickle(demo_dir, rby1_controller, delta_pose):
    # stack data
    # dic = get_dict(demo)
    with open(demo_dir, 'rb') as f:
        data = pickle.load(f)
    actions = data["lowdim_qpos"]
    demo_length = actions.shape[0]

    for step_idx in tqdm(range(demo_length)):
        act = actions[step_idx]
        chunk_right = act[0:8]
        chunk_left = act[8:16]
        delta_pose = update_delta_pose(delta_pose, chunk_right, chunk_left)
        rby1_controller.execute_action_abs(delta_pose)

def get_pickle_len(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return len(data["lowdim_qpos"])

def get_start_end_index(root_dir, current_episode_name):
    pkl_dir = f"{root_dir}_pickle"
    pkl_files = sorted(os.listdir(pkl_dir))  # ['000000.pkl', '000001.pkl', ..., '000007.pkl']    # 只保留 .pkl 文件
    pkl_files = [f for f in pkl_files if f.endswith(".pkl")]   
    cur_idx = pkl_files.index(current_episode_name)   
    start_index = 0
    for f in pkl_files[:cur_idx]:
        path = os.path.join(pkl_dir, f)
        start_index += get_pickle_len(path)    
    current_path = os.path.join(pkl_dir, current_episode_name)
    current_length = get_pickle_len(current_path)    
    end_index = start_index + current_length
    return start_index, end_index

def replay_episode_json(root_dir, rby1_controller, delta_pose, episode_number):
    start_idx, end_idx = get_start_end_index(root_dir, f"{episode_number}.pkl")

    json_file = f"{root_dir}/unified.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    t = 0
    for i in trange(end_idx - start_idx):
        act = np.array(eval(data[i + start_idx]["raw_action"]))
        chunk_right = act[0:8]
        chunk_left = act[8:16]
        delta_pose = update_delta_pose(delta_pose, chunk_right, chunk_left)
        rby1_controller.execute_action_delta(delta_pose)
        t += 1
        if t % 8 == 0:
            time.sleep(5)
            t = 0



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)

def invert_gripper(chunk):
    curr_gripper = chunk[-1]
    if curr_gripper > 0.5: 
        curr_gripper = 0
    else:
        curr_gripper = 1
    
    result = chunk.copy()
    result[-1] = curr_gripper
    return result

def model_inference(obs, msg, url, multi_views, model):
    # sanity check           
    act = send_request(obs, msg, url, multi_views=multi_views)
    print(f"[{model}] Received Action (deltas): {act}")
    return act
            
def update_delta_pose(delta_pose: Pose, chunk_right: np.ndarray, chunk_left: np.ndarray):
    
    delta_pose.right_arm = chunk_right[:-1]
    delta_pose.left_arm = chunk_left[:-1]
    delta_pose.right_gripper = chunk_right[-1]
    delta_pose.left_gripper = chunk_left[-1]
    
    delta_pose.right_arm[-1] = 0.0
    return delta_pose

def run_experiment(url, msg):
        
    url = url
    msg = msg
    action_chunking = True
    multi_views = True

    rby1_controller = RBY1Controller(
        address="192.168.30.1:50051", 
        teleop=False, 
        control_freq=10,
    )
    rby1_controller.start_control()

    delta_pose = Pose(
        toros=np.zeros(6),
        right_arm=np.zeros(7),
        left_arm=np.zeros(7),
        head=np.zeros(2),
        right_gripper = np.zeros(1),
        left_gripper = np.zeros(1)
    )
    # mode = input("Enter mode (1: replay, 2: close_loop, 3: open_loop): ")
    mode = input("Enter mode (1, 2, 3): ")
    while mode != "1" and mode != "2" and mode != "3":
        mode = input("Invalid input. Please enter 1 for replay, 2 for close_loop, or 3 for open_loop: ")
    if mode == "1":
        mode = "replay"
    elif mode == "2":
        mode = "close_loop"
    elif mode == "3":
        mode = "open_loop"
    # mode = "open_loop"

    all_rs_cameras = gather_realsense_cameras()
    cameras = {
        "wrist_r": all_rs_cameras[RIGHT_WRIST_CAM_ID],
        "wrist_l": all_rs_cameras[LEFT_WRIST_CAM_ID],
        # "front": all_rs_cameras[FRONT_CAM_ID]
    }
    time.sleep(5)

    prev_gripper_right = 0
    prev_gripper_left = 0
   
    root_dir = "/home/nvidia/ws/shuo/RBY1_data_collection/data/102/train"
    episode_number = "000020" 
    demo_dir = f"{root_dir}_pickle/{episode_number}.pkl"
    save_dir = "/home/nvidia/ws/shuol/rby1-sdk/molmoact/demo_video"

    os.makedirs(save_dir, exist_ok=True)
    
    print("[INFO] mode: ", "close_loop")
    print("[INFO] url: ", url)

    if mode == "replay":
        print('[INFO] demo_dir: ', demo_dir)
        # replay_episode_pickle(demo_dir, rby1_controller, delta_pose)
        replay_episode_json(root_dir, rby1_controller, delta_pose, episode_number)

    elif mode == "close_loop":
        # reset env
        obs = {}
        for cam_name, cam in cameras.items():
            obs[cam_name] = cam.read_camera()[0]["array"]

        while True:
            model = "molmo_act"

            try:
                count = 0
                while True:
                    start = time.time()
                    act = model_inference(obs, msg, url, multi_views, model)
                    end = time.time()
                    if action_chunking:
                        if model == "openvla":
                            raise ValueError(f"Didn't implement action chunking on openvla.")
                        # execute h actions at once
                        for i in range(len(act)):
          
                            # chunk = action_queue.pop(0)
                            chunk_right = act[i][0:8]
                            chunk_left = act[i][8:16]
                            chunk_right = invert_gripper(chunk_right)
                            chunk_left = invert_gripper(chunk_left)
                            delta_pose = update_delta_pose(delta_pose, chunk_right, chunk_left)
                            rby1_controller.execute_action_delta(delta_pose)
                            if prev_gripper_right != chunk_right[-1] or \
                                prev_gripper_left != chunk_left[-1]:
                                time.sleep(2)
                            for cam_name, cam in cameras.items():
                                obs[cam_name] = cam.read_camera()[0]["array"]
    
                            prev_gripper_right = chunk_right[-1]
                            prev_gripper_left = chunk_left[-1]
                        
                    else:
                        chunk_right = act[0:8]
                        chunk_left = act[8:16]
                        chunk_right = invert_gripper(chunk_right)
                        chunk_left = invert_gripper(chunk_left)
                        delta_pose = update_delta_pose(delta_pose, chunk_right, chunk_left)
                        rby1_controller.execute_action_delta(delta_pose)
                        if prev_gripper_right != chunk_right[-1] or \
                            prev_gripper_left != chunk_left[-1]:
                            time.sleep(2)
                        for cam_name, cam in cameras.items():
                            obs[cam_name] = cam.read_camera()[0]["array"]
                        prev_gripper_right = chunk_right[-1]
                        prev_gripper_left = chunk_left[-1]

                        print(
                            f"Model: {model} | Time {np.around(end-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act,3)}"
                        )
                        time.sleep(1.0)
                        for cam_name, cam in cameras.items():
                            obs[cam_name] = cam.read_camera()[0]["array"]
                        
            except Exception as e:
                print("Close_loop error: ", e)
                reset = input("reset(y/n)?")
                if reset.lower() == "y":
                    rby1_controller.reset()
                    for cam_name, cam in cameras.items():
                        obs[cam_name] = cam.read_camera()[0]["array"]
                    print("[INFO] Resetting environment...")
                continue

    elif mode == "open_loop":
        rgb_paths = {}
        for cam_name, _ in cameras.items():
            dir = f"{root_dir}/{cam_name}/{episode_number}"
            rgb_paths[cam_name] = sorted(glob.glob(os.path.join(dir, "*.png")))
    
        model = "molmo_act"

        training_obs = {}

        #################################################### INDEX #############################################################
        # For test the episode_list
        # while True:
        #     time.sleep(1.0)
        #     current_img = Image.fromarray(cameras["front"].read_camera()[0]["array"].astype(np.uint8))
        #     current_img.show()
        #     step = int(input("Input step index: "))
        
        #################################################### INDEX #############################################################

        #################################################### list #############################################################
        
        ## for pick tool episode 15
        # episode_list = [0, 6, 12, 18, 24, 38, 39, 44, 48, 51, 79, 88, 95, 95, 115, 117 ,127 ,130,145]

        # for pouring water episode 23
        # episode_list = [0, 0, 0, 21, 52, 93, 93, 93, 93, 93, 95, 139, 139, 135, 136, 136, 136]
        
        # for press
        # episode_list = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 100, 108, 116, 120, 140]
        # episode_list = [10, 15, 18, 18, 22, 22, 25, 40, 47, 73, 80, 80, 80, 88, 105, 110, 120]
        # episode_list = [73, 80, 80, 80, 88, 105, 112, 120]
        # episode_list = [73, 80, 80, 80, 88, 105, 112, 120]
        # episode_list = [5, 15, 15, 15, 30, 43, 50, 60, 65, 75, 89, 90, 100, 100, 116]
        episode_list = [5, 15, 15, 15, 30, 43, 50, 60, 65, 75, 89, 90, 100, 100, 112, 116]
        # episode_list = [89, 90, 100, 100, 112, 116]
        for step in episode_list:
        #     if step > 90:
        #         time.sleep(1.0)
        #         current_img = Image.fromarray(cameras["front"].read_camera()[0]["array"].astype(np.uint8))
        #         current_img.show()
        #         step = int(input("Input step index: "))

        #################################################### LIST #############################################################
            
        # for step in range(len(rgb_paths["front"]), 8):
            # print(f'Step: {step} | lang: {msg}')
            start = time.time()

            for cam_name, paths in rgb_paths.items():
                training_obs[cam_name] = Image.open(paths[step])

            # training_obs["front"].show()

            act = model_inference(training_obs, msg, url, multi_views, model=model)
            end = time.time()
            if action_chunking:
                # execute h actions at once
                # action_queue.extend([chunk for chunk in act])   
                
                for i in range(int(len(act))):
      
                    # chunk = action_queue.pop(0)
                    chunk_right = act[i][0:8]
                    chunk_left = act[i][8:16]
                    chunk_right = invert_gripper(chunk_right)
                    chunk_left = invert_gripper(chunk_left)
                    delta_pose = update_delta_pose(delta_pose, chunk_right, chunk_left)
                    rby1_controller.execute_action_delta(delta_pose)
                    if prev_gripper_right != chunk_right[-1] or \
                        prev_gripper_left != chunk_left[-1]:
                        time.sleep(2)
                    prev_gripper_right = chunk_right[-1]
                    prev_gripper_left = chunk_left[-1]
            else:
                # step
                chunk_right = act[0:8]
                chunk_left = act[8:16]
                chunk_right = invert_gripper(chunk_right)
                chunk_left = invert_gripper(chunk_left)
                delta_pose = update_delta_pose(delta_pose, chunk_right, chunk_left)
                rby1_controller.execute_action_delta(delta_pose)
                if prev_gripper_right != chunk_right[-1] or \
                    prev_gripper_left != chunk_left[-1]:
                    time.sleep(2)
                prev_gripper_right = chunk_right[-1]
                prev_gripper_left = chunk_left[-1]
                print(
                    f"Model: {model} | Time {np.around(end-start, 3)} EE {np.around(obs['lowdim_ee'][:3],3)} Act {np.around(act,3)}"
                )
       
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--url", type=str, default="https://56cd19852a91.ngrok-free.app/act", help="Server url")
    parser.add_argument("--instruction", type=str, default = "put sushi into drawer")
    args = parser.parse_args()
    run_experiment(args.url, args.instruction)




