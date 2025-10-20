import os
import cv2
import numpy as np
from realsense_camera import gather_realsense_cameras
from constants import RIGHT_WRIST_CAM_ID, LEFT_WRIST_CAM_ID, FRONT_CAM_ID
from PIL import Image
test_dir = "/home/nvidia/ws/shuo/test_camera"
all_rs_cameras = gather_realsense_cameras()
       
cameras = {
    "wrist_r": all_rs_cameras[RIGHT_WRIST_CAM_ID],
    "wrist_l": all_rs_cameras[LEFT_WRIST_CAM_ID],
    # "front": all_rs_cameras[FRONT_CAM_ID]
}

camera_dirs = {}
for key, _ in cameras.items():
    os.makedirs(
        os.path.join(test_dir, key),
        exist_ok=True
    )
    camera_dirs[key] = os.path.join(test_dir, key)

for cam_name, cam in cameras.items():
    print(f"{cam_name}:", cam._intrinsics)
    img = Image.fromarray(
        cam.read_camera()[0]["array"].astype(np.uint8)
    )
    filename = f"{cam_name}.png"
    img.save(os.path.join(test_dir, filename))