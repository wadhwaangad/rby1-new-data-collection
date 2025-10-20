import time

import cv2
import numpy as np
import pyrealsense2 as rs


def gather_realsense_cameras(hardware_reset=False, auto_exposure=False):
    context = rs.context()
    all_devices = list(context.devices)
    all_rs_cameras = []

    if len(all_devices) == 0:
        raise RuntimeError("No RealSense devices were found.")
    
    for device in all_devices:
        # if hardware_reset:
        #     device.hardware_reset() 
        # time.sleep(0.3)
        # print(f"reset device: {device.get_info(rs.camera_info.name)}")
        rs_camera = RealSenseCamera(device, auto_exposure=True)
        all_rs_cameras.append(rs_camera)
    
    return all_rs_cameras


class RealSenseCamera:
    def __init__(self, device, auto_exposure=True):
        self._pipeline = rs.pipeline()
        self._serial_number = str(device.get_info(rs.camera_info.serial_number))
        self._config = rs.config()

        self._config.enable_device(self._serial_number)

        self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        self.camera_name = device.get_info(rs.camera_info.name)
        # import pdb; pdb.set_trace()
        if device_product_line == "L500":
            self._config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        elif device_product_line == "D400":
            self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15) # 1280 × 720 
            self._config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 15)
        else:
            self._config.enable_stream(rs.stream.color,  480, 270, rs.format.bgr8, 30)
        
        try:
            self._pipeline.stop()
        except:
            pass
        cfg = self._pipeline.start(self._config)
            # pass
        
        self._align = rs.align(rs.stream.color)

        profile = cfg.get_stream(rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        self._intrinsics = {
            self._serial_number: self._process_intrinsics(intr),
        }
        
        # color_sensor.set_option(rs.option.exposure, 500)
        # color_sensor.set_option(rs.option.exposure, 500)
        # depth_sensor = device.query_sensors()[0]
        # depth_sensor.set_option(rs.option.enable_auto_exposure, False)

        # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view
        # vertical FOV for a D455 and 1280x800 RGB
        self._fovy = 65
        
        if self.camera_name != "Intel RealSense D405":
            color_sensor = device.query_sensors()[1]
            color_sensor.set_option(rs.option.enable_auto_exposure, auto_exposure)
            # color_sensor.set_option(rs.option.brightness, 56)
            # color_sensor.set_option(rs.option.contrast, 45)
            color_sensor.set_option(rs.option.brightness, 1)
            color_sensor.set_option(rs.option.contrast, 50)

    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array(
            [[params.fx, 0, params.ppx], [0, params.fy, params.ppy], [0, 0, 1]]
        )
        intrinsics["distCoeffs"] = np.array(list(params.coeffs))
        return intrinsics

    def read_camera(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        read_time = time.time()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())[...,None]
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # depth_colormap_dim = depth_colormap.shape
        # color_colormap_dim = color_image.shape
        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim and enforce_same_dim:
        # 	color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

        # color_image = cv2.resize(color_image, dsize=(128, 96), interpolation=cv2.INTER_AREA)
        # depth_colormap = cv2.resize(depth_colormap, dsize=(128, 96), interpolation=cv2.INTER_AREA)

        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

        dict_1 = {
            "array": color_image,
            "shape": color_image.shape,
            "type": "rgb",
            "read_time": read_time,
            "serial_number": self._serial_number,
        }
        dict_2 = {
            "array": depth_image,
            "shape": depth_image.shape,
            "type": "depth",
            "read_time": read_time,
            "serial_number": self._serial_number,
        }

        return [dict_1, dict_2]

    def disable_camera(self):
        self._pipeline.stop()
        self._config.disable_all_streams()

    # def __del__(self):
    #     self.disable_camera()