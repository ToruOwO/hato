from collections import OrderedDict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from .camera import CameraDriver


def get_device_ids() -> List[str]:
    device_ids = []
    for dev in rs.context().query_devices():
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    return device_ids


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_ids={self.device_ids})"

    def __init__(
        self,
        device_ids: Optional[List] = None,
        height: int = 480,
        width: int = 640,
        fps: int = 30,
        warm_start: int = 60,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        self.height = height
        self.width = width
        self.fps = fps
        if device_ids is None:
            self.device_ids = get_device_ids()
        else:
            self.device_ids = device_ids
        self.img_size = img_size

        # Start stream
        print(f"Connecting to RealSense cameras ({len(self.device_ids)} found) ...")
        self.pipes = []
        self.profiles = OrderedDict()
        for i, device_id in enumerate(self.device_ids):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            self.profiles[device_id] = pipe.start(config)
            print(f"Connected to camera {i} ({device_id}).")

        self.align = rs.align(rs.stream.color)

        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(warm_start):
            self._get_frames()

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_num_cameras(self):
        return len(self.device_ids)

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
        concatenate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W)
        """
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()
        rgbd = np.empty([num_cams, self.height, self.width, 4], dtype=np.uint16)
        if self.img_size is not None:
            rgbd_resized = np.empty(
                [num_cams, self.img_size[1], self.img_size[0], 4], dtype=np.uint16
            )

        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgbd[i, :, :, :3] = np.asanyarray(color_frame.get_data())

            depth_frame = frameset.get_depth_frame()
            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            depth_frame = rs.hole_filling_filter().process(depth_frame)
            rgbd[i, :, :, 3] = np.asanyarray(depth_frame.get_data())

            if self.img_size is not None:
                rgbd_resized[i] = cv2.resize(rgbd[i], self.img_size)

        if self.img_size is not None:
            rgbd = rgbd_resized

        if concatenate:
            image = np.concatenate(rgbd[..., :3], axis=1, dtype=np.uint8)
            depth = np.concatenate(rgbd[..., -1], axis=1, dtype=np.uint8)
        else:
            image = rgbd[..., :3].astype(np.uint8)
            depth = rgbd[..., -1].astype(np.uint16)
        return image, depth
