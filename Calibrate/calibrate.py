import glob

import numpy as np

from utils.utils import get_frames_from_video
from manually_process.utils import get_image_points


def get_object_points(config):
    object_points = np.zeros((config['width'] * config['height'], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:config['width'], 0:config['height']].T.reshape(-1, 2)
    return object_points


def get_intrinsics_from_video(video_path, frame_rate, output_path, config):
    # Get the frames
    get_frames_from_video(video_path, frame_rate, output_path)

    image_points = []
    object_points = []
    images_list = glob.glob(output_path + '/*.png')
    for file_name in images_list:
        image_points.append(get_image_points(file_name, config))
        object_points.append(get_object_points(config))


def calibrate_camera(config):
    for index in range(1, 5):
        # Path to the video
        video_path = 'E:/UU/Computer_Vision/Assignment-2/data/cam{}/intrinsics.avi'.format(index)

        # Frame rate
        frame_rate = 2

        # Output path
        output_path = 'E:/UU/Computer_Vision/Assignment-2/data/cam{}/frames'.format(index)
        get_intrinsics_from_video(video_path, frame_rate, output_path, config)
