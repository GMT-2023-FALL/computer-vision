import glob

import numpy as np

from utils.utils import get_frames_from_video
from manually_process.utils import get_image_points,save_params


def get_object_points(config):
    object_points = np.zeros((config['width'] * config['height'], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:config['width'], 0:config['height']].T.reshape(-1, 2)
    return object_points


def get_img_and_obj_points_from_images_folder(images_path, config):
    image_points = []
    object_points = []
    images_list = glob.glob(images_path + '/*.png')
    for file_name in images_list:
        image_points.append(get_image_points(file_name, config))
        object_points.append(get_object_points(config))
    return image_points, object_points


def calibrate_camera(config):
    for index in range(1, 2):
        # Path to the video
        video_path = 'data/cam{}/intrinsics.avi'.format(index)

        # Frame rate
        frame_rate = 0.5

        # Output path
        output_path = 'data/cam{}/frames'.format(index)
        # Get the frames
        get_frames_from_video(video_path, frame_rate, output_path)
        img_points, obj_points = get_img_and_obj_points_from_images_folder(output_path, config)
        camera_parameters_path = 'data/cam{}/camera_parameters'.format(index)
        save_params(camera_parameters_path, obj_points, img_points, config['image_size'])
