import glob
import os

import cv2
import numpy as np

from manually_process.utils import save_params, get_image_points

object_points = None
random_choose_images_list = []
random_choose_images_list_2 = []


def get_auto_detected_images_list(_config):
    # read all file name from images/result/auto
    return [_config['images_folder_path'] + '/' + name.split('_')[-1] for name in glob.glob(_config['auto_detected_images_folder_path'] + "/*.jpg")]


def init(_config):
    global object_points
    object_points = np.zeros((_config['width'] * _config['height'], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:_config['width'], 0:_config['height']].T.reshape(-1, 2)


def offlineRun(task, _config):
    global object_points, random_choose_images_list,random_choose_images_list_2
    init(_config)
    object_points_list = []
    image_points_list = []
    parameter_file_path = 'parameters/offline-run-{}'.format(task)
    print("\n"
          "Offline Run {} Start".format(task))
    if task == 1:
        images_list = glob.glob(_config['images_folder_path'] + _config['image_format'])
    elif task == 2:
        random_choose_images_list = np.random.choice(get_auto_detected_images_list(_config), 10, replace=False)
        images_list = random_choose_images_list.copy()
        print("Random choose 10 images: ", random_choose_images_list)
    elif task == 3:
        random_choose_images_list_2 = np.random.choice(random_choose_images_list, 5, replace=False)
        images_list = random_choose_images_list_2.copy()
        print("Random choose 5 images from Offline Run 2: ", random_choose_images_list_2)
    else:
        print("Invalid task number")
        return
    # get image points
    for file_name in images_list:
        image_points_list.append(get_image_points(file_name, _config))
        object_points_list.append(object_points)
    # save parameters
    if len(object_points_list) > 0 and len(image_points_list) > 0:
        save_params(parameter_file_path, object_points_list, image_points_list, _config['image_size'])

