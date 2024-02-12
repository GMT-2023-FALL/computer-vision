import glob
import os

import cv2
import numpy as np

from manually_process.utils import manually_find_corner_points, save_params


def offlineRun1(_config):

    object_points_list = []
    image_points_list = []
    object_points = np.zeros((_config['width'] * _config['height'], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:_config['width'], 0:_config['height']].T.reshape(-1, 2)
    pattern_size = (_config['width'], _config['height'])
    auto_detected_images_folder_path = _config['auto_detected_images_folder_path']
    parameter_file_path = 'parameters/offline-run-1'

    images = glob.glob(_config['images_folder_path'])
    for index, file_name in enumerate(images):
        # if os env is macos, then use the following code to get the file name
        # detect the os env
        if os.name == 'nt':
            file_name_index = file_name.split('\\')[-1]
        else:
            file_name_index = file_name.split('/')[-1]
        print("Processing: ", file_name_index)
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Finding sub-pixel corners based on the original corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _config['criteria'])
            object_points_list.append(object_points)
            image_points_list.append(corners2)
            # Draw and save the corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imwrite("{}/result_{}".format(auto_detected_images_folder_path, file_name_index), img)
            print("Auto Detected: {}".format(file_name_index))
        else:
            # print("No corners found in image: ", file_name_index)
            image_points_list.append(manually_find_corner_points(file_name, _config))
            object_points_list.append(object_points)
        cv2.destroyAllWindows()
    if len(object_points_list) > 0 and len(image_points_list) > 0:
        save_params(parameter_file_path, object_points_list, image_points_list, _config['image_size'])
