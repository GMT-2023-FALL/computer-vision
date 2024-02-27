import pickle

import cv2
import glm
import numpy as np
from tqdm import tqdm

from manually_process.utils import get_camera_extrinsic, get_camera_intrinsic, rotate_voxels, is_foreground_in_image

block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    data, colors = [], []
    cam_configs = []
    # Get camera intrinsic and extrinsic parameters
    for cam_index in range(1, 5):
        camera_matrix, dist_coeffs = get_camera_intrinsic(cam_index)
        rotation_vector, translation_vector = get_camera_extrinsic(cam_index)
        cam_configs.append([camera_matrix, dist_coeffs, rotation_vector, translation_vector])
    # Get foreground images
    foreground_images = [cv2.imread('data/cam{}/video_manual_segment.png'.format(index), cv2.IMREAD_GRAYSCALE) for index
                         in range(1, 5)]
    colorful_images = [cv2.imread('data/cam{}/segment/video.jpg'.format(index), cv2.IMREAD_COLOR) for index in
                       range(1, 5)]  # 确保路径正确

    # Generate voxel data
    for x in tqdm(np.linspace(0, width, num=500)):
        for y in np.linspace(0, height, num=500):
            for z in np.linspace(0, depth, num=500):
                position = [x * block_size - width / 2, y * block_size, z * block_size - depth / 2]
                data.append(position)
                colors.append([1, 1, 1])
                voxel_world = np.array([position],
                                       dtype=np.float32)  # 调整为正确的形状以用于projectPoints
                is_foreground = True
                voxel_colors = []  # 用于存储每个视角下的颜色值
                for cam_config, foreground_img, colorful_image in zip(cam_configs, foreground_images, colorful_images):
                    camera_matrix, dist_coeffs, rotation_vector, translation_vector = cam_config

                    # 使用projectPoints将3D点投影到2D
                    image_points, _ = cv2.projectPoints(voxel_world,
                                                        rotation_vector,
                                                        translation_vector,
                                                        camera_matrix,
                                                        dist_coeffs)
                    pixel = image_points[0][0]  # 提取投影点
                    #  检查是否在前景
                    if not is_foreground_in_image(pixel, foreground_img):
                        is_foreground = False
                        break  # 如果在任何一个视角中该体素不是前景，我们可以停止检查
                    else:
                        # 如果是前景，我们可以添加颜色
                        voxel_colors.append(colorful_image[int(pixel[1]), int(pixel[0])])
                if is_foreground:
                    # 如果体素在所有视角中都是前景，则添加到数据集
                    data.append(position)
                    # 颜色可以是根据视角综合得到的平均值
                    avg_color = np.mean(voxel_colors, axis=0).astype(int)/255
                    colors.append(avg_color.tolist())
    # save data and colors
    with open('data_500.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('colors_500.pkl', 'wb') as f:
        pickle.dump(colors, f)

    # load data and colors
    # with open('data_500.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # with open('colors_500.pkl', 'rb') as f:
    #     colors = pickle.load(f)
    # rotate the voxel data
    axis_of_rotation = 'x'
    angle_radians = np.radians(90)
    data = rotate_voxels(data, angle_radians, axis_of_rotation)
    return data, colors


def test_rotate(final_data):
    r_x = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]])
    r_y_1 = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
    r_y_2 = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [-1, 0, 0]])
    r_x_1 = [r_x.dot(p) for p in final_data]
    r_y_1_ = [r_y_1.dot(y) for y in r_x_1]
    r_y_2_ = [r_y_2.dot(y) for y in r_y_1_]
    return [np.multiply(m, 3) for m in r_y_2_]


def get_cam_positions():
    cam_position = []
    for i in range(1, 5):
        rvec, tvec = get_camera_extrinsic(i)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_position = -rotation_matrix.T.dot(tvec)
        final_position = [camera_position[0] * 2, camera_position[2] * -1, camera_position[1] * 2]
        cam_position.append(final_position)
    color = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cam_position, color


def get_cam_rotation_matrices():
    cam_rotations = []
    # Flip Y axis sign + Rotate 90 degrees around the Y-axis
    adjustment = glm.rotate(np.pi / 2.0,
                            glm.vec3(0, 1, 0)) * \
                 glm.mat4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    for i in range(1, 5):
        rvec, _ = get_camera_extrinsic(i)
        cv_rotation, _ = cv2.Rodrigues(rvec)
        gl_rotation = adjustment * glm.mat4(cv_rotation[0][0], cv_rotation[1][0], cv_rotation[2][0], 0,
                                            cv_rotation[0][2], cv_rotation[1][2], cv_rotation[2][2], 0,
                                            cv_rotation[0][1], cv_rotation[1][1], cv_rotation[2][1], 0,
                                            0, 0, 0, 1)
        cam_rotations.append(gl_rotation)
    return cam_rotations
