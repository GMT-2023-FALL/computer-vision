import glob

import numpy as np
import cv2 as cv

from utils.utils import get_frames_from_video
from manually_process.utils import get_image_points, save_params, reduce_light_reflections, draw_chessboard, \
    resize_image_to_screen, get_screen_resolution


def get_object_points(config):
    object_points = np.zeros((config['width'] * config['height'], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:config['width'], 0:config['height']].T.reshape(-1, 2)
    return object_points


def get_img_and_obj_points_from_images_folder(images_path, config):
    image_points = []
    object_points = []
    counter = 0
    images_list = glob.glob(images_path + '/*.png')
    for file_name in images_list:
        if counter >= 25:
            break
        img_points = get_image_points(file_name, config)
        if len(img_points) != 0:
            image_points.append(get_image_points(file_name, config))
            object_points.append(get_object_points(config))
            counter += 1
    return image_points, object_points


def render_axis_on_video(video_path, config, width=644, height=486, step=1):
    w = config['width']
    h = config['height']
    # # 打开视频文件
    video = cv.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("can not open the camera. Exiting...")
        exit()

    # 设置分辨率
    video.set(cv.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # generate object points
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    while True:
        # 逐帧捕获
        ret, frame = video.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("can not receive frame (stream end?). Exiting...")
            break
        cv.imwrite('data/cam{}/checkerboard.png'.format(step), frame)
        break

    frame = cv.imread('data/cam{}/checkerboard.png'.format(step))
    # draw the chessboard
    image_points = get_image_points('data/cam{}/checkerboard.png'.format(step), config)
    draw_chessboard(frame, frame, get_object_points(config), step, config, manually_detected_corners=image_points)
    cv.imshow('Webcam Capture for Online Run {}'.format(step), frame)
    cv.waitKey(0)
    cv.imwrite('data/cam{}/checkerboard_with_axis.png'.format(step), frame)
    cv.destroyAllWindows()


def calibrate_camera(config):
    for index in range(1, 5):
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
        # rendering the world axis on the video
        video_to_render_path = 'data/cam{}/checkerboard.avi'.format(index)
        render_axis_on_video(video_to_render_path, config, step=index)
