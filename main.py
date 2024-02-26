import cv2
import cv2 as cv

from Background_Subtraction.background_subtraction import background_process
from Calibrate.calibrate import calibrate_camera
from Voxel_Restruction.voxel_restruction import generate_voxel_map
from manually_process.utils import get_camera_intrinsic, get_camera_extrinsic

WIDTH = 8
HEIGHT = 6
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
IMAGES_FOLDER_PATH = 'images/samples'
IMAGE_FORMAT = '/*.jpg'
AUTO_DETECTED_IMAGES_FOLDER_PATH = 'images/result/auto'
HUMAN_DETECTED_IMAGES_FOLDER_PATH = 'images/result/manual'
image_size = (644, 486)
square_size = 2
brightness_reduction = 15

CONFIG = {
    "width": WIDTH,
    "height": HEIGHT,
    "criteria": criteria,
    "images_folder_path": IMAGES_FOLDER_PATH,
    "image_format": IMAGE_FORMAT,
    "auto_detected_images_folder_path": AUTO_DETECTED_IMAGES_FOLDER_PATH,
    "human_detected_images_folder_path": HUMAN_DETECTED_IMAGES_FOLDER_PATH,
    "image_size": image_size,
    "square_size": square_size,
    "brightness_reduction": brightness_reduction
}


if __name__ == '__main__':
    # calibrate_camera(config=CONFIG)
    # background_process(config=CONFIG)
    generate_voxel_map(config=CONFIG)
    # cv2.imshow('image', cv2.imread('data/cam1/video_manual_segment.png', cv2.IMREAD_GRAYSCALE))
    # cv2.waitKey(0)
