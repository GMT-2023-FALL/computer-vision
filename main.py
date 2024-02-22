import cv2 as cv
from Calibrate.calibrate import calibrate_camera

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
    calibrate_camera(config=CONFIG)
