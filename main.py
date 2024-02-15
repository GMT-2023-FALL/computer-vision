import cv2
import numpy as np

from OfflineRun.offline_run import offlineRun
from onlineRun.online_run import onlineRun

WIDTH = 9
HEIGHT = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
IMAGES_FOLDER_PATH = 'images/samples'
IMAGE_FORMAT = '/*.jpg'
AUTO_DETECTED_IMAGES_FOLDER_PATH = 'images/result/auto'
HUMAN_DETECTED_IMAGES_FOLDER_PATH = 'images/result/manual'
image_size = (1280, 720)
square_size = 2
CONFIG = {
    "width": WIDTH,
    "height": HEIGHT,
    "criteria": criteria,
    "images_folder_path": IMAGES_FOLDER_PATH,
    "image_format": IMAGE_FORMAT,
    "auto_detected_images_folder_path": AUTO_DETECTED_IMAGES_FOLDER_PATH,
    "human_detected_images_folder_path": HUMAN_DETECTED_IMAGES_FOLDER_PATH,
    "image_size": image_size,
    "square_size": square_size
}


# offline phase, 3 runs
def offlinePhase(config):
    for i in range(1, 4):
        offlineRun(i, config)


# online phase, 3 runs
def onlinePhase(config):
    for i in range(1, 4):
        onlineRun(i, config)


if __name__ == "__main__":
    # offlinePhase(config=CONFIG)
    # onlinePhase(config=CONFIG)
    print(np.load('parameters/offline-run-3/mtx.npy'))
