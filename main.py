import cv2

from OfflineRun.offline_run_1 import offlineRun1

WIDTH = 9
HEIGHT = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
IMAGES_FOLDER_PATH = 'images/samples/*.jpg'
AUTO_DETECTED_IMAGES_FOLDER_PATH = 'images/result/auto'
HUMAN_DETECTED_IMAGES_FOLDER_PATH = 'images/result/manual'
image_size = (1280, 720)
CONFIG = {
    "width": WIDTH,
    "height": HEIGHT,
    "criteria": criteria,
    "images_folder_path": IMAGES_FOLDER_PATH,
    "auto_detected_images_folder_path": AUTO_DETECTED_IMAGES_FOLDER_PATH,
    "human_detected_images_folder_path": HUMAN_DETECTED_IMAGES_FOLDER_PATH,
    "image_size": image_size
}


# offline phase, 3 runs
def offlinePhase(config=CONFIG):
    offlineRun1(config)
    # offlineRun(2)
    # offlineRun(3)


# online phase, 3 runs
def onlinePhase():
    pass
    # onlineRun(1)
    # onlineRun(2)
    # onlineRun(3)


if __name__ == "__main__":
    offlinePhase()
    onlinePhase()
