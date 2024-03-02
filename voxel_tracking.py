import glob

import cv2
import numpy as np

from Calibrate.calibrate import get_object_points
from manually_process.utils import get_image_points, save_params, draw_axis, manually_find_corner_points, \
    get_camera_intrinsic


def generate_the_average_frame(each_file_path, index, _video_path):
    print("Processing... ", index + 1, each_file_path)
    # read the video
    video = cv2.VideoCapture(each_file_path)
    # Get the number of frames
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_frame = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if avg_frame is None:
            avg_frame =  np.zeros_like(frame, dtype=np.float64)
        avg_frame += frame.astype(np.float64)
    avg_frame /= frames
    avg_frame = np.round(avg_frame).astype(np.uint8)
    cv2.imshow("chessboard_screenshot_cam_{}.png".format(index), avg_frame)
    cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()
    cam_num = each_file_path.split('_')[-1]
    cam_num = cam_num.split('.')[0]
    cv2.imwrite(_video_path + "/chessboard_screenshot_cam_{}.png".format(cam_num), avg_frame)
    print("save the screen shot: ", "chessboard_screenshot_cam_{}.png".format(cam_num))


def generate_screenshots(_video_path, _screenshot_save_path):
    # read all files end with .avi in the video path
    for index, each_file_path in enumerate(glob.glob(_video_path + "/*.avi")):
        generate_the_average_frame(each_file_path, index, _screenshot_save_path)


def load_intrinsic(parameter_save_path):
    mtx = np.load('{}/mtx.npy'.format(parameter_save_path))
    dist = np.load('{}/dist.npy'.format(parameter_save_path))
    return mtx, dist


def load_extrinsic(parameter_save_path):
    rvecs = np.load('{}/rvecs.npy'.format(parameter_save_path))
    tvecs = np.load('{}/tvecs.npy'.format(parameter_save_path))
    return rvecs, tvecs


def calibrate_camera_from_screenshots(_screenshot_save_path):
    for each_screenshot in glob.glob(_screenshot_save_path + "/*.png"):
        cam_num = each_screenshot.split('_')[-1]
        cam_num = cam_num.split('.')[0]
        print("Finding the corner... ", each_screenshot)
        from main import CONFIG
        obj_points = get_object_points(CONFIG)
        img_points = manually_find_corner_points(each_screenshot, CONFIG)
        parameter_save_path = "4persons/camera_parameters/cam{}".format(cam_num)
        mtx, dist = get_camera_intrinsic(cam_num)
        frame = cv2.imread(each_screenshot)
        ret, rvecs, tvecs = cv2.solvePnP(obj_points, img_points, mtx, dist)
        R, _ = cv2.Rodrigues(rvecs)
        np.save('{}/rvecs.npy'.format(parameter_save_path,), rvecs)
        np.save('{}/tvecs.npy'.format(parameter_save_path), tvecs)
        # Project 3D points to image plane
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
        axis_points, jac = cv2.projectPoints(axis, R, tvecs, mtx, dist)
        draw_axis(frame, np.int32(img_points[0][0]), axis_points)
        cv2.imshow("calibrate cam{}".format(cam_num), frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "4persons/extrinsics"
    screenshot_save_path = "4persons/screenshots"
    # generate_screenshots(video_path, screenshot_save_path)
    calibrate_camera_from_screenshots(screenshot_save_path)


