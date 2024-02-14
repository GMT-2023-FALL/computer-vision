import cv2

from manually_process.utils import get_webcam_snapshot


def onlineRun(run_number, _config):
    print(f"Online phase, run {run_number}")
    frame = get_webcam_snapshot(run_number, _config, _config['image_size'][0], _config['image_size'][1])
    if frame is not None:
        cv2.imwrite('images/webcam/webcam_result_run_{}.jpg'.format(run_number), frame)
    else:
        print("No frame captured from webcam")


