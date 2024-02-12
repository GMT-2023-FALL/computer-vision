import cv2
import numpy as np

# Initialize global variables
clicked_points = []
image_points = None

original_image = None


def mouse_callback(event, x, y, flags, param):
    global clicked_points, original_image
    _image = param['image']

    # 如果点击鼠标左键且点的数量少于4，添加点并在图上绘制
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        cv2.circle(_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Chessboard', _image)
        if len(clicked_points) == 4:
            # 假设已经定义了interpolate_chessboard_corners函数
            interpolate_chessboard_corners(clicked_points, param)

    # 如果点击鼠标右键，清空所有点并从原始图像重置显示
    elif event == cv2.EVENT_RBUTTONDOWN:
        clicked_points.clear()  # 清空已点击的点
        _image[:] = original_image.copy()  # 从原始图像复制来重置图像
        cv2.imshow('Chessboard', _image)  # 显示重置后的图像


def clear():
    global clicked_points, image_points
    clicked_points = []
    # image_points = None


def interpolate_chessboard_corners(corners, param):
    global image_points
    _image = param['image']
    criteria = param['config']['criteria']
    w = param['config']['width']
    h = param['config']['height']
    world_coord = np.array([[0, (h - 1) * 23], [(w - 1) * 23, (h - 1) * 23], [(w - 1) * 23, 0], [0, 0]], np.float32)
    coordinates_array = np.array(corners, np.float32)
    M = cv2.getPerspectiveTransform(world_coord, coordinates_array)
    sub_work_coord = np.zeros((w * h, 2), np.float32)
    sub_work_coord[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * 23
    sub_work_coord = np.array(sub_work_coord, np.float32)
    res = cv2.perspectiveTransform(sub_work_coord.reshape(-1, 1, 2), M)
    # show the chessboard grid
    gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    corners = np.array(res, np.float32)
    image_points = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Return the array of points
    cv2.drawChessboardCorners(_image, (w, h), image_points, True)
    cv2.imshow('Chessboard', _image)


def get_screen_resolution():
    from screeninfo import get_monitors
    monitor = get_monitors()[0]
    return monitor.width, monitor.height


def resize_image_to_screen(_image, _screen_width, _screen_height, buffer=50):
    # 计算缩放比例
    scale_width = (_screen_width - buffer) / _image.shape[1]
    scale_height = (_screen_height - buffer) / _image.shape[0]
    scale = min(scale_width, scale_height)

    # 确保图片不会放大，只缩小
    scale = min(scale, 1.0)

    # 计算新的图片尺寸
    new_width = int(_image.shape[1] * scale)
    new_height = int(_image.shape[0] * scale)
    dim = (new_width, new_height)

    # 缩放图片
    img = cv2.resize(_image, dim, interpolation=cv2.INTER_AREA)
    return img


def custom_process(img_path, config):
    global original_image
    clear()
    print("Custom process for image: ", img_path.split('\\')[-1])

    # 获取屏幕分辨率
    screen_width, screen_height = get_screen_resolution()

    # Load the _image
    image = cv2.imread(img_path)

    # 根据屏幕分辨率缩放图片
    resized_image = resize_image_to_screen(image, screen_width, screen_height)
    original_image = resized_image.copy()

    # Set mouse callback function
    cv2.namedWindow('Chessboard')
    cv2.setMouseCallback('Chessboard', mouse_callback, {
        "image": resized_image,
        "config": config,
        'file_name': img_path.split('\\')[-1]
    })

    # Display the _image
    cv2.imshow('Chessboard', resized_image)
    cv2.waitKey(0)
    cv2.imwrite("{}/result_{}".format(config["human_detected_images_folder_path"], img_path.split('\\')[-1]),
                resized_image)
    cv2.destroyAllWindows()
    print("Custom process finished for image: ", img_path)
    return image_points
