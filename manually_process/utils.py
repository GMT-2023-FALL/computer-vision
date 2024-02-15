import os

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
    # image_project_points = None


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
    enhanced_img = reduce_light_reflections(_image)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
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


def manually_find_corner_points(img_path, config):
    if os.name == 'nt':
        file_name = img_path.split('\\')[-1]
    else:
        file_name = img_path.split('/')[-1]
    print("manually finding corners points for: ", file_name)
    global original_image
    clear()

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
    })

    # Display the _image
    cv2.imshow('Chessboard', resized_image)
    cv2.waitKey(0)
    cv2.imwrite("{}/result_{}".format(config["human_detected_images_folder_path"], file_name),
                resized_image)
    cv2.destroyAllWindows()


def save_params(_parameter_file_path, _object_points_list, _image_points_list, _dimension):
    # Filter out bad images
    filtered_object_points, filtered_image_points, ret, mtx, dist, rvecs, tvecs = filter_bad_images(
        _object_points_list, _image_points_list, _dimension
    )
    # Save the parameters to a file
    np.save('{}/mtx.npy'.format(_parameter_file_path), mtx)
    np.save('{}/dist.npy'.format(_parameter_file_path), dist)
    np.save('{}/rvecs.npy'.format(_parameter_file_path), rvecs)
    np.save('{}/tvecs.npy'.format(_parameter_file_path), tvecs)


def filter_bad_images(_object_points_list, _image_points_list, _dimension, max_reprojection_error=0.25):
    """
    Iteratively calibrates the camera using the provided object and image points,
    removing the image points with the highest reprojection error until all remaining
    images have a reprojection error below the specified threshold.

    Parameters:
    - _object_points_list: List of object points
    - _image_points_list: List of image points corresponding to object points
    - _dimension: Dimension (width, height) of the images
    - max_reprojection_error: Maximum allowed reprojection error

    Returns:
    - object_points_filtered: Filtered list of object points
    - image_points_filtered: Filtered list of image points
    - ret: The overall RMS re-projection error
    - mtx: Camera matrix
    - dist: Distortion coefficients
    - rvecs: Rotation vectors
    - tvecs: Translation vectors
    """
    # Ensure the lists are numpy arrays for easier indexing
    object_points_array = np.array(_object_points_list)
    image_points_array = np.array(_image_points_list)
    count = 0
    while True:
        # Calibrate the camera using the current sets of points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_array, image_points_array, _dimension, None, None)

        # Calculate the reprojection errors for each image
        max_error = 0
        max_error_index = -1
        for i in range(len(object_points_array)):
            imgpoints2, _ = cv2.projectPoints(object_points_array[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(image_points_array[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            # print("Reprojection error for image {}: {}".format(i, error))
            if error > max_error:
                max_error = error
                max_error_index = i

        # If the max error is below the threshold or we have one image left, break
        if max_error < max_reprojection_error or len(object_points_array) == 1:
            break

        # Remove the point set with the highest error
        object_points_array = np.delete(object_points_array, max_error_index, axis=0)
        image_points_array = np.delete(image_points_array, max_error_index, axis=0)
        count += 1
        print("Removed image with projection error {}".format(max_error))

    print("Filtered {} images with high projection error".format(count))
    return object_points_array.tolist(), image_points_array.tolist(), ret, mtx, dist, rvecs, tvecs


def reduce_light_reflections(image, brightness_reduction=5):
    # 将图片从BGR转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 分割HSV通道
    h, s, v = cv2.split(hsv)

    # 减少亮度通道的亮斑（可通过降低亮度值实现）
    v = np.clip(v - brightness_reduction, 0, 255)

    # 合并通道
    final_hsv = cv2.merge((h, s, v))

    # 将HSV转换回BGR色彩空间
    final_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final_image


def get_image_points(file_name, _config):
    # if os env is macOS, then use the following code to get the file name
    pattern_size = (_config['width'], _config['height'])
    auto_detected_images_folder_path = _config['auto_detected_images_folder_path']
    images_points = []
    # detect the os env
    if os.name == 'nt':
        file_name_index = file_name.split('\\')[-1]
    else:
        file_name_index = file_name.split('/')[-1]
    img = cv2.imread(file_name)

    enhanced_img = reduce_light_reflections(img)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Finding sub-pixel corners based on the original corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _config['criteria'])
        # Draw and save the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imwrite("{}/result_{}".format(auto_detected_images_folder_path, file_name_index), img)
        print("Auto Detected: {}".format(file_name_index))
        images_points = corners2
    else:
        manually_find_corner_points(file_name, _config)
        images_points = image_points
    cv2.destroyAllWindows()
    return images_points


def get_camera_intrinsic(_config, step):
    parameter_file_path = 'parameters/offline-run-{}'.format(step)
    mtx = np.load('{}/mtx.npy'.format(parameter_file_path))
    dist = np.load('{}/dist.npy'.format(parameter_file_path))
    return mtx, dist


def get_camera_extrinsic(_config, step):
    parameter_file_path = 'parameters/offline-run-{}'.format(step)
    rvecs = np.load('{}/rvecs.npy'.format(parameter_file_path))
    tvecs = np.load('{}/tvecs.npy'.format(parameter_file_path))
    return rvecs, tvecs


def draw_cube(frame, origin, step, _config, rvecs, tvecs, mtx, dist):
    criteria = _config['criteria']
    # 定义立方体的8个顶点在世界坐标中的位置
    # 假设棋盘格的大小为单位长度，立方体的底面与棋盘格的四个角对齐
    objp_cube = np.float32([[0, 0, 0],
                            [0, 1, 0],
                            [1, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1],
                            [0, 1, -1],
                            [1, 1, -1],
                            [1, 0, -1]])

    # 调整立方体的尺寸和位置，使其放置在棋盘的右上角
    # 你可以根据需要调整这些值
    cube_size = _config['square_size']
    objp_cube = objp_cube * cube_size
    # objp_cube[:, :2] += origin  # 将立方体底面平移到棋盘的第一个角点

    # 投影立方体顶点到图像平面
    imgpts, _ = cv2.projectPoints(objp_cube, rvecs, tvecs, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # 绘制底面的边
    for i in range(4):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[(i + 1) % 4]), (0, 255, 255), 3)

    # 绘制顶面的边
    for i in range(4, 8):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[(i + 1 - 4) % 4 + 4]), (0, 255, 255), 3)

    # 你已有的绘制立柱（垂直边）的代码
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 255), 3)


def draw_chessboard(enhanced_img, frame, object_points, step, _config):
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (_config["width"], _config["height"]), None,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_NORMALIZE_IMAGE
                                             + cv2.CALIB_CB_FAST_CHECK)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _config["criteria"])
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (_config["width"], _config["height"]), corners2, ret)
        # get the camera intrinsic parameters
        mtx, dist = get_camera_intrinsic(_config, step)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(object_points, corners2, mtx, dist)
        # Project 3D points to image plane
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])
        axis_points, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        draw_axis(frame, np.int32(corners2[0][0]), axis_points)
        # draw a cube and xyz axis on the chessboard
        draw_cube(frame, np.int32(corners2[0][0]), step, _config, rvecs, tvecs, mtx, dist)


def draw_axis(img, origin, image_project_points, scale=1.5):
    """
    Draw 3D axis on the image, with the ability to extend the axis length.

    Parameters:
    - img: The image where the axis will be drawn.
    - origin: The origin point where the axis lines start. This is a 2D point (x, y).
    - image_project_points: The end points of the 3 axis lines in image plane. This is a numpy array of shape (3,1,2).
    - scale: Factor to scale the length of the axis. Default is 1.5.
    """
    # Axis colors
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors for X, Y, Z axis respectively
    # Draw axis lines
    for i in range(3):
        pt1 = (origin[0], origin[1])
        pt2 = (int(image_project_points[i][0][0]), int(image_project_points[i][0][1]))
        # Calculate direction vector
        direction = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
        # Scale the direction vector
        direction = direction * scale
        # Calculate new end point
        pt2_scaled = (pt1[0] + int(direction[0]), pt1[1] + int(direction[1]))
        # Draw extended axis line
        cv2.line(img, pt1, pt2_scaled, color[i], 5)


def get_webcam_snapshot(step, _config, width=1280, height=720):
    w = _config['width']
    h = _config['height']
    # 打开默认的相机
    cap = cv2.VideoCapture(0)

    # 检查相机是否成功打开
    if not cap.isOpened():
        print("can not open the camera. Exiting...")
        exit()

    # 设置相机分辨率为1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    captured_frame = None  # 初始化一个变量来存储捕获的帧

    # generate object points
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("can not receive frame (stream end?). Exiting...")
            break
        enhanced_img = reduce_light_reflections(frame)
        # draw the chessboard
        draw_chessboard(enhanced_img, frame, objp, step, _config)
        cv2.imshow('Webcam Capture for Online Run {}'.format(1), frame)
        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord(' '):
            captured_frame = frame
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放相机资源
    cap.release()
    cv2.destroyAllWindows()

    return captured_frame  # 返回捕获的帧
