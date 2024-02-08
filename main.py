import cv2
import numpy as np

# Initialize global variables
clicked_points = []
dimensions = (8, 11)  # Example for a 8x8 chessboard


def mouse_callback(event, x, y,  flags, param):
    global clicked_points, dimensions
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        cv2.circle(resized_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Chessboard', resized_image)
        if len(clicked_points) == 4:
            interpolate_chessboard_corners(clicked_points, dimensions)


def interpolate_chessboard_corners(corners, _dimensions):
    # Assuming corners are provided in the order: top-left, top-right, bottom-right, bottom-left
    top_left, top_right, bottom_right, bottom_left = corners

    # Calculate all corner points based on the four provided
    all_points = []
    for i in range(_dimensions[1]):  # Vertical points
        for j in range(_dimensions[0]):  # Horizontal points
            # Linearly interpolate the coordinates
            weight_x = j / (_dimensions[0] - 1)
            weight_y = i / (_dimensions[1] - 1)

            # Interpolate the top and bottom positions
            top_x = (1 - weight_x) * top_left[0] + weight_x * top_right[0]
            top_y = (1 - weight_x) * top_left[1] + weight_x * top_right[1]
            bottom_x = (1 - weight_x) * bottom_left[0] + weight_x * bottom_right[0]
            bottom_y = (1 - weight_x) * bottom_left[1] + weight_x * bottom_right[1]

            # Final interpolated position
            x = (1 - weight_y) * top_x + weight_y * bottom_x
            y = (1 - weight_y) * top_y + weight_y * bottom_y

            point = (int(x), int(y))
            all_points.append(point)

            # Draw the point on the image
            cv2.circle(resized_image, point, 5, (0, 255, 0), -1)
            # Put the coordinates text near the point
            cv2.putText(resized_image, str(point), (int(x + 5), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Chessboard', resized_image)

# def interpolate_chessboard_corners(corners, _dimensions):
#     # Assuming corners are provided in the order: top-left, top-right, bottom-right, bottom-left
#     top_left, top_right, bottom_right, bottom_left = corners
#
#     # Calculate all corner points based on the four provided
#     all_points = []
#     for i in range(_dimensions[0]):
#         for j in range(_dimensions[1]):
#             # Linearly interpolate the coordinates
#             x = (top_left[0] * (1 - i / (_dimensions[0] - 1)) + top_right[0] * (i / (_dimensions[0] - 1)))
#             y = (top_left[1] * (1 - j / (_dimensions[1] - 1)) + bottom_left[1] * (j / (_dimensions[1] - 1)))
#             all_points.append((int(x), int(y)))
#
#     # Draw all points on the image
#     for point in all_points:
#         cv2.circle(resized_image, point, 3, (0, 255, 0), -1)
#
#     cv2.imshow('Chessboard', resized_image)


def get_screen_resolution():
    from screeninfo import get_monitors
    monitor = get_monitors()[0]
    return monitor.width, monitor.height


def resize_image_to_screen(image, screen_width, screen_height, buffer=50):
    # 计算缩放比例
    scale_width = (screen_width - buffer) / image.shape[1]
    scale_height = (screen_height - buffer) / image.shape[0]
    scale = min(scale_width, scale_height)

    # 确保图片不会放大，只缩小
    scale = min(scale, 1.0)

    # 计算新的图片尺寸
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    dim = (new_width, new_height)

    # 缩放图片
    img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return img


# 获取屏幕分辨率
screen_width, screen_height = get_screen_resolution()


# Load the image
image = cv2.imread('calib-checkerboard.png')

# Set mouse callback function
cv2.namedWindow('Chessboard')
cv2.setMouseCallback('Chessboard', mouse_callback)

# 根据屏幕分辨率缩放图片
resized_image = resize_image_to_screen(image, screen_width, screen_height)

# Display the image
cv2.imshow('Chessboard', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
