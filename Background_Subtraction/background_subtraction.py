from multiprocessing import Pool

import cv2
import numpy as np


def create_background_model(video_file, output_file_path):
    # Initialize background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Initialize variables
    cap = cv2.VideoCapture(video_file)
    background_model = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)

        # Get background image
        if background_model is None:
            background_model = frame.copy()
        else:
            background_model[fg_mask == 0] = frame[fg_mask == 0]

        # Display the result (optional)
        cv2.imshow('Background Model', background_model)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()
    cv2.imwrite(output_file_path + '/background_image.png', background_model)


def calculate_similarity(mask1, mask2):
    """Calculate similarity between two binary masks using XOR."""
    xor_result = cv2.bitwise_xor(mask1, mask2)
    similarity = np.count_nonzero(xor_result == 0) / xor_result.size
    return similarity


def calculate_similarity_parallel(threshold, diff_hsv, manual_segmentation):
    h, s, v = threshold
    _, thresh_hue = cv2.threshold(diff_hsv[:, :, 0], h, 255, cv2.THRESH_BINARY)
    _, thresh_saturation = cv2.threshold(diff_hsv[:, :, 1], s, 255, cv2.THRESH_BINARY)
    _, thresh_value = cv2.threshold(diff_hsv[:, :, 2], v, 255, cv2.THRESH_BINARY)
    foreground_mask = cv2.bitwise_or(thresh_hue, cv2.bitwise_or(thresh_saturation, thresh_value))
    similarity = calculate_similarity(foreground_mask, manual_segmentation)
    return threshold, similarity


def calculate_similarity_wrapper(args):
    # Wrapper function to handle arguments for calculate_similarity_parallel
    threshold, diff_hsv, manual_segmentation = args
    return calculate_similarity_parallel(threshold, diff_hsv, manual_segmentation)


def find_optimal_thresholds(video_image, background_image, manual_segmentation):
    # Read the background image and manual segmentation
    background_model = cv2.imread(background_image)
    frame_hsv = cv2.cvtColor(cv2.imread(video_image), cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
    manual_segmentation = cv2.imread(manual_segmentation, cv2.IMREAD_GRAYSCALE)

    # Initialize variables
    best_similarity = 0
    optimal_thresholds = None
    diff_hsv = cv2.absdiff(frame_hsv, background_hsv)

    # Generate combinations of thresholds
    thresholds = [(h, s, v) for h in range(0, 256, 10)
                            for s in range(0, 256, 10)
                            for v in range(0, 256, 10)]

    # Use multiprocessing Pool to parallelize the calculations
    with Pool() as pool:
        args = [(threshold, diff_hsv, manual_segmentation) for threshold in thresholds]
        results = pool.map(calculate_similarity_wrapper, args)

    # Find optimal thresholds based on highest similarity
    for threshold, similarity in results:
        if similarity > best_similarity:
            best_similarity = similarity
            optimal_thresholds = threshold

    return optimal_thresholds, best_similarity


def post_process_foreground(foreground_image, kernel_size=1, iterations=3):
    # 连通组件标记，去除较小的不连通部分，保留主要的前景部分
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_image)

    min_area = 4800  # 设置最小面积阈值，用于去除小连通区域
    for i in range(1, stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            foreground_image[labels == i] = 0  # 将面积小于阈值的连通区域置为背景

    # 对前景掩码再次进行膨胀操作，以填补前景中的空洞
    kernel_dilate = np.ones((6, 6), np.uint8)
    foreground_mask = cv2.dilate(foreground_image, kernel_dilate, iterations=1)

    # 对处理后的前景掩码进行腐蚀操作，以去除较小的孤立像素
    kernel_erode = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.erode(foreground_mask, kernel_erode, iterations=1)

    return foreground_mask


def background_subtraction(video_file, background_image, output_file_path, index):
    # Read the background image
    background_model = cv2.imread(background_image)
    # Automatically Determine threshold values
    video_image = "data/cam{}/segment/video.jpg".format(index)
    manual_segmentation = "data/cam{}/video_manual_segment.png".format(index)
    optimal_thresholds, best_similarity = find_optimal_thresholds(video_image, background_image, manual_segmentation)
    print("Optimal thresholds: {} Similarity: {}".format(optimal_thresholds, best_similarity))
    hue_threshold, saturation_threshold, value_threshold = optimal_thresholds
    # hue_threshold, saturation_threshold, value_threshold = 110, 180, 30
    print("Hue threshold: {} Saturation threshold: {} Value threshold: {}".format(hue_threshold, saturation_threshold,
                                                                                  value_threshold))

    # Convert the background image to HSV color space
    background_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)

    # Read the video
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    foreground_mask = None
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to HSV color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the absolute difference between the frame and the background model
        diff_hsv = cv2.absdiff(frame_hsv, background_hsv)

        # Threshold the differences for the Hue, Saturation, and Value channels
        _, thresh_hue = cv2.threshold(diff_hsv[:, :, 0], hue_threshold, 255, cv2.THRESH_BINARY)
        _, thresh_saturation = cv2.threshold(diff_hsv[:, :, 1], saturation_threshold, 255, cv2.THRESH_BINARY)
        _, thresh_value = cv2.threshold(diff_hsv[:, :, 2], value_threshold, 255, cv2.THRESH_BINARY)

        # Combine the thresholded channels to determine foreground and background
        foreground_mask = cv2.bitwise_or(thresh_hue, cv2.bitwise_or(thresh_saturation, thresh_value))

    cap.release()
    foreground_mask = post_process_foreground(foreground_mask)
    cv2.imwrite(output_file_path, foreground_mask)


def background_process(config):
    for index in range(1, 5):
        background_video_path = "data/cam{}/background.avi".format(index)
        model_path = "data/cam{}".format(index)
        create_background_model(background_video_path, model_path)
        print("Background model for camera {} created.".format(index))
        background_img_path = model_path + "/background_image.png".format(index)
        video_img_path = model_path + "/video.avi"
        foreground_img_path = model_path + "/foreground.png"
        background_subtraction(video_img_path, background_img_path, foreground_img_path, index)
