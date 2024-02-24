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


def find_optimal_thresholds(video_image, background_image, manual_segmentation):
    # Read the background image and manual segmentation
    background_model = cv2.imread(background_image)
    # Convert the frame to HSV color space
    frame_hsv = cv2.cvtColor(cv2.imread(video_image), cv2.COLOR_BGR2HSV)

    # Convert the background image to HSV color space
    background_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)

    manual_segmentation = cv2.imread(manual_segmentation, cv2.IMREAD_GRAYSCALE)

    # Initialize variables to store optimal thresholds and similarity score
    best_similarity = 0
    optimal_thresholds = None
    # Calculate the absolute difference between the frame and the background model
    diff_hsv = cv2.absdiff(frame_hsv, background_hsv)

    # Iterate over different combinations of thresholds
    for hue_threshold in range(0, 256, 10):
        for saturation_threshold in range(0, 256, 10):
            for value_threshold in range(0, 256, 10):
                # Perform background subtraction using current thresholds
                _, thresh_hue = cv2.threshold(diff_hsv[:, :, 0], hue_threshold, 255, cv2.THRESH_BINARY)
                _, thresh_saturation = cv2.threshold(diff_hsv[:, :, 1], saturation_threshold, 255, cv2.THRESH_BINARY)
                _, thresh_value = cv2.threshold(diff_hsv[:, :, 2], value_threshold, 255, cv2.THRESH_BINARY)
                foreground_mask = cv2.bitwise_or(thresh_hue, cv2.bitwise_or(thresh_saturation, thresh_value))

                # Calculate similarity between algorithm's output and manual segmentation
                similarity = calculate_similarity(foreground_mask, manual_segmentation)

                # Update optimal thresholds if similarity is higher
                if similarity > best_similarity:
                    best_similarity = similarity
                    optimal_thresholds = (hue_threshold, saturation_threshold, value_threshold)

    return optimal_thresholds, best_similarity


def post_process_foreground(foreground_image, kernel_size=5, iterations=2):
    # 定义腐蚀和膨胀的核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 对前景图像进行腐蚀操作，以去除小的白色斑点
    eroded = cv2.erode(foreground_image, kernel, iterations=iterations)

    # 对腐蚀后的图像进行膨胀操作，以恢复前景物体的形状和大小
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)

    return dilated


def background_subtraction(video_file, background_image, output_file_path, index):
    # Read the background image
    background_model = cv2.imread(background_image)
    # Automatically Determine threshold values
    video_image = "data/cam{}/segment/video.jpg".format(index)
    manual_segmentation = "data/cam{}/video_manual_segment.png".format(index)
    optimal_thresholds, best_similarity = find_optimal_thresholds( video_image, background_image, manual_segmentation)
    print("Optimal thresholds: {} Similarity: {}".format(optimal_thresholds, best_similarity))
    hue_threshold, saturation_threshold, value_threshold = optimal_thresholds
    # hue_threshold, saturation_threshold, value_threshold = 110, 180, 30
    print("Hue threshold: {} Saturation threshold: {} Value threshold: {}".format(hue_threshold, saturation_threshold, value_threshold))

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
    cv2.destroyAllWindows()
    foreground_mask = post_process_foreground(foreground_mask)
    cv2.imwrite(output_file_path, foreground_mask)


def background_process(config):
    for index in range(1, 5):
        background_video_path = "data/cam{}/background.avi".format(index)
        model_path = "data/cam{}".format(index)
        # create_background_model(background_video_path, model_path)
        # print("Background model for camera {} created.".format(index))
        background_img_path = model_path + "/background_image.png".format(index)
        video_img_path = model_path + "/video.avi"
        foreground_img_path = model_path + "/foreground.png"
        background_subtraction(video_img_path, background_img_path, foreground_img_path, index)
