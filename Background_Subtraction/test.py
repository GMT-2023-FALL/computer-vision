def automatic_thresholding(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to 2D array of pixels (rows * cols, channels)
    pixels = hsv_image.reshape((-1, 3)).astype(np.float32)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2  # Two clusters: foreground and background

    # Hue channel
    hue_pixels = pixels[:, 0].reshape((-1, 1))
    _, _, hue_centers = cv2.kmeans(hue_pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    hue_thresh = np.mean(hue_centers)

    # Saturation channel
    sat_pixels = pixels[:, 1].reshape((-1, 1))
    _, _, sat_centers = cv2.kmeans(sat_pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    sat_thresh = np.mean(sat_centers)

    # Value channel
    val_pixels = pixels[:, 2].reshape((-1, 1))
    _, _, val_centers = cv2.kmeans(val_pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    val_thresh = np.mean(val_centers)

    return hue_thresh, sat_thresh, val_thresh


def foreground_segmentation(image, hue_thresh, sat_thresh, val_thresh):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert thresholds to uint8
    hue_thresh = int(hue_thresh)
    sat_thresh = int(sat_thresh)
    val_thresh = int(val_thresh)

    # Thresholding for each channel
    hue_mask = cv2.inRange(hsv_image[:, :, 0], 0, hue_thresh)
    sat_mask = cv2.inRange(hsv_image[:, :, 1], 0, sat_thresh)
    val_mask = cv2.inRange(hsv_image[:, :, 2], 0, val_thresh)

    # Combine masks
    fg_mask = cv2.bitwise_not(hue_mask, cv2.bitwise_and(sat_mask, val_mask))

    return fg_mask


def process_video(video_file, background_image):
    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read first frame to get image size
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Get image size
    height, width = frame.shape[:2]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Automatic thresholding
        # background_image = cv2.imread(background_image)
        hue_thresh, sat_thresh, val_thresh = automatic_thresholding(background_image)

        # Foreground segmentation
        fg_mask = foreground_segmentation(frame, hue_thresh, sat_thresh, val_thresh)

        # Display foreground mask
        cv2.imshow('Foreground Mask', fg_mask)

        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()