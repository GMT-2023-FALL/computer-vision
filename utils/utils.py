import cv2 as cv


# Function to get the frames from the video and save them as images
def get_frames_from_video(video_path, frame_rate, output_path):
    # Open the video
    video = cv.VideoCapture(video_path)

    # Get the number of frames
    frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    # Get the width and height of the frames
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Get the number of digits in the number of frames
    digits = len(str(frames))

    # Get the number of frames to skip
    skip = int(video.get(cv.CAP_PROP_FPS) / frame_rate)

    # Initialize the frame counter
    count = 0

    print(frames, width, height, digits, skip, video.get(cv.CAP_PROP_FPS))

    # Loop through the frames
    while video.isOpened():
        # Read the frame
        ret, frame = video.read()

        # Break if the frame is not read
        if not ret:
            break

        # Save the frame
        cv.imwrite(f'{output_path}/frame_{str(count).zfill(digits)}.png', frame)

        # Skip the frames
        for _ in range(skip):
            video.grab()

        # Increment the frame counter
        count += 1

    # Release the video
    video.release()