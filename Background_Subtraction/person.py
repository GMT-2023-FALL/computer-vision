import cv2


def convert_person_to_white(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将灰度图像转换为二值图像（阈值化）
    _, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 获取反转的二值图像（黑色变白色，白色变黑色）
    inverted = cv2.bitwise_not(thresholded)
    # 对反转后的二值图像再次进行颜色反转，黑色变为白色，白色变为黑色
    result = cv2.bitwise_not(inverted)
    cv2.imshow('inverted', result)
    cv2.waitKey(0)
    return result


if __name__ == '__main__':
    for i in range(1, 5):
        image_path = "E:/UU/cv_assignment/computer-vision/data/cam{}/segment/manual_segment.png".format(i)
        white_person_image = convert_person_to_white(image_path)
        cv2.imwrite('E:/UU/cv_assignment/computer-vision/data/cam{}/video_manual_segment.png'.format(i),
                    white_person_image)
