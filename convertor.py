import os
from PIL import Image, ImageOps, ImageEnhance


# def process_images_for_fashion_mnist(source_folder, target_folder):
#     # Make sure the target folder exists
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#
#     # Loop through all files in the source folder
#     for file_name in os.listdir(source_folder):
#         if file_name.endswith(('.png', '.jpg', '.jpeg')):
#             # Construct the full file path
#             source_path = os.path.join(source_folder, file_name)
#             target_path = os.path.join(target_folder, file_name)
#
#             # Open the image
#             # img = Image.open(source_path).convert('L')
#             img = Image.open(source_path)
#             # Resize the image to 28x28
#             img_resized = img.resize((28, 28), Image.LANCZOS)
#
#             # Enhance the contrast if needed
#             # enhancer = ImageEnhance.Contrast(img_resized)
#             # img_contrast = enhancer.enhance(2)
#
#             img_resized.save(target_path, 'JPEG')
#
#             # # Invert the colors
#             # img_inverted = ImageOps.invert(img_contrast)
#             #
#             # # Save the processed image to the target folder
#             # img_inverted.save(target_path, 'JPEG')

def resize_image_with_padding(img, target_size=(28, 28)):
    # 计算最大的边，用于确定正方形新图像的尺寸
    max_side = max(img.size)

    # 创建一个正方形的白色背景图像
    new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))

    # 计算原图像在新图像中的位置
    left = (max_side - img.size[0]) // 2
    top = (max_side - img.size[1]) // 2
    right = (max_side + img.size[0]) // 2
    bottom = (max_side + img.size[1]) // 2

    # 将原图像粘贴到白色背景图像上
    new_img.paste(img, (left, top, right, bottom))

    # 等比例缩小图像，使用 Image.Resampling.LANCZOS
    img_resized = new_img.resize(target_size, Image.Resampling.LANCZOS)

    return img_resized


def process_images_for_fashion_mnist(source_folder, target_folder):
    # Make sure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full file path
            source_path = os.path.join(source_folder, file_name)
            target_path = os.path.join(target_folder, file_name)

            # Open the image
            img = Image.open(source_path)

            # Resize and pad the image if necessary to keep the aspect ratio
            img_resized = resize_image_with_padding(img)

            # Save the processed image to the target folder
            img_resized.save(target_path, 'JPEG')


if __name__ == '__main__':

    # Define the source and target folders
    source_folder = 'fashion_data/'
    target_folder = 'fashion_mnist/'

    # Process the images
    process_images_for_fashion_mnist(source_folder, target_folder)