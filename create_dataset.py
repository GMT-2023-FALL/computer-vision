from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Define a transformation to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Fashion MNIST training data
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)

# Load the Fashion MNIST test data
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# Splitting the training data into training and validation sets (80/20 split)
train_size = int(0.8 * len(train_data))
validation_size = len(train_data) - train_size
train_dataset, validation_dataset = random_split(train_data, [train_size, validation_size])

# Creating data loaders for the datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


def get_train_data_loader(batch_size=32):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_validation_data_loader(batch_size=32):
    return DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)


def get_test_data_loader(batch_size=32):
    return DataLoader(test_data, batch_size=batch_size, shuffle=False)


# custom data set
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): CSV 文件的路径，包含图片的 ID 和对应的类型。
            img_dir (string): 包含所有图片的文件夹路径。
            transform (callable, optional): 一个可选的转换操作，用于对样本进行处理。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")  # 假设图片的格式是 .jpg
        image = Image.open(img_path)  # 确保图片是 RGB 格式
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


# # 定义与 FashionMNIST 相似的转换操作
custom_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建自定义数据集的实例，包括转换
dataset = CustomImageDataset(csv_file='merged.csv', img_dir='fashion_mnist', transform=custom_transform)

# # 创建 DataLoader
# custom_data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


def get_custom_data_loader(batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
