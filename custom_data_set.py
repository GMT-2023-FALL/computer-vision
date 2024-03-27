from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        """
        Args:
            csv_file (string): CSV 文件的路径，包含图片的 ID 和对应的类型。
            img_dir (string): 包含所有图片的文件夹路径。
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_id = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")  # 假设图片的格式是 .jpg
        image = Image.open(img_path).convert('RGB')  # 确保图片是 RGB 格式
        label = self.img_labels.iloc[idx, 1]
        return image, label


# 创建自定义数据集的实例
dataset = CustomImageDataset(csv_file='merged.csv', img_dir='fashion_mnist/')

# 创建 DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)