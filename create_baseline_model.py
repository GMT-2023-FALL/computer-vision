import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_

from create_dataset import get_train_data_loader, get_validation_data_loader
from draw_diagram import plot_metrics
from train_model import train_model, test_model


# Define the LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # First convolutional layer with padding=2 to adjust for input size
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        # Fully connected layers with activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_baseline_model(num_epochs=10):
    # Create the model instance
    model = LeNet5()

    # Check if GPU is available and move the model to GPU if possible
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device_name}")
    device = torch.device(device_name)
    model.to(device)

    # Define your loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 定义训练的轮数
    num_epochs = 10

    # 运行训练
    history, trained_model = train_model(model, get_train_data_loader(), get_validation_data_loader(), get_baseline_criterion(),
                                optimizer,
                                num_epochs=num_epochs)

    print(history)
    plot_metrics(history, "Baseline Model")
    # 评估模型
    test_model(trained_model, get_validation_data_loader(), get_baseline_criterion())

    # 保存模型
    torch.save(trained_model, 'baseline_model.pth')




def get_baseline_criterion():
    return nn.CrossEntropyLoss()
