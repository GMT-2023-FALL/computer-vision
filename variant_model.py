import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_

from create_dataset import get_train_data_loader, get_validation_data_loader
from draw_diagram import plot_metrics
from train_model import train_model, test_model


# Define the LeNet-5 model
class LeNet5Variant1(nn.Module):
    def __init__(self):
        super(LeNet5Variant1, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, padding=2)  # Changed from 6 to 12
        self.conv2 = nn.Conv2d(3, 12, kernel_size=5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
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


class LeNet5Variant2(nn.Module):
    def __init__(self):
        super(LeNet5Variant2, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, padding=2)  # Changed from 6 to 12
        self.conv2 = nn.Conv2d(3, 12, kernel_size=5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)  # Dropout层
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
        x = self.dropout(x)  # 在全连接层之间应用Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # 可以再次应用Dropout或者只在一个全连接层后使用
        x = self.fc3(x)
        return x


class LeNet5Variant3(nn.Module):
    def __init__(self):
        super(LeNet5Variant3, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, padding=2)  # Changed from 6 to 12
        self.conv2 = nn.Conv2d(3, 12, kernel_size=5)
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)  # Dropout层
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
            x = F.max_pool2d(F.leaky_relu(self.conv1(x), negative_slope=0.01), (2, 2))
            x = F.max_pool2d(F.leaky_relu(self.conv2(x), negative_slope=0.01), (2, 2))
            x = torch.flatten(x, 1)
            x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
            x = self.dropout(x)
            x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
            x = self.dropout(x)
            x = self.fc3(x)
            return x


class LeNet5Variant5(nn.Module):
    def __init__(self):
        super(LeNet5Variant5, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=7, padding=3) # 核大小从5改为7，padding从2调整为3
        self.conv2 = nn.Conv2d(12, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.5)  # Dropout层
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x), negative_slope=0.01), (2, 2))
        x = F.max_pool2d(F.leaky_relu(self.conv2(x), negative_slope=0.01), (2, 2))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)
        return x


def get_variant_criterion(_model_version):
    return nn.CrossEntropyLoss()


def create_variant_model(_model_version):
    # init
    learning_rate = 0.001
    # Create the model instance
    if _model_version == 1:
        _model = LeNet5Variant1()
    elif _model_version == 2:
        _model = LeNet5Variant2()
    elif _model_version == 3:
        _model = LeNet5Variant3()
    elif _model_version == 4:
        learning_rate = 0.005
        _model = LeNet5Variant3()
    else:
        raise ValueError("Invalid model version")

    # Check if GPU is available and move the model to GPU if possible
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device_name}")
    device = torch.device(device_name)
    _model.to(device)

    # Define your loss function and optimizer
    optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)

    # Define the number of epochs
    num_epochs = 10

    # Run the training
    history, trained_model = train_model(_model, get_train_data_loader(),
                                get_validation_data_loader(),
                                get_variant_criterion(_model_version),
                                optimizer,
                                num_epochs=num_epochs)

    plot_metrics(history, "Variant Model {}".format(_model_version))

    # Evaluate the model
    test_model(trained_model, get_validation_data_loader(), get_variant_criterion(_model_version))

    # Save the model
    torch.save(trained_model, f'variant_model{_model_version}.pth')

    # save the history
    pickle.dump(history, open(f'variant_model{_model_version}_history.pkl', 'wb'))