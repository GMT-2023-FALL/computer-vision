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
