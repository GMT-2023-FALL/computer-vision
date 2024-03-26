# import torch
#
#
# def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10):
#     # 确保模型在训练模式下运行
#     model.train()
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct_predictions = 0
#         device_name = "cuda" if torch.cuda.is_available() else "cpu"
#         for inputs, labels in train_loader:
#
#             inputs, labels = inputs.to(device_name), labels.to(device_name)
#
#             # 清除梯度
#             optimizer.zero_grad()
#
#             # 前向传播
#             outputs = model(inputs)
#
#             # 计算损失
#             loss = criterion(outputs, labels)
#
#             # 反向传播
#             loss.backward()
#
#             # 更新模型参数
#             optimizer.step()
#
#             # 统计
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct_predictions += (predicted == labels).sum().item()
#
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = correct_predictions / len(train_loader.dataset)
#
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
#
#         # 每个epoch结束后在验证集上评估模型
#         validate_model(model, validation_loader, criterion)
#
#     return model
#
#
# def validate_model(model, validation_loader, criterion):
#     # 确保模型在评估模式下运行
#     model.eval()
#     validation_loss = 0.0
#     correct_predictions = 0
#     device_name = "cuda" if torch.cuda.is_available() else "cpu"
#     with torch.no_grad():
#         for inputs, labels in validation_loader:
#
#             inputs, labels = inputs.to(device_name), labels.to(device_name)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             validation_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct_predictions += (predicted == labels).sum().item()
#
#     validation_loss /= len(validation_loader.dataset)
#     validation_acc = correct_predictions / len(validation_loader.dataset)
#     print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_acc:.4f}\n')
#
#
def test_model(model, test_loader, criterion):
    model.eval()  # 确保模型在评估模式下
    test_loss = 0.0
    correct_predictions = 0
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():  # 在评估模式下，我们不需要计算梯度
        for inputs, labels in test_loader:

            inputs, labels = inputs.to(device_name), labels.to(device_name)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct_predictions / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')


import torch
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化记录指标的字典
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()  # 确保模型在训练模式
        running_loss = 0.0
        correct_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)  # 注意乘以batch size
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # 在验证集上评估
        val_loss, val_acc = validate_model(model, validation_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    return history, model


def validate_model(model, validation_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    val_running_loss = 0.0
    val_correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)  # 注意乘以batch size
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(validation_loader.dataset)
    val_acc = val_correct_predictions / len(validation_loader.dataset)
    return val_loss, val_acc


def cm_model(model, test_loader, version):
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            # Append current batch predictions and labels for confusion matrix
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if version == 0:
        plt.title('Confusion Matrix for Baseline Model')
        plt.savefig('Confusion Matrix for Baseline Model.png')
    else:
        plt.title('Confusion Matrix for Variant Model {}'.format(version))
        plt.savefig('Confusion Matrix for Variant Model {}.png'.format(version))
    plt.show()