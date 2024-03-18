import torch


def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10):
    # 确保模型在训练模式下运行
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device_name), labels.to(device_name)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / len(train_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # 每个epoch结束后在验证集上评估模型
        validate_model(model, validation_loader, criterion)

    return model


def validate_model(model, validation_loader, criterion):
    # 确保模型在评估模式下运行
    model.eval()
    validation_loss = 0.0
    correct_predictions = 0
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for inputs, labels in validation_loader:

            inputs, labels = inputs.to(device_name), labels.to(device_name)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    validation_loss /= len(validation_loader.dataset)
    validation_acc = correct_predictions / len(validation_loader.dataset)
    print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_acc:.4f}\n')


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
