import torch
from create_baseline_model import *
from create_dataset import get_test_data_loader
from train_model import test_model


if __name__ == '__main__':
    # create the baseline model
    # create_baseline_model()
    # load the model
    model = torch.load('baseline_model.pth')
    # evaluate the model
    test_model(model, get_test_data_loader(), get_baseline_criterion())
