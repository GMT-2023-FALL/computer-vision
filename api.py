import os
import pickle

from create_dataset import get_test_data_loader, get_custom_data_loader
from train_model import validate_model, cm_model
from variant_model import create_variant_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from create_baseline_model import *

if __name__ == '__main__':
    # create the baseline model
    # create_baseline_model()
    # model = torch.load('baseline_model.pth')
    # cm_model(model, get_custom_data_loader(), version=-1)
    # test_model(model, get_custom_data_loader(), get_baseline_criterion())
    for i in range(1, 5):
        # create_variant_model(i)
        # load the model
        model = torch.load('variant_model{}.pth'.format(i))
        # evaluate the model
        test_model(model, get_test_data_loader(), get_baseline_criterion())
        # cm_model(model, get_test_data_loader(), i)
        # history = pickle.load(open(f'variant_model{i}_history.pkl', 'rb'))
        # print(history['train_acc'][-1], history['val_acc'][-1])
