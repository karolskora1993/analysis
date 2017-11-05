import pickle
import os
from keras.models import load_model
import pandas as pd
from .params import IN_DATA_LENGTH

def prepare_data(X):
    df = pd.DataFrame(X)
    block = _get_block(df.shape[1])
    if block:
        return _norm_data(X, block)
    else:
        print('Original data returned')
        return X


def predict(X, model_path):
    model_name, ext = os.path.splitext(model_path)
    for_reg = False
    if ext.endswith('p'):
        for_reg = True
        predictors = _select_predictors(X)
        X = _extend_data(X)
    model = _load_serialized_model(model_path, for_reg=for_reg)


def _select_predictors(X, var_out):
    pass


def _extend_data(X):
    pass


def _get_block(in_data_length):
    try:
        return IN_DATA_LENGTH[in_data_length]
    except KeyError:
        print('{0 }key not in blocks dictionary'.format(in_data_length))
        return None


def _load_scaler(block, out=False):
    in_out = 'out' if out else 'in'
    scaler_name = '{0}_{1}_scaller.p'.format(block, in_out)
    path = os.path.dirname(os.path.abspath(__file__)) + '/scalers/{0}'.format(scaler_name)
    return pickle.load(open(path, 'rb'))


def _norm_data(X, block):
    scaler = _load_scaler(block)
    return scaler.transform(X)


def _denorm_data(y, block):
    scaler = _load_scaler(block, out=True)
    return scaler.transform(y)


def _load_serialized_model(path, for_reg):
    if for_reg:
        return pickle.load(open(path, 'rb'))
    else:
        return load_model(path)





