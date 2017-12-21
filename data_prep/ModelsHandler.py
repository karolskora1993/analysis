import pickle
import os
from keras.models import load_model
import pandas as pd
from .params import IN_DATA_LENGTH, PREDICTORS
from keras import backend as B
from time import time
import re

class ModelsHandler:
    def __init__(self):
        self._block = None
        self._in_scaler = None
        self._out_scalers = {}
        self._models = {}

    def prepare_data(self, data):
        reshaped = False
        if len(data.shape) > 2:
            data = data.reshape(data.shape[1], data.shape[2])
            reshaped = True
        if not self._block:
            self._block = self._get_block(data.shape[1])
            self._in_scaler = self._load_in_scaler()
            print("input_scaler loaded")
        if self._block:
            normalized = self._norm_data(data)
            if reshaped:
                return normalized.reshape(1, normalized.shape[0], normalized.shape[1])
            else:
                return normalized
        else:
            print('Original data returned')
            return data

    def predict(self, data, vars, model_path):

        path, ext = os.path.splitext(model_path)
        model_name = path.split('/')[-1]
        for_reg = False

        if ext.endswith('p'):
            start = time()
            for_reg = True
            dict = {}
            for i, var in enumerate(vars):
                dict[var] = pd.Series(data[:, i])
            df = pd.DataFrame(dict)
            df, predictors = self._choose_data(df, model_name)
            df = self._extend_data(df, predictors)
            predictors = self._select_predictors(model_name)
            data = df[predictors].as_matrix()
            print('extend data time: {0}'.format(time() - start))

        model = self._models.get(model_name, None)
        if not model:
            start = time()
            model = self._load_serialized_model(model_path, for_reg=for_reg)
            self._models[model_name] = model
            print("new model loaded")

        y_pred = model.predict(data)

        if ext.endswith('p'):
            return y_pred
        else:
            return self._denorm_data(y_pred, model_name)

    def _choose_data(self, df, model_name):
        predictors = self._select_predictors(model_name)
        choosen_predictors = {}
        for predictor in predictors:
            diff = int(re.findall(r'\d+', predictor)[-1])
            name = '_'.join(filter(lambda y: not y.startswith(('r', 'd', 'max', 'mean', 'min')), predictor.split('_')))
            if name in choosen_predictors:
                choosen_predictors[name].append(diff)
            else:
                choosen_predictors[name] = [diff]

        pred_list = list(set(choosen_predictors.keys()))
        return df[pred_list], choosen_predictors


    def _norm_data(self, data):
        return self._in_scaler.transform(data)

    def _load_in_scaler(self):
        scaler_name = '{0}_in_scaller.p'.format(self._block)
        path = os.path.dirname(os.path.abspath(__file__)) + '/scalers/{0}'.format(scaler_name)
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _get_block(self, in_data_length):
        try:
            return IN_DATA_LENGTH[in_data_length]
        except KeyError:
            print('{0 }key not in blocks dictionary'.format(in_data_length))
            return None

    def _select_predictors(self, var_out):
        return PREDICTORS[var_out]

    def _extend_data(self, X, predictors):
        diffs = [1, 2, 3, 4, 5]
        rolls = [5, 10, 15]

        diff_cols = {'{0}_df{1}'.format(colname, i): X[colname].diff(i) for colname in X.columns for i in diffs if i in predictors[colname]}
        means = {'{0}_r{1}_mean'.format(colname, i): X[colname].rolling(i, 1).mean() for colname in X.columns for i in rolls if i in predictors[colname]}
        mins = {'{0}_r{1}_min'.format(colname, i): X[colname].rolling(i, 1).min() for colname in X.columns for i in rolls if i in predictors[colname]}
        maxes = {'{0}_r{1}_max'.format(colname, i): X[colname].rolling(i, 1).max() for colname in X.columns for i in rolls if i in predictors[colname]}
        roll_dict = dict(means, **mins)
        roll_dict.update(maxes)
        roll_dict.update(diff_cols)
        rolls_df = pd.DataFrame.from_dict(roll_dict)
        X = pd.concat([X, rolls_df], axis=1)
        X = X.iloc[15:]
        return X


    def _load_out_scaler(self, var_name):
        scaler_name = '{0}_out_{1}_scaller.p'.format(self._block, var_name)
        path = os.path.dirname(os.path.abspath(__file__)) + '/scalers/{0}'.format(scaler_name)
        with open(path, 'rb') as f:
            scaler = pickle.load(f)

        return scaler

    def _denorm_data(self, y, var_name):

        scaler = self._out_scalers.get(var_name, None)
        if not scaler:
            scaler = self._load_out_scaler(var_name)
            self._out_scalers[var_name] = scaler
            print("new output_scaler added")
        return scaler.transform(y)

    def _load_serialized_model(self, path, for_reg):
        if for_reg:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            return load_model(path)

    def clear_model(self, model_name):
        if model_name in self._models:
            B.clear_session()
