import pickle
import os
from keras.models import load_model
import pandas as pd
from .params import IN_DATA_LENGTH, PREDICTORS
from keras import backend as B
from time import time


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
            data = self._extend_data(df)
            predictors = self._select_predictors(model_name)
            data = data[predictors].as_matrix()
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

    def _extend_data(self, X):
        temp = X.copy()

        diffs = [1, 2, 3, 4, 5]
        rolls = [5, 10, 15]

        start = time()
        diff_cols = {'{0}_df{1}'.format(colname, diffs[i]): temp[colname].diff(diffs[i]) for i in range(5) for colname in temp.columns}
        diff_df = pd.DataFrame.from_dict(diff_cols)
        X = pd.concat([X, diff_df], axis=1)
        print('diff:: {0}'.format(time() - start))


        start = time()
        means = {'{0}_r{1}_mean'.format(colname, rolls[i]): temp[colname].rolling(rolls[i], 1).mean() for i in range(3) for colname in temp.columns}
        mins = {'{0}_r{1}_min'.format(colname, rolls[i]): temp[colname].rolling(rolls[i], 1).min() for i in range(3) for colname in temp.columns}
        maxes = {'{0}_r{1}_max'.format(colname, rolls[i]): temp[colname].rolling(rolls[i], 1).max() for i in range(3) for colname in temp.columns}
        roll_dict = dict(means, **mins)
        roll_dict.update(maxes)
        rolls_df = pd.DataFrame.from_dict(roll_dict)
        X = pd.concat([X, rolls_df], axis=1)
        print('rolls:: {0}'.format(time() - start))
        X = X.iloc[15:]
        return X
        
        #
        # if i < 3:
        #     start = time()
        #     mean_col = temp[colname].rolling(rolls[i], 1).mean()
        #     mean_col.name = '{0}_r{1}_mean'.format(colname, rolls[i])
        #     X = pd.concat([X, mean_col], axis=1)
        #
        #     min_col = temp[colname].rolling(rolls[i], 1).min()
        #     min_col.name = '{0}_r{1}_min'.format(colname, rolls[i])
        #     X = pd.concat([X, min_col], axis=1)
        #
        #     max_col = temp[colname].rolling(rolls[i], 1).max()
        #     max_col.name = '{0}_r{1}_max'.format(colname, rolls[i])
        #     X = pd.concat([X, max_col], axis=1)
        #     print('rolls:: {0}'.format(time() - start))

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
