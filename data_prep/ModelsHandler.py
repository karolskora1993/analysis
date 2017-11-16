import pickle
import os
from keras.models import load_model
import pandas as pd
from .params import IN_DATA_LENGTH, PREDICTORS


class ModelsHandler:

    def __init__(self):
        self._block = None
        self._in_scaler = None
        self._out_scalers = {}
        self._models = {}

    def prepare_data(self, data):
        df = pd.DataFrame(data)
        if not self._block:
            self._block = self._get_block(df.shape[1])
            self._in_scaler = self._load_in_scaler()
            print("input_scaler loaded")
        if self._block:
            return self._norm_data(data)
        else:
            print('Original data returned')
            return data

    def predict(self, X, vars, model_path):
        dict = {}
        for i, var in enumerate(vars):
            dict[var] = pd.Series(X[:, i])
        df = pd.DataFrame(dict)
        path, ext = os.path.splitext(model_path)
        model_name = path.split('/')[-1]
        for_reg = False
        if ext.endswith('p'):
            for_reg = True
            X = self._extend_data(df)
            predictors = self._select_predictors(model_name)
            X = X[predictors].as_matrix()

        model = self._models.get(model_name, None)
        if not model:
            model = self._load_serialized_model(model_path, for_reg=for_reg)
            self._models[model_name] = model
            print("new model loaded")
        y_pred = model.predict(X)
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

        for df in diffs:
            for colname in temp.columns:
                z = temp[colname].copy()
                z.name = '{0}_df{1}'.format(z.name, df)
                z = z.diff(df)
                X = pd.concat([X, z], axis=1)

        for roll in rolls:
            for colname in temp.columns:
                mean_col = temp[colname].rolling(roll, 1).mean()
                mean_col.name = '{0}_r{1}_mean'.format(colname, roll)
                X = pd.concat([X, mean_col], axis=1)

                min_col = temp[colname].rolling(roll, 1).min()
                min_col.name = '{0}_r{1}_min'.format(colname, roll)
                X = pd.concat([X, min_col], axis=1)

                max_col = temp[colname].rolling(roll, 1).max()
                max_col.name = '{0}_r{1}_max'.format(colname, roll)
                X = pd.concat([X, max_col], axis=1)
        X = X.iloc[15:]
        X.reset_index(inplace=True)
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










