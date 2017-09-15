import pandas as pd
import numpy as np
import pickle
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.setrecursionlimit(10000)
COL_NAMES = ['in', 'control', 'out', 'delay']
LAST_TRAIN_IDX = 205038
LAST_VALIDATE_IDX = 257133
BATCH_SIZE = 128
HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_v4/'
MODEL_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/models/'
SCORE_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/models/stats/'
BLOCK_VARS_PATH_XLSX = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione_v3.xlsx'
BLOCK_NAMES = [
    'blok I',
    # 'blok II',
    # 'blok III',
    # 'blok IV'
]


class ModelTester(ABC):
    @abstractmethod
    def test_model(self, model, x_test, y_test):
        pass


class DataStandarizer(ABC):
    def fit(self, x_train, y_train):
        self._input_scaler = StandardScaler()
        self._input_scaler.fit(x_train)
        self._output_scaler = StandardScaler()
        self._output_scaler.fit(y_train.reshape(-1,1))

    @abstractmethod
    def standarize_data(self, input_data, output_data):
        pass


class SimpleStandarizer(DataStandarizer):
    def standarize_data(self, input_data, output_data):
        if not isinstance(input_data, list) and not isinstance(output_data, list):
            raise ValueError('input_data and output_data are not a list')
        standarized_input_data = []
        standarized_output_data = []
        for single_input in input_data:
            standarized_input_data.append(self._input_scaler.transform(single_input))
        for single_output in output_data:
            standarized_output_data.append(self._output_scaler.transform(single_output.reshape(-1, 1)))

        return standarized_input_data, standarized_output_data


class SimpleTester(ModelTester):
    def test_model(self, model, x_test, y_test):
        predictions = model.predict(x_test)
        return metrics.r2_score(y_test, predictions)


class Model(ABC):
    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer=None):
        self._input_data = input_data
        self._output_data = output_data
        self._model_tester = model_tester
        self._model = None
        self._last_train_idx = last_train_index
        self._last_validate_idx = last_validate_index
        self._input_scaler = None
        self._output_scaler = None
        self._model_standarizer = model_standarizer
        self.validate_data()

    def validate_data(self):
        if np.any(np.isnan(self._input_data)) or np.any(np.isnan(self._output_data)):
            raise ValueError('Data contains NaN values')
        if not np.all(np.isfinite(self._input_data)) or not np.all(np.isfinite(self._output_data)):
            raise ValueError('Data contains infinite values')

    def test_model(self):
        x_test = self._input_data[self._last_validate_idx:]
        x_train = self._input_data[:self._last_train_idx]
        y_test = self._output_data[self._last_validate_idx:]
        y_train = self._output_data[:self._last_train_idx]

        if self._model_standarizer:
            input_data, output_data = self._model_standarizer.standarize_data([x_train, x_test], [y_train, y_test])
            x_train = input_data[0]
            y_train = output_data[0]
            x_test = input_data[1]
            y_test = output_data[1]

        r2_test = self._model_tester.test_model(self._model, x_test, y_test)
        r2_train = self._model_tester.test_model(self._model, x_train, y_train)

        return r2_test, r2_train

    def get_model(self):
        return self._model

    @abstractmethod
    def create_model(self, input_size, output_size, network_shape, optimizer='adam', loss='mean_squared_error'):
        pass

    @abstractmethod
    def train_model(self, epochs=500, standarize=True):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class KerasMLPModel(Model):

    def create_model(self, input_size, output_size, network_shape=None, optimizer='adam', loss='mean_squared_error'):
        print('input_size: {0} output_size: {1} network_shape: {2}'.format(input_size, output_size, network_shape))
        self._model = Sequential()

        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None

        if network_shape and network_shape[0] > 0:
            for i, layer in enumerate(network_shape):
                if layer > 0:
                    if i == 0:
                        self._model.add(Dense(layer, input_dim=input_size, activation='relu'))
                    else:
                        self._model.add(Dense(layer, activation='relu'))
                    self._model.add(Dropout(0.2))
        else:
            print('No network shape provided')
            print('Default network shape: (5,)')
            self._model.add(Dense(5, input_dim=input_size, activation='linear'))

        self._model.add(Dense(output_size))

        self._model.compile(optimizer=optimizer, loss=loss, metrics=[coeff_determination])
        print('Model created')

    def train_model(self, epochs=500):
        print('Train models, number of epochs: {0}'.format(epochs))
        x_train = self._input_data[:self._last_train_idx]
        y_train = self._output_data[:self._last_train_idx]
        x_validate = self._input_data[self._last_train_idx: self._last_validate_idx]
        y_validate = self._output_data[self._last_train_idx: self._last_validate_idx]

        if self._model_standarizer:
            self._model_standarizer.fit(x_train, y_train)
            input_data, output_data = self._model_standarizer.standarize_data([x_train, x_validate], [y_train, y_validate])
            x_train = input_data[0]
            y_train = output_data[0]
            x_validate = input_data[1]
            y_validate = output_data[1]

        self._model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=2, validation_data=(x_validate, y_validate))

    def predict(self, x):
        return self._model.predict(x, batch_size= BATCH_SIZE)


class DataLoader(object):
    @staticmethod
    def load_data(block_name):
        df = pd.read_csv(LOAD_PATH + block_name + '.csv', index_col=0)
        print('data loaded')
        return df

    @staticmethod
    def load_block_vars():
        df = pd.read_excel(BLOCK_VARS_PATH_XLSX, sheetname=None)
        print('blocks vars loaded')
        return df

    @staticmethod
    def save_model(model, save_path):
        pickle.dump(model, open(save_path, 'wb'))
        print('Model saved')


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res) / (SS_tot + K.epsilon())


def get_network_shape():
    network_shape = None
    if len(sys.argv) > 1:
        cmd_line_args = []
        for i, arg in enumerate(sys.argv):
            if i > 0:
                cmd_line_args.append(int(arg))
        network_shape = tuple(cmd_line_args)
    return (network_shape[:-1], network_shape[-1]) if network_shape else (None, 500)


def shift_data(input_data, output_data, delay):
    input_data = input_data[: -delay]
    output_data = output_data[delay:]
    return input_data, output_data


def model_block(data, var_names):
    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    delays = var_names['delay']
    block_models = []
    network_shape, epochs = get_network_shape()
    input_data = data[vars_in].as_matrix()
    for i, var_out in enumerate(vars_out):
        print('var_out:\t{0}'.format(var_out))
        output_data = data[var_out].as_matrix()
        delay = int(delays[i]) if delays[i] >= 1 else 0
        if delay > 0:
            x, y = shift_data(input_data, output_data, delay)
        model = KerasMLPModel(x, y, LAST_TRAIN_IDX, LAST_VALIDATE_IDX, SimpleTester(), SimpleStandarizer())
        model.create_model(input_data.shape[1], 1, network_shape)
        model.train_model(epochs)
        r2 = model.test_model()
        block_models.append({'output': var_out, 'model': model.get_model()})
        print_line = 'zmienna:\t{0} r^2_test:\t {1} \t r^2_train:\t{2} \n'.format(var_out, r2[0], r2[1])
        print(print_line)
        save_path = MODEL_SAVE_PATH + var_out + '.p'
        DataLoader.save_model(model, save_path)
        with open(SCORE_SAVE_PATH + 'score_{network_shape}_{epochs}epochs_standarizedY.txt'.format(var_out=var_out, network_shape=network_shape, epochs=epochs), 'a') as file:
            file.write(print_line)

    return block_models


def main():
    block_vars = DataLoader.load_block_vars()
    for block_name in BLOCK_NAMES:
        print(block_name)
        data = DataLoader.load_data(block_name)
        var_names = block_vars[block_name][COL_NAMES]
        #keras MLP model
        block_models = model_block(data, var_names)


if __name__ == '__main__':
    main()
