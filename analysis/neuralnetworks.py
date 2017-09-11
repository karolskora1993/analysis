import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from abc import ABC, abstractmethod
import sys
import os

sys.setrecursionlimit(10000)

COL_NAMES = ['in', 'control', 'out', 'delay']
LAST_TRAIN_IDX = 205038
LAST_VALIDATE_IDX = 257133


HOME_PATH = str(os.path.expanduser('~')+'/')

LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_v4/'
MODEL_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/models/'

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


class SimpleTester(ModelTester):
    def test_model(self, model, x_test, y_test):
        predictions = model.predict(x_test)
        return metrics.r2_score(y_test, predictions)


class Model(ABC):
    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester):
        self._input_data = input_data
        self._output_data = output_data
        self._model_tester = model_tester
        self._model = None
        self._last_train_idx = last_train_index
        self._last_validate_idx = last_validate_index
        self.validate_data()

    def validate_data(self):
        if np.any(np.isnan(self._input_data)) or np.any(np.isnan(self._output_data)):
            raise ValueError('Data contains NaN values')
        if not np.all(np.isfinite(self._input_data)) or not np.all(np.isfinite(self._output_data)):
            raise ValueError('Data contains infinite values')

    def test_model(self):
        r2_test = self._model_tester.test_model(self._model, self._input_data[self._last_validate_idx: ], self._output_data[self._last_validate_idx:])
        r2_train = self._model_tester.test_model(self._model, self._input_data[:self._last_train_idx], self._output_data[:self._last_train_idx])
        return (r2_test, r2_train)

    def get_model(self):
        return self._model

    @abstractmethod
    def create_model(self, input_size, output_size, network_shape, optimizer='adam', loss='mean_squared_error'):
        pass

    @abstractmethod
    def train_model(self, epochs=500):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class KerasMLPModel(Model):
    def create_model(self, input_size, output_size, network_shape=(10,), optimizer='adam', loss='mean_squared_error'):
        print('input_size: {0} output_size: {1} network_shape: {2}'.format(input_size, output_size, network_shape))
        self._model = Sequential()

        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else (10,)

        self._model.add(Dense(network_shape[0], input_dim=input_size))
        self._model.add(Dropout(0.5))

        for layer in network_shape:
            if layer > 0:
                self._model.add(Dense(layer, activation='relu'))
                self._model.add(Dropout(0.5))

        self._model.add(Dense(output_size, activation='linear'))
        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def train_model(self, epochs=500):
        x_train = self._input_data[:self._last_train_idx]
        y_train = self._output_data[:self._last_train_idx]
        validation_data = (self._input_data[self._last_train_idx: self._last_validate_idx],
                            self._output_data[self._last_train_idx: self._last_validate_idx])

        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        self._model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2, validation_data=validation_data)

    def predict(self, x):
        return self._model.predict(x)


def load_data(block_name):
    df = pd.read_csv(LOAD_PATH + block_name + '.csv', index_col=0)
    print('data loaded')
    return df



def load_block_vars():
    df = pd.read_excel(BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def save_model(model, save_path):
    pickle.dump(model, open(save_path, 'wb'))
    print('Model saved')


def get_network_shape():
    network_shape = (10, 0)
    if len(sys.argv) > 1:
        cmd_line_args = []
        for i, arg in enumerate(sys.argv):
            if i > 0:
                cmd_line_args.append(int(arg))
        network_shape = tuple(cmd_line_args)
    return network_shape

def shift_Data(input_data, output_data, delay):
    input_data = input_data[: -delay]
    output_data = output_data[delay:]
    return (input_data, output_data)

def model_block(data, var_names):
    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    delays = var_names['delay']
    block_models = []
    network_shape = get_network_shape()
    input_data = data[vars_in].as_matrix()
    for i, var_out in enumerate(vars_out):
        print('var_out:\t{0}'.format(var_out))
        output_data = data[var_out].as_matrix()
        delay = int(delays[i]) if delays[i] >= 1 else 0
        input_data, output_data = shift_Data(input_data, output_data, delay)
        model = KerasMLPModel(input_data, output_data, LAST_TRAIN_IDX, LAST_VALIDATE_IDX, SimpleTester())
        model.create_model(input_data.shape[1], 1, network_shape)
        model.train_model()
        r2 = model.test_model()
        block_models.append({'output': var_out, 'model': model.get_model()})
        print('r^2_test:\t {0} \n r^2_train:\t{1}'.format(r2[0], r2[1]))
        save_path = MODEL_SAVE_PATH + var_out + '.p'
        save_model(model, save_path)

    return block_models


def main():
    block_vars = load_block_vars()
    for block_name in BLOCK_NAMES:
        print(block_name)
        data = load_data(block_name)
        var_names = block_vars[block_name][COL_NAMES]
        #keras MLP model
        block_models = model_block(data, var_names, )


if __name__ == '__main__':
    main()
