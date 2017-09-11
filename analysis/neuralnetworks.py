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

COL_NAMES = ['in', 'control', 'out']
LAST_TRAIN_IDX = 205.038
LAST_VALIDATE_IDX = 257.133


HOME_PATH = str(os.path.expanduser('~')+'/')

LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_v4/'
MODEL_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/models/'

BLOCK_VARS_PATH_XLSX = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione.xlsx'

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
    def __init__(self, input_data, output_data, model_tester):
        self._input_data = input_data
        self._output_data = output_data
        self._model_tester = model_tester
        self._model = None
        self.validate_data()

    def validate_data(self):
        if np.any(np.isnan(self.input_data)) or np.any(np.isnan(self._output_data)):
            raise ValueError('Data contains NaN values')
        if not np.all(np.isfinite(self.input_data)) or not np.all(np.isfinite(self._output_data)):
            raise ValueError('Data contains infinite values')

    def test_model(self, x_test, y_test):
        return self._model_tester.test(self._model, x_test, y_test)

    def get_model(self):
        return self._model

    @abstractmethod
    def create_model(self, input_size, output_size, network_shape, optimizer='adam', loss='mean_squared_error'):
        pass

    @abstractmethod
    def train_model(self, x_train, y_train, validation_data=None, epochs=500):
        pass

    @abstractmethod
    def predict(self, x):
        pass



class KerasMLPModel(Model):
    def create_model(self, input_size, output_size, network_shape=(10,), optimizer='adam', loss='mean_squared_error'):
        self._model = Sequential()

        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else (10,)

        self._model.add(Dense(network_shape[0], input_dim=input_size))
        self._model.add(Dropout(0.5))

        for layer in network_shape:
            self._model.add(Dense(layer, activation='relu'))
            self._model.add(Dropout(0.5))

        self._model.add(Dense(output_size, activation='linear'))
        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def train_model(self, x_train, y_train, validation_data, epochs=500):
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
        for i, arg in sys.argv.enumerate():
            if i > 0:
                cmd_line_args.append(int(arg))
        network_shape = tuple(cmd_line_args)
    print('Network shape: {0}'.format(network_shape))
    return network_shape



def model_block(data, var_names):
    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    block_models = []
    network_shape = get_network_shape()
    input_data = data[vars_in].as_matrix().transpose()
    for var_out in vars_out:
        print('var_out:\t{0}'.format(var_out))
        output_data = data[var_out].as_matrix().transpose()
        model = KerasMLPModel(input_data, output_data, SimpleTester())
        model.create_model(input_data.shape[1], output_data, network_shape)
        model.train_model(input_data[:LAST_TRAIN_IDX], output_data[LAST_TRAIN_IDX],
                          (input_data[LAST_TRAIN_IDX: LAST_VALIDATE_IDX], output_data[LAST_TRAIN_IDX: LAST_VALIDATE_IDX]))
        r2 = model.test_model(input_data[LAST_VALIDATE_IDX:], output_data[LAST_VALIDATE_IDX:])
        block_models.append({'output': var_out, 'model': model.get_model()})
        print('r^2:\t {0}'.format(r2))
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
