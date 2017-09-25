import pandas as pd
import numpy as np
import pickle
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.optimizers import adam
from keras.layers import Dense, SimpleRNN, Dropout, Flatten
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.setrecursionlimit(10000)
COL_NAMES = ['in', 'control', 'out', 'delay']
LAST_TRAIN_IDX = 205038
LAST_VALIDATE_IDX = 257133
BATCH_SIZE = 500
DROPOUT = 0.4
TIMESTEPS = None
OPTIMIZER = adam(lr=0.0001)
HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_v4/'
MODEL_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/models/'
SCORE_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/models/stats/'
BLOCK_VARS_PATH_XLSX = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione_v3.xlsx'
SAVE_FILE_NAME = 'score_{network_shape}_{epochs}epochs_v2_lr_mod.txt'
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
        self._output_scaler.fit(y_train.reshape(-1, 1))

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
        return metrics.r2_score(y_test, predictions.astype(np.float64))


class Model(ABC):
    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer=None):
        self._model_tester = model_tester
        self._model = None
        self._last_train_idx = last_train_index
        self._last_validate_idx = last_validate_index
        self._input_scaler = None
        self._output_scaler = None
        self._model_standarizer = model_standarizer
        self._validate_data(input_data, output_data)
        self._divide_data(input_data, output_data)
        if self._model_standarizer:
            self.standarize_data()

    def standarize_data(self):
        self._model_standarizer.fit(self._x_train, self._y_train)
        input_data, output_data = self._model_standarizer.standarize_data([self._x_train, self._x_validate, self._x_test],
                                                                          [self._y_train, self._y_validate, self._y_test])
        self._x_train = input_data[0]
        self._y_train = output_data[0]
        self._x_validate = input_data[1]
        self._y_validate = output_data[1]
        self._x_test = input_data[2]
        self._y_test = output_data[2]


    def _divide_data(self, input_data, output_data):
        self._x_test = input_data[self._last_validate_idx:]
        self._x_train = input_data[:self._last_train_idx]
        self._y_test = output_data[self._last_validate_idx:]
        self._y_train = output_data[:self._last_train_idx]
        self._x_validate = input_data[self._last_train_idx: self._last_validate_idx]
        self._y_validate = output_data[self._last_train_idx: self._last_validate_idx]


    def _validate_data(self, input_data, output_data):
        if np.any(np.isnan(input_data)) or np.any(np.isnan(output_data)):
            raise ValueError('Data contains NaN values')
        if not np.all(np.isfinite(input_data)) or not np.all(np.isfinite(output_data)):
            raise ValueError('Data contains infinite values')

    def test_model(self):
        r2_test = self._model_tester.test_model(self._model, self._x_test, self._y_test)
        r2_train = self._model_tester.test_model(self._model, self._x_train, self._y_train)

        return r2_test, r2_train

    def get_model(self):
        return self._model

    def train_model(self, epochs=500):
        print('Train models, number of epochs: {0}'.format(epochs))


        self._fit(self._x_train, self._y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(self._x_validate, self._y_validate))

    @abstractmethod
    def create_model(self, input_size, output_size, network_shape, optimizer='adam', loss='mean_squared_error'):
        pass

    @abstractmethod
    def _fit(self, x_train, y_train, epochs, batch_size, validation_data):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class RecurrentModel(ABC, Model):

    def _series_to_supervised(data, n_in=1, n_out=1):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        return agg

    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer=None, steps_back=1):
        Model.__init__(input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer)
        self._steps_back = steps_back
        self._transform_data()

    def _transform_data(self):
        self._x_train = self._series_to_supervised(self._x_train, self._steps_back, self._steps_back)
        self._y_train = self._series_to_supervised(self._y_train, self._steps_back, self._steps_back)
        self._x_validate = self._series_to_supervised(self._x_validate, self._steps_back, self._steps_back)
        self._y_validate = self._series_to_supervised(self._y_validate, self._steps_back, self._steps_back)
        self._x_test = self._series_to_supervised(self._x_test, self._steps_back, self._steps_back)
        self._y_test = self._series_to_supervised(self._y_test, self._steps_back, self._steps_back)




class KerasMLPModel(Model):

    def create_model(self, input_size, output_size, network_shape=None, optimizer='adam', loss='mean_squared_error'):
        print('input_size: {0} output_size: {1} network_shape: {2}'.format(input_size, output_size, network_shape))
        self._model = Sequential()

        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None

        if network_shape and network_shape[0] > 0:
            for i, layer in enumerate(network_shape):
                if layer > 0:
                    if i == 0:
                        self._model.add(Dense(layer, input_dim=input_size, activation='relu', kernel_initializer='normal'))
                    else:
                        self._model.add(Dense(layer, activation='relu', kernel_initializer='normal'))
                    self._model.add(Dropout(DROPOUT))
        else:
            print('No network shape provided')
            print('Default network shape: (5,)')
            self._model.add(Dense(5, input_dim=input_size, activation='relu'))

        self._model.add(Dense(output_size))

        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def _fit(self, x_train, y_train, epochs, batch_size, validation_data):
        self._model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=2,
                        validation_data=validation_data)

    def predict(self, x):
        return self._model.predict(x, batch_size=BATCH_SIZE)


class KerasSimpleRNNModel(RecurrentModel):

    def create_model(self, input_size, output_size, network_shape=None, optimizer='adam', loss='mean_squared_error'):
        print('KerasSimpleRNN, input_size: {0} output_size: {1} network_shape: {2}'.format(input_size, output_size, network_shape))
        self._model = Sequential()

        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None

        if network_shape and network_shape[0] > 0:
            for i, layer in enumerate(network_shape):
                if layer > 0:
                    if i == 0:
                        self._model.add(SimpleRNN(layer, input_shape =(TIMESTEPS, input_size), activation='relu', kernel_initializer='normal', return_sequences=True))
                    if i == len(network_shape) - 1:
                        self._model.add(SimpleRNN(layer, activation='relu', kernel_initializer='normal'))

                    else:
                        self._model.add(SimpleRNN(layer, activation='relu', kernel_initializer='normal', return_sequences=True))
                    self._model.add(Dropout(DROPOUT))
        else:
            print('No network shape provided')
            print('Default network shape: (5,)')
            self._model.add(SimpleRNN(5, input_shape=(TIMESTEPS, input_size), activation='relu'))

        self._model.add(Dense(output_size, kernel_initializer='normal'))

        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def _fit(self, x_train, y_train, epochs, batch_size, validation_data):
        # x_train = x_train.reshape(x_train.shape + (1,))
        self._model.fit(x_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, verbose=2)

    def predict(self, x):
        return self._model.predict(x, batch_size=BATCH_SIZE)


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


def get_network_shape():
    network_shape = None
    if len(sys.argv) > 1:
        cmd_line_args = []
        for i, arg in enumerate(sys.argv):
            if i > 0:
                cmd_line_args.append(int(arg))
        network_shape = tuple(cmd_line_args)
    return (network_shape[:-1], network_shape[-1]) if network_shape else (None, 10)


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
        else:
            x, y = input_data, output_data

        model = KerasMLPModel(x, y, LAST_TRAIN_IDX, LAST_VALIDATE_IDX, SimpleTester(), SimpleStandarizer())
        model.create_model(input_data.shape[1], 1, network_shape,  optimizer=OPTIMIZER)
        model.train_model(epochs)

        r2 = model.test_model()
        block_models.append({'output': var_out, 'model': model.get_model()})
        save_path = MODEL_SAVE_PATH + var_out + '.p'
        DataLoader.save_model(model, save_path)
        with open(SCORE_SAVE_PATH + SAVE_FILE_NAME.format(var_out=var_out, network_shape=network_shape, epochs=epochs), 'a') as file:
            file.write('zmienna:\t{0} r^2_test:\t {1} \t r^2_train:\t{2} \n'.format(var_out, r2[0], r2[1]))

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
