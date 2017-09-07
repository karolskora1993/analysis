import pandas as pd
from analysis import blocks
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout
from abc import ABC, abstractmethod
import sys
sys.setrecursionlimit(10000)

COL_NAMES = ['in', 'control', 'out']
LAST_TRAIN_IDX = 205.038
LAST_VALIDATE_IDX = 257.133


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
        self._model_tester.test(self._model, x_test, y_test)

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

    def train_model(self, x_train, y_train, validation_data, epochs=500):
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        self._model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2, validation_data=validation_data)

    def predict(self, x):
        return self._model.predict(x)


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv', index_col=0)
    print('data loaded')
    return df

def get_score(predictions, y_test, y_labels, score_file):
    columns = predictions.shape[1]
    with open(score_file, 'w') as f:
        for i in range(0, columns):
            r2 = metrics.r2_score(y_test[:, i], predictions[:, i])
            mse = metrics.mean_squared_error(y_test[:, i], predictions[:, i])
            mae = metrics.mean_absolute_error(y_test[:, i], predictions[:, i])
            ews = metrics.explained_variance_score(y_test[:, i], predictions[:, i])
            print_line = 'zmienna {0}- r^2: {1}  mse: {2} mae: {3} ews: {4} \n'.format(y_labels[i], r2, mse, mae, ews)
            print(print_line)
            f.write(print_line)


def sklearn_model(x, y, x_test, y_test, y_labels, score_file):
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 100, 100, 100, 100), max_iter=100000)
    print('Any NaN: {0}'.format(np.any(np.isnan(x))))
    print('All finite: {0}'.format(np.all(np.isfinite(x))))
    x = np.transpose(np.matrix(x))
    y = np.transpose(np.matrix(y))
    x_test = np.transpose(np.matrix(x_test))
    y_test = np.transpose(np.matrix(y_test))
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    print('scale')
    mlp.fit(x, y)
    print('fit')
    predictions = mlp.predict(x_test)
    print('predict')
    get_score(predictions, y_test, y_labels, score_file)
    return mlp




def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))

def model_single_output(input_data, output_data):
    model =



def model_block(data, var_names):
    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    network_shape = (10, 0)
    if len(sys.argv) > 1:
        cmd_line_args = []
        for i, arg in sys.argv.enumerate():
            if i > 0:
                cmd_line_args.append(int(arg))
        network_shape = tuple(cmd_line_args)
    block_models = []
    input_data = data[vars_in].as_matrix()
    for var_out in vars_out:
        output_data = data[var_out].as_matrix()
        model = KerasMLPModel(input_data, output_data, SimpleTester())
        model.create_model(input_data.shape[1], output_data, network_shape)
        model.train_model()
        model.test_model()
        block_models.append({'output': var_out, 'model':model.get_model()})

    return block_models



def main():
    block_vars = load_block_vars()
    for block_name in blocks.names:
        print(block_name)
        data = load_data(block_name)
        var_names = block_vars[block_name][COL_NAMES]
        #keras MLP model
        block_models = model_block(data, var_names, )


if __name__ == '__main__':
    main()
