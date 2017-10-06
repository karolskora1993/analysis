from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


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
        self._x_train = input_data[:self._last_train_idx]
        self._x_validate = input_data[self._last_train_idx: self._last_validate_idx]
        self._x_test = input_data[self._last_validate_idx:]

        self._y_train = output_data[:self._last_train_idx]
        self._y_validate = output_data[self._last_train_idx: self._last_validate_idx]
        self._y_test = output_data[self._last_validate_idx:]


    def _validate_data(self, input_data, output_data):
        if np.any(np.isnan(input_data)) or np.any(np.isnan(output_data)):
            raise ValueError('Data contains NaN values')
        if not np.all(np.isfinite(input_data)) or not np.all(np.isfinite(output_data)):
            raise ValueError('Data contains infinite values')

    def test_model(self):
        r2_test = self._model_tester.test_model(self._model, self._x_test, self._y_test, self._y_train)
        r2_validate = self._model_tester.test_model(self._model, self._x_validate, self._y_validate, self._y_train)
        r2_train = self._model_tester.test_model(self._model, self._x_train, self._y_train, self._y_train)

        return r2_test, r2_validate, r2_train

    def get_model(self):
        return self._model

    def train_model(self, epochs=500, batch_size=500):
        print('Train models, number of epochs: {0}'.format(epochs))
        self._fit(self._x_train, self._y_train, epochs=epochs, batch_size=batch_size, validation_data=(self._x_validate, self._y_validate))

    @abstractmethod
    def create_model(self, network_shape, optimizer='adam', loss='mean_squared_error', dropout=0.5):
        pass

    @abstractmethod
    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        pass

    @abstractmethod
    def predict(self, x, batch_size=500):
        pass


class RecurrentModel(Model):

    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer=None, steps_back=1):
        super().__init__(input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer)
        self._steps_back = steps_back
        self._transform_data()
        self._reshape_data()

    def _shift_data(self, data):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(self._steps_back, -1, -1):
            cols.append(df.shift(-i))

        agg = pd.concat(cols, axis=1)[:-self._steps_back]
        return agg.values

    def _transform_data(self):
        self._x_train = self._shift_data(self._x_train)
        # self._y_train = self._series_to_supervised(self._y_train)
        self._x_validate = self._shift_data(self._x_validate)
        # self._y_validate = self._series_to_supervised(self._y_validate)
        self._x_test = self._shift_data(self._x_test)
        # self._y_test = self._series_to_supervised(self._y_test)

    def _reshape_data(self):
        self._x_train = self._x_train.reshape((self._x_train.shape[0], self._steps_back, self._x_train.shape[1]//self._steps_back))
        self._x_validate = self._x_validate.reshape((self._x_validate.shape[0], self._steps_back, self._x_validate.shape[1]//self._steps_back))
        self._x_test = self._x_test.reshape((self._x_test.shape[0], self._steps_back, self._x_test.shape[1]//self._steps_back))

        self._y_train = self._y_train[1:]
        self._y_validate = self._y_validate[1:]
        self._y_test = self._y_test[1:]


    @abstractmethod
    def create_model(self, network_shape, optimizer='adam', loss='mean_squared_error'):
        pass

    @abstractmethod
    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        pass

    @abstractmethod
    def predict(self, x, batch_size=500):
        pass
