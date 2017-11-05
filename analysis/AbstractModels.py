from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer=None, steps_back=1):
        self._model_tester = model_tester
        self._model = None
        self._last_train_idx = last_train_index
        self._last_validate_idx = last_validate_index
        self._input_scaler = None
        self._output_scaler = None
        self._model_standarizer = model_standarizer
        self._steps_back = steps_back
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

    def test_model(self):
        r2_validate = self._model_tester.test_model(self._model, self._x_validate, self._y_validate, self._y_train)
        r2_train = self._model_tester.test_model(self._model, self._x_train, self._y_train, self._y_train)
        r2_test = self._model_tester.test_model(self._model, self._x_test, self._y_test, self._y_train)

        return r2_test, r2_validate, r2_train

    def get_model(self):
        return self._model

    def train_model(self, epochs=500, batch_size=500):
        print('Train models, number of epochs: {0}'.format(epochs))
        self._fit(self._x_train, self._y_train, epochs, (self._x_validate, self._y_validate), batch_size)

    def save_model(self, path, model_name):
        self._model.save(path+"{0}.h5".format(model_name))
        print('{0} saved'.format(model_name))

    @abstractmethod
    def create_model(self, network_shape, optimizer='adam', loss='mean_squared_error', dropout=0.5, activation='relu', l=0.01, kernel_init='normal'):
        pass

    @abstractmethod
    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        pass

    @abstractmethod
    def predict(self, x, batch_size=500):
        pass


class RecurrentModel(Model):

    def __init__(self, input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer=None, steps_back=1):
        super().__init__(input_data, output_data, last_train_index, last_validate_index, model_tester, model_standarizer, steps_back)
        self._transform_data()
        self._reshape_data()

    def _shift_data(self, data):
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(self._steps_back, -1, -1):
            cols.append(df.shift(i))

        agg = pd.concat(cols, axis=1)[self._steps_back:]
        return agg.values

    def _transform_data(self):
        self._x_train = self._shift_data(self._x_train)
        self._x_validate = self._shift_data(self._x_validate)
        self._x_test = self._shift_data(self._x_test)

    def _reshape_data(self):
        shape_1 = self._steps_back + 1
        shape_2 = self._x_train.shape[1] // (self._steps_back + 1)
        self._x_train = self._x_train.reshape((self._x_train.shape[0], shape_1, shape_2))
        self._x_validate = self._x_validate.reshape((self._x_validate.shape[0], shape_1, shape_2))
        self._x_test = self._x_test.reshape((self._x_test.shape[0], shape_1, shape_2))

        self._y_train = self._y_train[self._steps_back:]
        self._y_validate = self._y_validate[self._steps_back:]
        self._y_test = self._y_test[self._steps_back:]


    @abstractmethod
    def create_model(self, network_shape, optimizer='adam', loss='mean_squared_error', activation='relu', l=0.01, kernel_init='normal'):
        pass

    @abstractmethod
    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        pass

    @abstractmethod
    def predict(self, x, batch_size=500):
        pass
