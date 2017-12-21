from AbstractModels import Model, RecurrentModel
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, SimpleRNN, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D
import numpy as np

from keras.callbacks import EarlyStopping

PATIENCE = 20

class KerasMLPModel(Model):

    def create_model(self, network_shape=None, optimizer='adam', loss='mean_squared_error', dropout=0.5, activation='relu', l=0.01, kernel_init='normal'):
        self._model = Sequential()
        input_size = self._x_train.shape[1]
        output_size = self._y_train[1] if isinstance(self._y_train[1], list) else 1
        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None
        print('input_size: {0} output_size: {1} network_shape: {2}'.format(input_size, output_size, network_shape))

        if network_shape and network_shape[0] > 0:
            for i, layer in enumerate(network_shape):
                if layer > 0:
                    if i == 0:
                        self._model.add(Dense(layer,
                                              input_dim=input_size,
                                              activation=activation,
                                              kernel_initializer=kernel_init,
                                              kernel_regularizer=regularizers.l2(l)
                                              ))
                    else:
                        self._model.add(Dense(layer,
                                              activation=activation,
                                              kernel_initializer=kernel_init,
                                              kernel_regularizer=regularizers.l2(l)
                                              ))
                        self._model.add(Dropout(dropout))

        self._model.add(Dense(output_size))

        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        self._model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=validation_data,
                        callbacks=[EarlyStopping(patience=PATIENCE)])

    def predict(self, x, batch_size=500):
        return self._model.predict(x, batch_size=batch_size)

    def __str__(self):
        return "KerasMLP"


class KerasSimpleRNNModel(RecurrentModel):

    def create_model(self, network_shape=None, optimizer='adam', loss='mean_squared_error', dropout=0.5, activation='relu', l=0.01, kernel_init='normal'):
        self._model = Sequential()
        input_dim = (self._x_train.shape[1], self._x_train.shape[2])
        output_size = self._y_train[1] if isinstance(self._y_train[1], list) else 1
        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None
        print('KerasSimpleRNN, input_size: {0} output_size: {1} network_shape: {2} steps_back:{3}'.format(input_dim, output_size, network_shape,
                                                                                           self._steps_back))

        if network_shape and network_shape[0] > 0:
            for i, layer in enumerate(network_shape):
                if layer > 0:
                    if i == 0:
                        self._model.add(SimpleRNN(layer,
                                                  input_shape=input_dim,
                                                  activation=activation,
                                                  kernel_initializer=kernel_init,
                                                  kernel_regularizer=regularizers.l2(l),
                                                  return_sequences=True))
                    else:
                        self._model.add(SimpleRNN(layer,
                                                  activation=activation,
                                                  kernel_initializer=kernel_init,
                                                  kernel_regularizer=regularizers.l2(l),
                                                  return_sequences=True))
                    self._model.add(Dropout(dropout))

        self._model.add(Flatten())
        self._model.add(Dense(output_size))
        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        self._model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=validation_data,
                        callbacks=[EarlyStopping(patience=PATIENCE)])

    def predict(self, x, batch_size=500):
        return self._model.predict(x, batch_size=batch_size)

    def __str__(self):
        return "KerasSimpleRNN"


class KerasLSTMModel(RecurrentModel):

    def create_model(self, network_shape=None, optimizer='adam', loss='mean_squared_error', dropout=0.5, activation='relu'):
        self._model = Sequential()
        input_dim = (self._x_train.shape[1], self._x_train.shape[2])
        output_size = self._y_train[1] if isinstance(self._y_train[1], list) else 1
        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None
        print('KerasLSTM, input_size: {0} output_size: {1} network_shape: {2} steps_back:'.format(input_dim, output_size, network_shape),
              self._steps_back)

        if network_shape and network_shape[0] > 0:
            for i, layer in enumerate(network_shape):
                if layer > 0:
                    if i == 0:
                        self._model.add(LSTM(layer, input_shape=input_dim, activation=activation, kernel_initializer='normal', return_sequences=True))
                    else:
                        self._model.add(LSTM(layer, activation=activation, kernel_initializer='normal', return_sequences=True))
                    self._model.add(Dropout(dropout))
        else:
            print('No network shape provided')
            print('Default network shape: (5,)')
            self._model.add(LSTM(5, input_shape=input_dim, activation=activation, return_sequences=True))

        self._model.add(Flatten())
        self._model.add(Dense(output_size))
        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        self._model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, x, batch_size=500):
        return self._model.predict(x, batch_size=batch_size)

    def __str__(self):
        return "KerasLSTM"

class KerasConvModel(Model):

    def __init__(self, input_data, output_data, last_train_index, last_validate_index):
        super().__init__(input_data, output_data, last_train_index, last_validate_index)
        self._reshape_data()

    def create_model(self, network_shape=None, optimizer='adam', loss='mean_squared_error', dropout=0.5, activation='relu'):
        self._model = Sequential()
        input_dim = (self._x_train.shape[1], self._x_train.shape[2])
        output_size = self._y_train[1] if isinstance(self._y_train[1], list) else 1
        network_shape = network_shape if network_shape is not None and isinstance(network_shape, tuple) else None
        print('KerasConvModel, input_dim: {0} output_size: {1} network_shape: {2}'.format(input_dim, output_size, network_shape))
        self._model.add(Conv1D(32, 2, activation='relu', input_shape=input_dim, kernel_initializer='normal'))
        self._model.add(Conv1D(64, 2, activation='relu', kernel_initializer='normal'))
        self._model.add(MaxPooling1D())
        self._model.add(Dropout(dropout))
        self._model.add(Conv1D(128, 2, activation='relu', kernel_initializer='normal'))
        self._model.add(Conv1D(128, 2, activation='relu', kernel_initializer='normal'))
        self._model.add(GlobalAveragePooling1D())
        self._model.add(Dropout(dropout))
        self._model.add(Dense(output_size))
        self._model.compile(optimizer=optimizer, loss=loss)
        print('Model created')

    def _reshape_data(self):
        self._x_train = np.reshape(self._x_train, (self._x_train.shape[0], self._x_train.shape[1], 1))
        self._x_validate = np.reshape(self._x_validate, (self._x_validate.shape[0], self._x_validate.shape[1], 1))
        self._x_test = np.reshape(self._x_test, (self._x_test.shape[0], self._x_test.shape[1], 1))


    def _fit(self, x_train, y_train, epochs, validation_data, batch_size=500):
        self._model.fit(x_train, y_train, epochs=epochs, validation_data=validation_data, batch_size=batch_size, verbose=2)

    def predict(self, x, batch_size=500):
        return self._model.predict(x, batch_size=batch_size)

    def __str__(self):
        return "KerasConv"


