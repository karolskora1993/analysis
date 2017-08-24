import pandas as pd
from analysis import blocks
from numpy import nan
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import sklearn.metrics as metrics
from math import floor

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import sys
sys.setrecursionlimit(10000)

COL_NAMES = ['in', 'control', 'out']

def simple_model(x, y, x_test, y_test, y_labels, score_file):
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
    score(y_test, predictions, y_labels, score_file)
    return mlp


def score(predictions, y_test, y_labels, score_file):
    columns = len(y_test[0, :])
    with open(score_file, 'w') as f:
        for i in range(0, columns):
            r2 = metrics.r2_score(y_test[:, i], predictions[:, i])
            mse = metrics.mean_squared_error(y_test[:, i], predictions[:, i])
            mae = metrics.mean_absolute_error(y_test[:, i], predictions[:, i])
            ews = metrics.explained_variance_score(y_test[:, i], predictions[:, i])
            print_line = 'zmienna {0}- r^2: {1}  mse: {2} mae: {3} ews: {4} \n'.format(y_labels[i], r2, mse, mae, ews)
            print(print_line)
            f.write(print_line)


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv', index_col=0)
    print('data loaded')
    return df


def lasagne_model(x, y, x_test, y_test, y_labels, score_file):
    x = np.transpose(np.matrix(x))
    y = np.transpose(np.matrix(y))
    net = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('dense', layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, None, x.shape[0], x.shape[1]),
        # layer conv2d1
        conv2d1_incoming=(None, None, x.shape[0], x.shape[1]),
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.linear,
        conv2d1_W=lasagne.init.GlorotUniform(),
        # dropout1
        dropout1_p=0.5,
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.linear,
        # dropout2
        dropout2_p=0.5,
        # output
        output_nonlinearity=lasagne.nonlinearities.linear,
        output_num_units=len(y_labels),
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=200,
        verbose=1,
    )
    print('create net')
    nn = net.fit(x, y)
    print('fit net')
    predictions = net.predict(x_test)
    score(y_test, predictions, y_labels, score_file)

    return nn


def keras_model(x, y, x_test, y_test, y_labels, score_file):
    print('Keras model')
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

    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(y_labels), activation='linear'))

    model.compile(optimizer='adam',
                  loss='mse')

    model.fit(x, y, epochs=500, batch_size=32, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32)

    predictions = model.predict(x_test)
    score(predictions, y_test, y_labels,score_file)

    return model


def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


def main():
    blocks_vars = load_block_vars()
    for block_name in blocks.names:
        print(block_name)
        df = load_data(block_name)

        vars = blocks_vars[block_name][COL_NAMES]
        vars_in = [x for x in vars['in'] if not x is nan and not '#' in x]
        vars_control = [x for x in vars['control'] if not x is nan and not '#' in x]
        vars_all_in = vars_in + vars_control
        vars_out = [x for x in vars['out'] if not x is nan and not '#' in x]
        input_data = [np.array(df[x].tolist()) for x in vars_all_in]
        output_data = [np.array(df[x].tolist()) for x in vars_out]
        last_train_idx = floor(len(input_data[0]) * 0.875)
        last_test_idx = len(input_data[0]) - 1
        x_train = [x[:last_train_idx] for x in input_data]
        y_train = [x[:last_train_idx] for x in output_data]
        x_test = [x[last_train_idx:last_test_idx] for x in input_data]
        y_test = [x[last_train_idx:last_test_idx] for x in output_data]

        #sklearn model
        # name = 'sklearn_model_default_6x100_{0}'.format(block_name)
        # print('sklearn model {0}'.format(name))
        # score_file = blocks.MODEL_SAVE_PATH + '/scores/' + name + '.txt'
        # model = simple_model(x_train, y_train, x_test, y_test, vars_out, score_file)
        # save_file = blocks.MODEL_SAVE_PATH + name + '.p'
        # save_model(model, save_file)

        #lasagne model
        # name = 'lasagne_model_1_{0}'.format(block_name)
        # print('lasagne model {0}'.format(name))
        # score_file = blocks.MODEL_SAVE_PATH + '/scores/' + name + '.txt'
        # model = lasagne_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), vars_out, score_file)
        # save_file = blocks.MODEL_SAVE_PATH + name + '.p'
        # save_model(model, save_file)

        #keras model
        name = 'keras_model_test_{0}'.format(block_name)
        print('keras model {0}'.format(name))
        score_file = blocks.MODEL_SAVE_PATH + '/scores/' + name + '.txt'
        model = keras_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), vars_out, score_file)
        save_file = blocks.MODEL_SAVE_PATH + name + '.p'
        save_model(model, save_file)


if __name__ == '__main__':
    main()
