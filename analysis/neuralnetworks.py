import pandas as pd
from analysis import blocks
from numpy import nan
from numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from math import floor

COL_NAMES = ['in', 'control', 'out']


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv')
    print('data loaded')
    return  df


def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def define_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(output_size, input_shape=(input_size,)))
    return model


def compile_model(model):
    model.compile(loss=losses.mean_squared_error, optimizer='sgd')

def train_model(model, x, y, epochs, batch_size = 32):
    model.fit(x, y, epochs=epochs, batch_size=batch_size)


def evaluate_model(model, x , y):
    scores = model.evaluate(x, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def main():
    blocks_vars = load_block_vars()
    for block_name in blocks.names:
        df = load_data(block_name)
        vars = blocks_vars[block_name][COL_NAMES]
        vars_in = [x for x in vars['in'] if not x is nan and not '#' in x]
        vars_out = [x for x in vars['out'] if not x is nan and not '#' in x]
        input_data = [np.array(df[x].tolist()) for x in vars_in]
        output_data = [np.array(df[x].tolist()) for x in vars_out]
        last_train_idx = floor(len(input_data[0]) * 0.875)
        x_train = [x[:last_train_idx] for x in input_data]
        y_train = [x[:last_train_idx] for x in output_data]
        x_test = [x[last_train_idx:len(input_data[0]) - 1] for x in input_data]
        y_test = [x[last_train_idx:len(output_data[0]) - 1] for x in output_data]

        model = define_model(len(input_data), len(output_data))
        compile_model(model)
        train_model(model, x_train, y_train, 100)
        evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
