import pandas as pd
from analysis import blocks
from numpy import nan
from keras.models import Sequential
from keras.layers import Dense

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
    # model.add(Dense(input_size, input_dim=input_size, activation='relu'))
    # model.add(Dense(input_size, activation='relu'))
    # model.add(Dense(output_size, activation='sigmoid'))
    return model


def compile_model(model):
    model.compile(optimizer='rmsprop', loss='mse')


def train_model(model, x, y):
    model.fit(x, y, epochs=10, batch_size=32)


def evaluate_model(model, x , y):
    scores = model.evaluate(x, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def main():
    blocks_vars = load_block_vars()
    for block_name in blocks.names:
        df = load_data(block_name)
        vars = blocks_vars[block_name][COL_NAMES]
        vars_in = [x for x in vars['in'] if not x is nan and not '#' in x]
        vars_control = [x for x in vars['control'] if not x is nan and not '#' in x]
        vars_out = [x for x in vars['out'] if not x is nan and not '#' in x]
        input_data = [df[x] for x in vars_in]
        control_data = [df[x] for x in vars_control]
        output_data = [df[x] for x in vars_out]
        model = define_model(len(input_data), len(output_data))
        compile_model(model)
        train_model(model)
        evaluate_model(model, input_data, output_data)


if __name__ == '__main__':
    main()
