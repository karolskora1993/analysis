import pandas as pd
import os
from sklearn import linear_model
import collections

HOME_PATH = str(os.path.expanduser('~') + '/')
VARS_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione.xlsx'
DATA_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_v4/'
SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/delays/'
BLOCK_NAMES = [
    'blok I',
    'blok II',
    'blok III',
    'blok IV'
]

COLUMN_NAMES = ['in', 'control', 'out']
SAVE_COLUMN_NAMES = ['var_out', 'conf_interval', 'close_delays', 'delay']
MAX_DELAY = 80


def find_best(models):
    sorted_models = sorted(models, key=lambda x: x['score'])
    return sorted_models[-1], '{0}, {1}'.format(sorted_models[-2]['delay'], sorted_models[-3]['delay'])


def load_block_vars():
    df = pd.read_excel(VARS_PATH, sheetname=None)
    print('blocks vars loaded')
    return df


def load_data(block_name):
    df = pd.read_csv(DATA_PATH + block_name + '.csv', index_col=0)
    print('data loaded')
    return df


def find_most_common(delays):
    return collections.Counter(delays).most_common()[0][0]


def create_model(x, y, delay):
    x_delayed = (x[MAX_DELAY - delay:-delay - 1]).reshape(-1, 1)
    model = linear_model.LinearRegression()
    model.fit(x_delayed, y)

    return model

def get_score(model, delay, x, y):
    x_delayed = (x[MAX_DELAY - delay:-delay - 1]).reshape(-1, 1)
    return model.score(x_delayed, y)



def find_delays(block_name, v):
    print(block_name)
    data = load_data(block_name)
    vars_in = v['in'].append(v['control']).dropna().tolist()
    vars_out = v['out'].dropna().tolist()
    df = pd.DataFrame(columns=SAVE_COLUMN_NAMES)
    for var_out in vars_out:
        y = data[var_out][MAX_DELAY:-1].as_matrix()
        models = []
        for delay in range(0, MAX_DELAY + 1):
            model_score = 0
            for var_in in vars_in:
                x = (data[var_in]).as_matrix()
                model = create_model(x, y, delay)
                model_score += get_score(model, delay, x, y)
            models.append({'delay': delay, 'score': model_score})
        best_model, close_delays = find_best(models)
        conf_interval = get_conf_interval(models, best_model)
        data_to_append = [[var_out, conf_interval, close_delays, best_model['delay']]]
        df = df.append(pd.DataFrame(data_to_append, columns=SAVE_COLUMN_NAMES))

    return df

def get_conf_interval(models, best_model, level=0.02):
    best_delay = best_model['delay']
    start = best_delay
    end = best_delay
    end_finished = False
    start_finished = False
    for i in range(0, len(models)):
        if not end_finished:
            if best_delay + i < len(models):
                if models[best_delay + i]['score']/best_model['score'] >= (1-level):
                    end = best_delay + i
                else:
                    end_finished = True
        if not start_finished:
            if best_delay - i >= 0:
                if models[best_delay - i]['score']/best_model['score'] >= (1-level):
                    start = best_delay - i
                else:
                    start_finished = True

    return '{0}-{1}'.format(start, end)

def save_data(block_name, data):
    data.to_csv(SAVE_PATH + 'delays_{0}.csv'.format(block_name))

def main():
    vars_in_blocks = load_block_vars()
    for block_name in BLOCK_NAMES:
        block_df = find_delays(block_name, vars_in_blocks[block_name][COLUMN_NAMES])
        save_data(block_name, block_df)

if __name__ == '__main__':
    main()
