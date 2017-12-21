import pandas as pd
import pickle
import os

HOME_PATH = str(os.path.expanduser('~')+'/')
DATA_PATH = HOME_PATH + 'Dokumenty/analysis/data/all/all_v5.csv'
SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_v5/all_v5.csv'

IDX_PATH = HOME_PATH + 'Dokumenty/analysis/data/przestoje_polowa_v2.p'


def remove(dataframe, indexes, save_path=None):
    for i in indexes:
        first = i['first_idx']
        last = i['last_idx']
        to_drop = [i for i in range(first - 1, last)]
        dataframe.drop(to_drop, inplace=True)
    df = pd.DataFrame(dataframe)
    if save_path:
        df.to_csv(SAVE_PATH)


def _main():
    df = pd.read_csv(DATA_PATH, index_col=0)
    idx = pickle.load(open(IDX_PATH, 'rb'))
    remove(df, idx, SAVE_PATH)


if __name__ == '__main__':
    _main()
