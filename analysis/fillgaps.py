import pandas as pd
import numpy as np
from analysis import blocks

LOAD_PATH = blocks.HOME_PATH + '/Dokumenty/analysis/data/csv_all_v4.csv'
SAVE_PATH = blocks.HOME_PATH + '/Dokumenty/analysis/data/csv_all_v5.csv'


def main():
    df = pd.read_csv(LOAD_PATH, index_col=0)
    nan = df.isnull().any().any()
    print('Any NaN: {0}'.format(nan))
    if nan:
        df.interpolate(method='linear', inplace=True)
        print('interpolated')
        nan = df.isnull().any().any()
        print('Any NaN: {0}'.format(nan))
        df.to_csv(LOAD_PATH)
        print('saved')


if __name__ == '__main__':
    main()
