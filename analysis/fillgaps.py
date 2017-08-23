import pandas as pd
import numpy as np
from analysis import blocks


def main():
    for block_name in blocks.names:
        print(block_name)
        df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv', index_col=0)
        nan = df.isnull().any().any()
        print('Any NaN: {0}'.format(nan))
        if nan:
            df.interpolate(method='linear', inplace=True)
            print('interpolated')
            nan = df.isnull().any().any()
            print('Any NaN: {0}'.format(nan))
            df.to_csv(blocks.SAVE_PATH + block_name + '.csv')
            print('saved')


if __name__ == '__main__':
    main()


