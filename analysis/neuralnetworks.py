import pandas as pd
from analysis import blocks

COL_NAMES = ['in', 'control', 'out']


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv')
    print('data loaded')
    return  df


def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def main():
    blocks_vars = load_block_vars()
    for block_name in blocks.names:
        df = load_data(block_name)
        vars = blocks_vars[block_name][COL_NAMES]
        print(vars)

if __name__ == '__main__':
    main()
