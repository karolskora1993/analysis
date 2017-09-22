import pandas as pd
import pickle

DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v3.csv'
IDX_PATH = '/Users/apple/Desktop/mag/przestoje_polowa_v2.p'
SAVE_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v4.csv'


def remove(dataframe, indexes, save_path = None):
    for i in indexes:
        first = i['first_idx']
        last = i['last_idx']
        to_drop = [i for i in range(first - 1, last)]
        dataframe.drop(to_drop, inplace=True)
    df = pd.DataFrame(dataframe)
    if save_path:
        df.to_csv(SAVE_PATH)

def main():
    df = pd.read_csv(DATA_PATH, index_col=0)
    idx = pickle.load(open(IDX_PATH, 'rb'))
    remove(df, idx, SAVE_PATH)


if __name__ == '__main__':
    main()
