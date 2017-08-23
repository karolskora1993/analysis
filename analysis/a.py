import pandas as pd
import pickle

DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v3.csv'
IDX_PATH = '/Users/apple/Desktop/mag/przestoje_polowa_v2.p'
SAVE_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v4.csv'


def main():
    df = pd.read_csv(DATA_PATH, index_col=0)
    idx = pickle.load(open(IDX_PATH, 'rb'))
    for i in idx:
        first = i['first_idx']
        last = i['last_idx']
        to_drop = [i for i in range(first - 1, last)]
        df.drop(to_drop, inplace=True)
    df = pd.DataFrame(df)
    df.to_csv(SAVE_PATH)


if __name__ == '__main__':
    main()
