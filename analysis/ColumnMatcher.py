import pandas as pd

DATA_PATH = DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v2'
EX_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/odpowiedniki'

def main():
    df = pd.read_csv(DATA_PATH+'.csv')
    print("read data")
    columns = df.columns

    df = pd.read_excel(EX_PATH+'.xlsx')
    dataToMatch = df['schemat']

    for schemeItem in dataToMatch:
        for dataItem in columns:
            if schemeItem in dataItem or dataItem in schemeItem:
                print("Zmienna w schemacie: {0} odpowiednik w zbiorze: {1}".format(schemeItem, dataItem))
        print('--------------------')


if __name__ == '__main__':
    main()


