import pandas as pd

DATA_PATH = DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v2'
EX_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/odpowiedniki'

def match(columns, data_to_match):
    matches = []
    for scheme_item in data_to_match:
        for data_item in columns:
            if scheme_item in data_item or data_item in scheme_item:
                matches.append((scheme_item, data_item))
    return matches


def _main():
    df = pd.read_csv(DATA_PATH+'.csv')
    print("read data")
    columns = df.columns
    df = pd.read_excel(EX_PATH+'.xlsx')
    data_to_match = df['schemat']
    matches = match(columns, data_to_match)
    for scheme_item, data_item in matches:
        print("Zmienna w schemacie: {0} odpowiednik w zbiorze: {1}".format(scheme_item, data_item))


if __name__ == '__main__':
    _main()


