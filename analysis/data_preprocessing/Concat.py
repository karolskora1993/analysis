import pandas as pd
from analysis import filesToConcat

DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/'

def concat():
    dataframes = []
    for file in filesToConcat.files:
        dataframes.append(pd.read_csv(file['destinationName'] + '.csv'))

    resultDf = pd.concat(dataframes)
    resultDf.to_csv(DEST_PATH + 'VRM-Próbki-1s-1min.csv')


def append():

    all_data_file = DEST_PATH + 'csv_all_v2'
    all_data__dest_file = DEST_PATH + 'csv_all_v3'

    files_to_append = [
        DEST_PATH + 'VRM-Brakujące-Próbki-15min-1min',
        DEST_PATH + 'VRM-Brakujące-Próbki-70min-1min',
        DEST_PATH + 'VRM-Brakujące-Próbki-60s'
    ]

    all_data = pd.read_csv(all_data_file+'.csv')
    print('read all data')
    print("columns:{0}".format(len(all_data.columns)))

    for file_to_append in files_to_append:
        new_data = pd.read_csv(file_to_append + '.csv')
        print('read data to append')
        print("columns:{0}".format(len(new_data.columns)))
        try:
            new_data.pop('Timestamp')
        except KeyError:
            print("Timestamp not found")
        finally:
            all_data = all_data.merge(new_data)

    print("all columns after merge :{0}".format(len(all_data.columns)))

    all_data.to_csv(all_data__dest_file + '.csv', index=False)
    print("save to file")

if __name__ == '__main__':
    append()

