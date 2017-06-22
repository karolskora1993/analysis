import pandas as pd
from analysis import filesToConcat

DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/'

def main():
    dataframes = []
    for file in filesToConcat.files:
        dataframes.append(pd.read_csv(file['destinationName'] + '.csv'))

    resultDf = pd.concat(dataframes)
    resultDf.to_csv(DEST_PATH + 'VRM-Próbki-1s-1min.csv')


def append():
    files = [
        DEST_PATH + 'csv_all',
        DEST_PATH + 'VRM-Próbki-60s',
        DEST_PATH + 'csv_all_v2'

    ]

    allData = pd.read_csv(files[0]+'.csv')
    print('read all data')
    print("columns:{0}".format(len(allData.columns)))
    dataToAppend = pd.read_csv(files[1]+'.csv',)
    print('read data to append')
    print("columns:{0}".format(len(dataToAppend.columns)))
    dataToAppend.pop('timestamp')
    print("columnsa fter pop:{0}".format(len(dataToAppend.columns)))
    allData = allData.join(dataToAppend)
    print("columns:{0}".format(len(allData.columns)))

    allData.to_csv(files[2]+'.csv')
    print("save to file")

if __name__ == '__main__':
    append()

