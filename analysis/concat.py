import pandas as pd
from analysis import filesToConcat

DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/'

def main():
    dataframes = []
    for file in filesToConcat.files:
        dataframes.append(pd.read_csv(file['destinationName'] + '.csv'))

    resultDf = pd.concat(dataframes)
    resultDf.to_csv(DEST_PATH + 'VRM-Próbki-1s-1min.csv')


if __name__ == '__main__':
    main()


