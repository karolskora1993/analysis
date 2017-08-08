import pandas as pd
import os


BLOCKS_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/bloki_v2/'
ALL_DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v3.csv'

OUTPUT_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/statystyki_v2/'


def calculateStats(dataFrame):
    stats = {}
    for label in dataFrame.keys():
        stats[label] = dataFrame[label].describe()
    return stats


def calculateCorr(dataFrame):
    corr = {}
    for method in ['pearson']:
        corr[method] = dataFrame.corr(method=method)
    return corr

def loadDataFrame(file):
    df = pd.read_csv(file)
    return df


def saveStatsToFile(stats, fileName):
    outputFullPath = OUTPUT_PATH + fileName + ''
    with open(outputFullPath, 'w') as file:
        for label in stats.keys():
            file.write('Zmienna: ' + label + '\n')
            file.write(stats[label].to_string())
            file.write('\n\n\n')


def saveCorrToFile(corr, fileName):
    for key in corr:
        outputFullPath = OUTPUT_PATH + fileName + key + '.csv'
        corr[key].to_csv(outputFullPath)


def main():
    for csv_file in [file for file in os.listdir(BLOCKS_PATH) if file.endswith('.csv')]:
        fullPath = BLOCKS_PATH + csv_file
        df = loadDataFrame(fullPath)
        stats = calculateStats(df)
        saveStatsToFile(stats, csv_file)
        print('file {0} saved'.format(csv_file))

    df = loadDataFrame(ALL_DATA_PATH)
    print('data loaded')
    corr = calculateCorr(df)
    print('corr calculated')
    saveCorrToFile(corr, "all_corr")


main()
