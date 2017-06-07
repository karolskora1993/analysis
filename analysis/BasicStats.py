import pandas as pd
import os


PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/'
OUTPUT_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/stats/'


def calculateStats(dataFrame):
    stats = {}
    for label in dataFrame.keys():
        stats[label] = dataFrame[label].describe()
    return stats


def calculateCorr(dataFrame):
    corr = {}
    for method in ['pearson', 'kendall', 'spearman']:
        corr[method] = dataFrame.corr(method=method)
    return corr

def loadDataFrame(file):
    df = pd.read_csv(file)
    return df


def saveStatsToFile(stats, fileName):
    outputFullPath = OUTPUT_PATH + fileName + '_basicstats'
    with open(outputFullPath, 'w') as file:
        for label in stats.keys():
            file.write('Zmienna: ' + label + '\n')
            file.write(stats[label].to_string())
            file.write('\n\n\n')


def saveCorrToFile(corr, fileName):
    outputFullPath = OUTPUT_PATH + fileName + '_corr'
    with open(outputFullPath, 'w') as file:
        for method in corr.keys():
            file.write('Metoda: ' + method + '\n')
            file.write(corr[method].to_string())
            file.write('\n\n\n')


def main():
    for csv_file in [file for file in os.listdir(PATH) if file.endswith('.csv')]:
        fullPath = PATH + csv_file
        df = loadDataFrame(fullPath)
        stats = calculateStats(df)
        saveStatsToFile(stats, csv_file)
        corr = calculateCorr(df)
        saveCorrToFile(corr, csv_file)


main()
