import pandas as pd

DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/'


def fromCSV(inputPath, outputPath):

    df = pd.read_csv(inputPath+'.csv')
    print('data loaded')
    df.to_excel(outputPath+'.xlsx')
    print('data saved')

def fromExcel(inputPath, sheetName,  outputPath):

    df = pd.read_excel(inputPath+'.xlsx')
    print('data loaded')
    df.to_csv(outputPath+'.csv')
    print('data saved')


if __name__ == '__main__':
    inputPath = DEST_PATH + 'VRM-Próbki-60s'
    outputPath = DEST_PATH + 'VRM-Próbki-60s'
    fromExcel(inputPath, 'Arkusz1', outputPath)


