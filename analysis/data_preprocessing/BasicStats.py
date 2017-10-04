import pandas as pd
import os


BLOCKS_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/bloki_v2/'
ALL_DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v3.csv'

OUTPUT_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/statystyki_v2/'


def calculate_stats(df):
    stats = {}
    for label in df.keys():
        stats[label] = df[label].describe()
    return stats


def calculate_corr(df):
    corr = {}
    for method in ['pearson']:
        corr[method] = df.corr(method=method)
    return corr

def _load_data_frame(file):
    df = pd.read_csv(file)
    return df


def _save_stats_to_file(stats, file_name):
    outputFullPath = OUTPUT_PATH + file_name + ''
    with open(outputFullPath, 'w') as file:
        for label in stats.keys():
            file.write('Zmienna: ' + label + '\n')
            file.write(stats[label].to_string())
            file.write('\n\n\n')


def _save_corr_to_file(corr, file_name):
    for key in corr:
        outputFullPath = OUTPUT_PATH + file_name + key + '.csv'
        corr[key].to_csv(outputFullPath)


def _main():
    for csv_file in [file for file in os.listdir(BLOCKS_PATH) if file.endswith('.csv')]:
        fullPath = BLOCKS_PATH + csv_file
        df = _load_data_frame(fullPath)
        stats = calculate_stats(df)
        _save_stats_to_file(stats, csv_file)
        print('file {0} saved'.format(csv_file))

    df = _load_data_frame(ALL_DATA_PATH)
    print('data loaded')
    corr = calculate_corr(df)
    print('corr calculated')
    _save_corr_to_file(corr, "all_corr")


if __name__ == '__main__':
    _main()
