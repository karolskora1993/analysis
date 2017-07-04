import pandas as pd

BLOCKS_PATH = '/Users/apple/Desktop/mag/bloki_poprawione.xlsx'
DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v2.csv'
VARS = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/kolumny.xlsx'
DEST_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/bloki/'

class DataDivider:
    
    def __init__(self, blocks_path, block_names, data_path, destination_path):
        self.blocks_data = pd.read_excel(blocks_path, sheetname=block_names)
        self.blocks_names = block_names
        self.data = pd.read_csv(data_path)
        self.dest_path = destination_path
        print('data loaded')

    def find_mismatch(self):
        variables = self.data.columns.values
        mismatches = []
        for key, value in self.blocks_data.items():
            for name in value['zmienne']:
                if '#' in name:
                    continue
                match = None
                for var in variables:
                    if name == var:
                        match = var
                if match is None:
                    print('Nie znaleziono pasujacego dla zmiennej {0}'.format(name))
                    mismatches.append(name)
        if len(mismatches)>0:
            print("Nie dopasowane:{0}".format(mismatches))
        return mismatches if len(mismatches) > 0 else None

    def divide_into_groups(self):
        if not self.find_mismatch():
            for key in self.blocks_data.keys():
                block_vars = []
                for var_name in self.blocks_data[key]['zmienne']:
                    if not '#' in var_name:
                        block_vars.append(var_name)
                block_df = self.data[block_vars]
                block_df.to_csv(self.dest_path+key+'.csv')
                print('{0} zapisany'.format(key))

def main():
    data_divider = DataDivider(BLOCKS_PATH, ['blok I', 'blok II', 'blok III', 'blok IV'], DATA_PATH, DEST_PATH)
    data_divider.divide_into_groups()

if __name__ == '__main__':
    main()
