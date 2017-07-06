BLOCK_DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/bloki/';
BLOCKS = '/Users/apple/Desktop/mag/bloki_poprawione.xlsx';

BLOCK_NAMES = {'blok I', 'blok II', 'blok III', 'blok IV'};

for name = BLOCK_NAMES
    block_name = name{1};
    [in, control, out] = load_block_vars(BLOCKS, block_name);
    block_data = importfile(strcat(BLOCK_DATA_PATH, strcat(block_name,'.csv')));
    in_data = block_data(1:end, in);
    control_data = block_data(1:end, control);
    out_data = block_data(1:end, out);
end
    

