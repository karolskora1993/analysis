BLOCK_DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/bloki/';
BLOCKS = '/Users/apple/Desktop/mag/bloki_poprawione.xlsx';

BLOCK_NAMES = {'blok I', 'blok II', 'blok III', 'blok IV'};

for name = BLOCK_NAMES
    block_name = name{1};
    [in, control, out] = load_block_vars(BLOCKS, block_name);
    variable_names = {in{:}, control{:}, out{:}};
    for i = length(variable_names)
        if contains(variable_names{i}, '.')
            variable_names{i} = strrep(variable_names{i},'.','_');
        end
    end
    block_data = importfile(strcat(BLOCK_DATA_PATH, strcat(block_name,'.csv')), variable_names);
    in_data = block_data(1:end, in);
    control_data = block_data(1:end, control);
    out_data = block_data(1:end, out);
    train_length = length(in_data) * 7/8
%     ann = fitnet(30);
%     ann = train(in_data, out_data);
    
end
    

