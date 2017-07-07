BLOCK_DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/bloki/';
BLOCKS = '/Users/apple/Desktop/mag/bloki_poprawione.xlsx';

BLOCK_NAMES = {'blok I', 'blok II', 'blok III', 'blok IV'};

for name = BLOCK_NAMES
    block_name = name{1};
    [in, control, out] = load_block_vars(BLOCKS, block_name);
    for i = 1:length(in)
        if contains(in{i}, '.')
                in{i} = strrep(in{i},'.','_');
        end
    end
    for i = 1:length(control)
        if contains(control{i}, '.')
                control{i} = strrep(control{i},'.','_');
        end
    end
    for i = 1:length(out)
        if contains(out{i}, '.')
                out{i} = strrep(out{i},'.','_');
        end
    end
    variable_names = {in{:}, control{:}, out{:}};
    block_data = importfile(strcat(BLOCK_DATA_PATH, strcat(block_name,'.csv')), variable_names);
    in_data = block_data(1:end, in);
    control_data = block_data(1:end, control);
    out_data = block_data(1:end, out);
    train_length = length(in_data) * 7/8
%     ann = fitnet(30);
%     ann = train(in_data, out_data);
    
end
    

