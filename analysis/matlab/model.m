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
    data_size = size(in_data);
    train_length = data_size(1) * 7/8;
    ann = fitnet(30);
    x = table2cell(in_data(1:train_length, :))';
    y = table2cell(out_data(1:train_length, :))';
    ann = train(ann, x, y);
    evaluate(in_data(train_length+1:end, :), out_data(train_length+1:end, :), ann)
end
