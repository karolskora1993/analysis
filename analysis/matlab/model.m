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
    x = tonndata(table2array(in_data),false,false);
    y = tonndata(table2array(out_data),false,false);
    initSize = 4;
    endSize = length(in) + 10;
    if (length(in)-10 > 4); initSize = length(in)-10; end
    best_mse = realmax('single');
    best_net = fitnet(1);
    for hiddenSizes = initSize:2:endSize
        net = fitnet(20);
        net.divideParam.trainRatio = 87/100;
        net.divideParam.valRatio = 0/100;
        net.divideParam.testRatio = 13/100;
        net = train(net, x, y);
        y_out = net(x);
        mse = evaluate(y,y_out, net);
        if mse< best_mse
            best_net = net;
        end
    end
    filename = strcat(name,'.mat');
    save(filename,best_net);
end
