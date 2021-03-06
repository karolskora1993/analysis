
%% Import the data
function [in,control,out] = load_block_vars(file_path, block_name)
    [~, ~, raw] = xlsread(file_path,block_name);
    raw = raw(2:end,2:4);
    stringVectors = string(raw(:,[1,2,3]));
    stringVectors(ismissing(stringVectors)) = '#';

    in_all = cellstr(stringVectors(:,1));
    control_all = cellstr(stringVectors(:,2));
    out_all = cellstr(stringVectors(:,3));
    [I, ~] = find(cellfun(@(s) contains(s, '#'), in_all));
    in_all(I, :) = [];
    [I, ~] = find(cellfun(@(s) contains(s, '#'), control_all));
    control_all(I, :) = [];    
    [I, ~] = find(cellfun(@(s) contains(s, '#'), out_all));
    out_all(I, :) = [];
    
    in = in_all;
    control = control_all;
    out = out_all;
    
    
end
