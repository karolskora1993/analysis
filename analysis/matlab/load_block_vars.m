
%% Import the data
function [in,control,out] = load_block_vars(file_path, block_name)
    [~, ~, raw] = xlsread(file_path,block_name);
    raw = raw(2:end,2:4);
    stringVectors = string(raw(:,[1,2,3]));
    stringVectors(ismissing(stringVectors)) = '';

    in = stringVectors(:,1);
    control = stringVectors(:,2);
    out = stringVectors(:,3);
end
