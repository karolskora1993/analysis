function output = importfile(filename, variable_names)
    delimiter = ',';
    startRow = 2;
    endRow = inf;
    
    format = '%*s';
    for i = 1:length(variable_names)
        format = strcat(format,'%f');
        if contains(variable_names{i}, '.')
            variable_names{i} = strrep(variable_names{i},'.','_');
        end
    end
    format = strcat(format, '%[^\n\r]');
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, format, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, format, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
        end
    end
    
    fclose(fileID);
    output = table(dataArray{1:end-1}, 'VariableNames', variable_names);
end