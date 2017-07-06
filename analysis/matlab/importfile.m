function output = importfile(filename, startRow, endRow)
    delimiter = ',';
    if nargin<=2
        startRow = 2;
        endRow = inf;
    end
    
    formatSpec = '%*s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    
    for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
        end
    end
    
    fclose(fileID);
    output = table(dataArray{1:end-1}, 'VariableNames', {'NKZG_AN1_METAN','S0FK318DACAPV','S0FC027PIDAPV','S0PC025PIDAPV','S0F026DACAPV','S0T001_1DACAPV','S0FK319DACAPV','S0PC318PIDAPV','S0FC301PIDAPV','S0FC301PIDASP','S0FC301PIDAOP','S0PC304PIDAPV','S0RFC322PIDAPV','S0RFC316CTLALGOPV','S0FC316PIDAPV','S0PC304PIDASP','S0PC304PIDAOP','S0PC027PIDAPV','S0T302_2DACAPV','S0T302_2ADACAPV','S0A301DACAPV','S0T302_S14DACAPV','S0T302_S15DACAPV','S0T302_S16DACAPV','S0T302_S17DACAPV','S0T302_S18DACAPV','S0T302_S19DACAPV','S0P304_2DACAPV','S0T301_6DACAPV','S0PZ304DACAPV'});
end