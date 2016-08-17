% write data to file_name
% use binary format

function WriteData2File(file_name, data, precision)
    file_id = fopen(file_name, 'w');
    fwrite(file_id, data, precision);
    fclose(file_id);
end