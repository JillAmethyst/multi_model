file = fopen('YArr.dat','wt');
for i = 1:33
    for j = 1:10000
        fwrite(file,YArr(j,i),'double');
    end
end
for i = 1:33
    fwrite(file,ttls(i),'int');
end
fclose(file);