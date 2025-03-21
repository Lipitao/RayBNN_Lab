function COO = CSRtoCOO(CSR)

COO = [];
j = 0;
for i=1:(size(CSR,1)-1)
    num = CSR(i+1) - CSR(i);
    if num > 0
        newCOO = j*ones(num,1);
        COO = [COO; newCOO];
    end
    j = j + 1;
end


end

