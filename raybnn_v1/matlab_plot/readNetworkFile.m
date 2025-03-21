function [WValues,WRowIdxCSR,WColIdx,H,A,B,C,D,E,glia_pos,neuron_pos,neuron_idx,net_data] = readNetworkFile(filename)




WValues = [];
WRowIdxCSR = [];
WColIdx = [];

H = [];
A = [];
B = [];
C = [];
D = [];
E = [];


glia_pos = [];
neuron_pos = [];
neuron_idx = [];

net_data = [];

fid = fopen(filename);
% skip 2 lines
for i=1:13
    str = fgetl(fid);
    strcell = split(str,",");
    switch i
        case 1
            WValues = str2double(strcell);
        case 2
            WRowIdxCSR = str2double(strcell);
        case 3
            WColIdx = str2double(strcell);
        case 4
            H = str2double(strcell);
        case 5
            A = str2double(strcell);
        case 6
            B = str2double(strcell);
        case 7
            C = str2double(strcell);
        case 8
            D = str2double(strcell);
        case 9
            E = str2double(strcell);
        case 10
            glia_pos = str2double(strcell);
        case 11
            neuron_pos = str2double(strcell);
        case 12
            neuron_idx = str2double(strcell);
        case 13
            net_data = str2double(strcell);
        otherwise
            disp('other value')
    end

end
fclose(fid);





end

